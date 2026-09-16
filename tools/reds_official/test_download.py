#!/usr/bin/env python3
"""Offline regression tests. No network, credentials, GPU or large REDS files."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location('reds_download', Path(__file__).with_name('download.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def item(name='val_blur.zip', data=b'abcdefgh'):
    return {'name': name, 'size': len(data), 'algorithm': 'sha256', 'digest': hashlib.sha256(data).hexdigest()}


def zip_bytes(name='val/val_blur/000/00000000.png', payload=b'image'):
    b = io.BytesIO()
    with zipfile.ZipFile(b, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr(name, payload)
    return b.getvalue()


class Response:
    def __init__(self, data, status=200, headers=None):
        self.data, self.status_code = data, status
        self.headers = headers or {'Content-Length': str(len(data))}
        self.raw = io.BytesIO(data)
        self.url = 'https://huggingface.co/test'
        self.closed = False
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
    def close(self): self.closed = True
    def iter_content(self, n):
        for i in range(0, len(self.data), n): yield self.data[i:i+n]


class FakeClient:
    def __init__(self, payload, ignore_range=False, wrong_range=False):
        self.payload, self.calls = payload, []
        self.ignore_range, self.wrong_range = ignore_range, wrong_range
    def get(self, url, headers=None, stream=False):
        self.calls.append(headers)
        if headers and 'Range' in headers and not self.ignore_range:
            start, end = headers['Range'][6:].split('-')
            start, end = int(start), int(end) if end else len(self.payload)-1
            content = self.payload[start:end+1]
            return Response(content, 206, {'Content-Length': str(len(content)),
                'Content-Range': 'bytes %d-%d/%d' % (start + int(self.wrong_range), end, len(self.payload))})
        return Response(self.payload)


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()

    def test_normal_path(self):
        self.assertEqual(m.safe_target(self.root, 'train/train_blur/000/f.png'),
                         self.root/'train/train_blur/000/f.png')
    def test_unsafe_paths(self):
        for value in ('../x', '/x', 'C:/x', 'a\\b', 'a//b', './x', '', 'x/../../bad'):
            with self.subTest(value=value), self.assertRaises(m.Blocked): m.safe_parts(value)
    def test_existing_parent_symlink(self):
        (self.root/'link').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(m.Blocked): m.safe_target(self.root, 'link/f.png')
    def test_sha256(self):
        p=self.root/'file'; p.write_bytes(b'abcdefgh')
        self.assertTrue(m.hash_ok(p,item()))
        p.write_bytes(b'wrong'); self.assertFalse(m.hash_ok(p,item()))
    def test_git_blob_sha1(self):
        data=b'abc'; p=self.root/'README.md'; p.write_bytes(data)
        x={'name':'README.md','size':3,'algorithm':'git-sha1','digest':hashlib.sha1(b'blob 3\0abc').hexdigest()}
        self.assertTrue(m.hash_ok(p,x))
    def test_manifest_rejects_placeholder(self):
        class C:
            def json(self, url): return {'sha':m.REVISION,'siblings':[{'rfilename':'raw/REDS.tar.gz.00','size':125}]}
        with self.assertRaises(m.Blocked): m.get_manifest(C())
    def test_manifest_rejects_wrong_revision(self):
        class C:
            def json(self,url): return {'sha':'wrong','siblings':[]}
        with self.assertRaises(m.Blocked): m.get_manifest(C())
    def test_manifest_lfs(self):
        class C:
            def json(self,url):
                return {'sha':m.REVISION,'siblings':[{'rfilename':n,'size':m.GIB,'lfs':{'size':m.GIB,'sha256':'a'*64}} for n in m.CORE]}
        self.assertEqual(len(m.get_manifest(C())),4)
    def test_manifest_missing_hash(self):
        class C:
            def json(self,url): return {'sha':m.REVISION,'siblings':[{'rfilename':'train_blur.zip','size':m.GIB}]}
        with self.assertRaises(m.Blocked): m.get_manifest(C())
    def test_remote_zip_preflight_small_ranges(self):
        data=zip_bytes(); c=FakeClient(data)
        with zipfile.ZipFile(m.RemoteZip(c,item(data=data))) as z:
            self.assertEqual(m.zip_inventory(z)['images'],1)
        self.assertTrue(all('Range' in h for h in c.calls))
    def test_range_ignored_stops_without_body_download(self):
        data=zip_bytes(); r=m.RemoteZip(FakeClient(data,ignore_range=True),item(data=data))
        with self.assertRaises(m.Blocked): r.read(10)
    def test_range_wrong_offset(self):
        data=zip_bytes(); r=m.RemoteZip(FakeClient(data,wrong_range=True),item(data=data))
        with self.assertRaises(m.Blocked): r.read(10)
    def test_empty_archive_rejected(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z: z.writestr('REDS/','')
        with zipfile.ZipFile(io.BytesIO(b.getvalue())) as z, self.assertRaises(m.Blocked): m.zip_inventory(z)
    def test_symlink_in_zip_rejected(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:
            i=zipfile.ZipInfo('link.png'); i.create_system=3; i.external_attr=0o120777 << 16
            z.writestr(i,'target')
        with zipfile.ZipFile(io.BytesIO(b.getvalue())) as z, self.assertRaises(m.Blocked): m.zip_inventory(z)
    def test_download_and_skip(self):
        c=FakeClient(b'abcdefgh'); x=item()
        m.download_one(c,x,self.root)
        self.assertEqual((self.root/x['name']).read_bytes(),b'abcdefgh')
        m.download_one(c,x,self.root); self.assertEqual(len(c.calls),1)
    def test_resume(self):
        x=item(); (self.root/(x['name']+'.part')).write_bytes(b'abc')
        m.save_json(self.root/(x['name']+'.part.json'),x)
        c=FakeClient(b'abcdefgh'); m.download_one(c,x,self.root)
        self.assertEqual(c.calls[0]['Range'],'bytes=3-')
        self.assertEqual((self.root/x['name']).read_bytes(),b'abcdefgh')
    def test_complete_partial_promoted_without_network(self):
        x=item(); (self.root/(x['name']+'.part')).write_bytes(b'abcdefgh')
        m.save_json(self.root/(x['name']+'.part.json'),x)
        c=FakeClient(b'abcdefgh'); m.download_one(c,x,self.root); self.assertEqual(c.calls,[])
    def test_ignored_resume_preserves_bytes(self):
        x=item(); part=self.root/(x['name']+'.part'); part.write_bytes(b'abc')
        m.save_json(self.root/(x['name']+'.part.json'),x)
        with self.assertRaises(m.Blocked): m.download_one(FakeClient(b'abcdefgh',ignore_range=True),x,self.root)
        self.assertEqual(part.read_bytes(),b'abc')
    def test_unknown_partial_preserved(self):
        x=item(); part=self.root/(x['name']+'.part'); part.write_bytes(b'abc')
        with self.assertRaises(m.Blocked): m.download_one(FakeClient(b'abcdefgh'),x,self.root)
        self.assertEqual(part.read_bytes(),b'abc')
    def test_conflicting_final_preserved(self):
        x=item(); p=self.root/x['name'];p.write_bytes(b'original')
        with self.assertRaises(m.Blocked): m.download_one(FakeClient(b'abcdefgh'),x,self.root)
        self.assertEqual(p.read_bytes(),b'original')
    def test_download_hash_mismatch_keeps_part(self):
        x=item();
        with self.assertRaises(m.Blocked): m.download_one(FakeClient(b'XXXXXXXX'),x,self.root)
        self.assertFalse((self.root/x['name']).exists())
        self.assertEqual((self.root/(x['name']+'.part')).read_bytes(),b'XXXXXXXX')
    def test_extract_crc_resume_and_conflict(self):
        archive=self.root/'sample.zip'; archive.write_bytes(zip_bytes(payload=b'data'))
        out=self.root/'out';out.mkdir()
        m.extract_one(archive,out); m.extract_one(archive,out)
        p=out/'val/val_blur/000/00000000.png';p.write_bytes(b'xxxx')
        with self.assertRaises(m.Blocked):m.extract_one(archive,out)
        self.assertEqual(p.read_bytes(),b'xxxx')
    def test_existing_files_reduce_space_plan(self):
        name='train/train_blur/000/f.png'; target=self.root/name;target.parent.mkdir(parents=True);target.write_bytes(b'123')
        with zipfile.ZipFile(io.BytesIO(zip_bytes(name,b'123'))) as z:
            inv=m.zip_inventory(z,self.root)
        self.assertEqual(inv['uncompressed_bytes'],3);self.assertEqual(inv['additional_extract_bytes'],0)
    def test_png_header(self):
        p=self.root/'image.png';p.write_bytes(b'\x89PNG\r\n\x1a\n'+struct.pack('>I',13)+b'IHDR'+struct.pack('>II',1280,720))
        self.assertEqual(m.png_size(p),(1280,720))
    def test_disk_insufficient(self):
        raw=self.root/'raw';out=self.root/'out';raw.mkdir();out.mkdir()
        with patch.object(m.shutil,'disk_usage',return_value=m.shutil._ntuple_diskusage(100,90,10)):
            with self.assertRaises(m.Blocked):m.check_space(raw,out,[item()],100,1)
    def test_sample_pair_audit(self):
        from PIL import Image
        raw=self.root/'raw';out=self.root/'out';raw.mkdir();out.mkdir()
        buf=io.BytesIO();Image.new('RGB',(1280,720)).save(buf,format='PNG')
        for kind in ('blur','sharp'):
            p=raw/('val_'+kind+'.zip');p.write_bytes(zip_bytes('val/val_'+kind+'/000/00000000.png',buf.getvalue()))
            m.extract_one(p,out)
        result=m.audit_pairs(raw,out,expected={'val':(1,1)})
        self.assertEqual(result['val']['pairing'],'PASS')
    def test_verify_argument_and_no_netrc(self):
        c=m.Client(False)
        with patch.object(c.session,'get',return_value=Response(b'ok')) as call:
            with c.get('https://huggingface.co/test'): pass
        self.assertIs(call.call_args.kwargs['verify'],False)
        self.assertTrue(callable(c.session.auth))
    def test_proxy_exception_is_redacted(self):
        c=m.Client()
        with patch.object(c.session,'get',side_effect=m.requests.exceptions.ProxyError('secret-password')):
            with self.assertRaises(m.Blocked) as error: c.get('https://huggingface.co/test')
        self.assertNotIn('secret-password',str(error.exception))


if __name__=='__main__': unittest.main(verbosity=2)
