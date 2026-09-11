"""Pinned upstream loader and width-compressed Shift-Net-s, preserving all stages.

Upstream stays untouched. Only four constructor constants are parameterized in
memory. Default model remains checkpoint-compatible; compressed models use exact
shape weight transfer only (no misleading arbitrary channel slicing).
"""
import hashlib
from pathlib import Path
import types
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

UPSTREAM_COMMIT = '450a4f246dedccd306aa0bc02d615d797874e1ce'
SOURCE_SHA256 = 'b2b86200f00d37ce69dfa43e2f4f86696d6f8f46d715e6b131f0de4b120c3d4c'
VARIANTS = {'teacher': (64, 14, 4), 'quality': (32, 9, 4), 'compact': (32, 8, 3)}


def load_upstream(root, widths):
    path = Path(root) / 'basicsr/models/archs/gshift_deblur2.py'
    source = path.read_text()
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError('Upstream source mismatch; use pinned commit, do not patch upstream')
    core, spatial, increment = widths
    for old, new in [('self.n_feats2 = 64', f'self.n_feats2 = {core}'),
                     ('self.n_feats0 = 14', f'self.n_feats0 = {spatial}'),
                     ('n_feat0 = 14', f'n_feat0 = {spatial}'),
                     ('scale_unetfeats = 4\n', f'scale_unetfeats = {increment}\n')]:
        if source.count(old) != 1:
            raise RuntimeError(f'Expected one constructor anchor: {old}')
        source = source.replace(old, new)
    module = types.ModuleType('pinned_gshift_deblur2')
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


class Bypass(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class ShiftModel(nn.Module):
    def __init__(self, upstream, variant='quality', activation_checkpointing=False, widths=None):
        super().__init__()
        self.variant = variant
        self.widths = tuple(widths or VARIANTS[variant])
        mod = load_upstream(upstream, self.widths)
        self.net = mod.GShiftNet(future_frames=2, past_frames=2)
        # These four modules are instantiated upstream but never used in forward.
        for name in ('orb4', 'orb5', 'rorb4', 'rorb5', 'lrelu'):
            delattr(self.net, name)
        if variant != 'teacher':
            # Drop repeated spatial UNets; keep one before and one after fusion.
            for name in ('orb2', 'orb3', 'rorb2', 'rorb3'):
                setattr(self.net, name, nn.Identity())
            # Keep one half-resolution GSTS group, two quarter-resolution groups.
            for level in ('encoder_level1', 'decoder_level1'):
                for suffix in ('_1', '_2'):
                    setattr(self.net.stage1, level + suffix, Bypass())
            for level in ('encoder_level2', 'decoder_level2'):
                setattr(self.net.stage1, level + '_2', Bypass())
                if variant == 'compact':
                    setattr(self.net.stage1, level + '_1', Bypass())
        self.activation_checkpointing = activation_checkpointing
        if activation_checkpointing:
            for child in self.net.modules():
                if isinstance(child, (mod.TFR_UNet, mod.Encoder_shift_block)):
                    original = child.forward
                    def wrapped(*args, _forward=original, **kwargs):
                        if self.training and torch.is_grad_enabled():
                            return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
                        return _forward(*args, **kwargs)
                    child.forward = wrapped

    def forward(self, x):
        if x.ndim != 5 or x.shape[0] != 1 or x.shape[1] < 5:
            raise ValueError('Expected [1,T>=5,3,H,W]; upstream supports batch=1 only')
        h, w = x.shape[-2:]
        if min(h, w) < 32:
            raise ValueError('Spatial shift requires dimensions >=32')
        pad_h, pad_w = (-h) % 4, (-w) % 4
        flat = F.pad(x[0], (0, pad_w, 0, pad_h), mode='replicate')
        return self.net(flat.unsqueeze(0))[..., :h, :w].unsqueeze(0)


def checkpoint_state(path):
    value = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('model', 'params_ema', 'params', 'state_dict'):
        if key in value and isinstance(value[key], dict):
            value = value[key]
            break
    result = {}
    for key, tensor in value.items():
        key = key.removeprefix('module.').removeprefix('net.')
        if key.split('.')[0] in ('orb4', 'orb5', 'rorb4', 'rorb5', 'lrelu'):
            continue
        result[key] = tensor
    return result


def load_teacher(model, path):
    model.net.load_state_dict(checkpoint_state(path), strict=True)


def initialize_student(student, teacher):
    source = teacher.net.state_dict()
    target = student.net.state_dict()
    exact, fresh = [], []
    for key, tensor in target.items():
        if key in source and source[key].shape == tensor.shape:
            target[key] = source[key].clone()
            exact.append(key)
        else:
            fresh.append(key)
    student.net.load_state_dict(target, strict=True)
    return {'exact_tensors': exact, 'fresh_tensors': fresh,
            'note': 'Width compression + distillation; not function-preserving pruning.'}
