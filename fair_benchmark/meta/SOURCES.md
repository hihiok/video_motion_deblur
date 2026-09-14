# Frozen test metadata and protocol sources

The sequence lists are copied verbatim from the authors' BSSTNet repository. They are data identifiers and counts, not a substitute for checking the local data contents.

| File / fact | Primary source | Git blob SHA |
|---|---|---|
| `DVD_test.txt`: 10 sequences, 1000 frames, native dimensions | https://github.com/huicongzhang/BSSTNet/blob/main/basicsr/data/meta_info/DVD_test.txt | `d79d081ff75d3c717d78ea4399f21b5312f10f00` |
| `GoPro_test.txt`: 11 sequences, 1111 frames | https://github.com/huicongzhang/BSSTNet/blob/main/basicsr/data/meta_info/GoPro_test.txt | `febbe03e7b96df7b5bd70e4728363523201d8544` |
| BSD test uses 150 frames per sequence and 480x640 RGB | https://github.com/zzh-tech/ESTRNN/blob/master/train/test.py | `22e9cc714c4b30d73bc9e21e9f25cc2a439aeaa5` |
| BSD three exposure configurations | https://github.com/zzh-tech/ESTRNN | Original BSD authors |
| DSTNet GOPRO: 30 frames / no overlap | https://github.com/xuboming8/DSTNet/blob/main/options/test/Deblur/test_Deblur_GOPRO.yml | `867df1facf4367ee5cb69fa5f64a1ed11dadcdb5` |
| DSTNet public BSD test YAML points to **1ms8ms**, not proof that BSD.pth is 3ms24ms | https://github.com/xuboming8/DSTNet/blob/main/options/test/Deblur/test_Deblur_BSD.yml | `a8203619c89583a6787385f7f1ad2f77b3e36c7c` |
| Shift-Net+ official inference: FP16, skips temporal boundaries / incomplete chunks, float-domain metrics | https://github.com/dasongli1/Shift-Net/blob/main/inference/test_deblur.py | `2c94b1475b790bcb88174d6297fab690cde1efad` |
| BSST GoPro: 48 frames, 256 spatial patches, RGB PSNR with crop border 0 | https://github.com/huicongzhang/BSSTNet/blob/main/options/test/BSST/gopro_BSST.yml | `24c9800ce61607640a1256ed433d70e04f3d8a20` |
| RVRT official testing: quantized RGB8 metric, sequence-mean summary, optional tiling | https://github.com/JingyunLiang/RVRT/blob/main/main_test_rvrt.py | `75032b40eae0bf9be8487a46c3e23137daf7420a` |
| Turtle public GoPro config has `num_frames_tocache: 3`; preserve checkpoint-matched configuration | https://github.com/Ascend-Research/Turtle/blob/main/options/Turtle_Deblur_Gopro.yml | `8a888080b9e6727fb763d9c7468a0642c34c9492` |

The unified benchmark is explicitly a new all-frame, native-size RGB8 protocol. Its FP32 precision, completion of omitted temporal boundaries, and fixed method-specific tiling/chunk policies must not be described as exact reproduction of every paper. A separate explicit-frame official-output anchor can document equivalence or its limits. No published PSNR number is inserted into the measured table.

BSD 3ms24ms uses the original explicit test split: 20 sequences x 150 frames. Do not replace it with 20 sequences of 100 training/validation frames, another exposure variant, or RAW data.
