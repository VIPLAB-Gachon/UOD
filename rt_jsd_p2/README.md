# Ours-P2 · RT-JSD on M3FD

Ours-P2 uses a shared YOLOv5s backbone and distribution head, separate RGB and
thermal necks, and a P2 detection level. Training uses selective cross-modal JSD
on box edge distributions and RGB confidence ranking for People, Lamp and
Motorcycle. The detector runs on RGB alone, thermal alone, or both inputs with
late fusion of candidate boxes.

[`checkpoints/ours-p2-e184.pt`](checkpoints/ours-p2-e184.pt) contains the original
FP32 EMA tensors and the settings needed to reconstruct the detector.
It corresponds to epoch 184, selected by Validation RGBT AP50–95 with linear
Soft-NMS. [The saved evaluation](reference/epoch184_metrics.json) records
both Soft-NMS and CP-Cluster metrics and the SHA-256 of the original training
checkpoint. M3FD Test CP-Cluster AP50 / AP50–95 was 87.93 / 58.82 for RGB,
86.85 / 56.69 for thermal and 88.96 / 59.84 for RGBT.
The export contains inference weights. Resuming training requires a full
checkpoint with optimizer, scheduler and scaler state.

## Evaluate

From `rt_jsd_p2/`, use Python 3.11 and the pinned [dependencies](../rt_jsd/requirements.txt)
from the lab's RT-JSD environment. The original run used PyTorch 2.4.1 with
CUDA 11.8. Obtain aligned M3FD images and YOLO labels separately, and prepare
the lab's fixed 3,360/420/420 split:

```bash
python ../rt_jsd/tools/make_split_manifests.py \
  --data-root /path/to/M3FD --output data/m3fd_uod_811_seed2026
python tools/evaluate.py --manifest data/m3fd_uod_811_seed2026/test.jsonl --device cuda:0
```

The evaluator uses linear Soft-NMS by default and supports CPU. For the CP-Cluster
numbers, pass `--cp-loader` pointing to the lab's trusted
`reports/baselines_cp_cluster_test_20261111/tools/evaluate_cp.py` in the original
workspace. The loader checks its existing local CP build. CP binaries are
environment-specific. This snapshot shares the pinned YOLOv5 source already
tracked under `rt_jsd/runtime/third_party/yolov5`.

## Train

Use the [official YOLOv5s v7.0 COCO checkpoint](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt).
Use the lab's original common initialization if available, or generate a new
model-only initialization from the pinned COCO weights. The commands below
assume it has been downloaded to `weights/yolov5s.pt`.

```bash
python tools/prepare_initial.py --weights weights/yolov5s.pt --output weights/common_initial.pt
python tools/configure.py --weights weights/yolov5s.pt \
  --initial-checkpoint weights/common_initial.pt --manifest-root data/m3fd_uod_811_seed2026
torchrun --standalone --nproc_per_node=2 tools/train_ddp.py --config config.json
```

The template records the original 640 px, seed 2026, 200-epoch DDP recipe.
Generated configurations and experiment outputs stay local. The frozen trainer
and loss modules are from the completed November 16 run. The vendored YOLOv5
import path points to the copy in the lab repository.
