# RT-JSD with Score Calibration and Varifocal Loss

Snapshot of the implementation running on M3FD on 2026-11-04. It contains
the actual isolated runtime, rather than the older reference implementation.
The snapshot does not include images, trained checkpoints, optimizer states,
feature caches or experimental variants that are not enabled in this run.

## Architecture

- Shared YOLOv5l backbone and detection head; independent RGB and thermal necks.
- One detector checkpoint supports RGB-only or thermal-only inference, with the
  corresponding neck selected by the input modality.
- Four anchor-normalized boundary-distance distributions: 25 bins per side,
  support [-2, 4], spacing 0.25; boxes use the distribution expectations.
- Both modalities receive their supervised CIoU, objectness, classification and
  Distribution Focal Loss. There are six M3FD classes.
- Select one common positive candidate per GT **before** quality gating. The
  object teacher has the higher IoU; teacher IoU >= 0.7 and gap >= 0.1 are required.
  A side is transferred only when the object teacher has both lower GT-distance
  error and lower DFL. Teacher logits, selection and masks are detached.
- Student JSD gradients update the full detector. Shared parameters can still
  affect both modalities after an update; detach does not freeze the teacher's
  shared backbone/head globally. JSD alpha is zero for the first 3 epochs, then
  ramps to 0.1 over the next 5 epochs.
- A separate 27 -> 64 (ReLU) -> 6 residual score calibrator takes top-four
  probability statistics plus their mean for each side, objectness, and the six
  original class probabilities. It corrects logit(objectness * class probability).

## Implemented training protocol

The implemented pipeline is **two-stage**:

1. Train the paired detector for 200 epochs.
2. Freeze detector parameters and BatchNorm; train a fresh score calibrator for
   10 epochs with Varifocal Loss, then evaluate the selected pipeline on Test.

VFL supervises the calibrated final class score. Its positive target is detached
GT IoU in the assigned GT class. Negative weighting is 0.75 * p^2. The current
calibration also retains the high-IoU, same-class, unassigned-candidate ignore rule
(IoU >= 0.75). It is not an unconditional background suppression objective.

The proposed end-to-end joint training, S/X variants and parameter-reduction
designs discussed later are **not implemented by this snapshot**.

| Item | Setting |
|---|---|
| Data | M3FD, fixed own 8:1:1 split: 3360/420/420 pairs |
| Class order | People, Car, Bus, Lamp, Motorcycle, Truck |
| Input | 640 x 640 bilinear letterbox; padding 114/255 |
| Thermal | uint8 grayscale /255, repeated to 3 channels at model input |
| Detector | COCO-initialized YOLOv5l, NFE disabled |
| Global paired batch | 16 = 8 pairs/GPU x 2 GPUs, accumulation 1 |
| Optimizer | SGD, Nesterov, LR 0.01, momentum 0.937, actual decay 0.0005 |
| Schedule | 200-epoch linear decay, lrf 0.01, warmup 3 epochs |
| Augmentation | Paired Mosaic; flip 0.5, scale +/-0.2, translation +/-0.05 |
| Mosaic closure | Epochs 191-200 |
| Numerics | AMP, EMA, deterministic mode, clip norm 10 |
| Calibrator | Frozen detector; 10 epochs, Adam, LR 0.001, batch 65,536 candidate rows |
| Primary Best | Validation RGBT AP50-95; earliest tie |
| Additional detector Bests | RGBT AP50, RGB/T mean AP50, RGB/T mean AP50-95 |
| Postprocessing | Class-aware linear Soft-NMS, score 0.001, IoU 0.6, max detections 300 |
| RGBT | Union RGB and thermal candidates followed by Soft-NMS |

Primary detector and calibrator selection currently use **RGBT AP50-95**.
The live viewer reports the latest validation, RGBT AP50 Best and RGBT AP50-95
Best separately. Each table uses one epoch for all modes and all columns.
This is not a completed multi-seed experiment or a claimed published benchmark.

## Local setup and data

The reference runtime uses Python 3.11.9, PyTorch 2.4.1+cu118 and
torchvision 0.19.1+cu118 on RTX A5000 GPUs. Install the matching PyTorch build
and the packages listed in `requirements.txt` in an isolated environment.
`environment_reference.json` records the installed reference versions.
From the repository root, using Python 3.11.9:

```bash
mkdir -p "$PWD/.cache/tmp" "$PWD/.cache/matplotlib" "$PWD/.cache/xdg" "$PWD/.cache/yolo"
export TMPDIR="$PWD/.cache/tmp"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache/xdg"
export YOLOV5_CONFIG_DIR="$PWD/.cache/yolo"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Keep the checkout under `/data` on the reference server so these temporary files
and caches also use `/data`. Set these environment variables in each new shell
before running the tools.

The vendored YOLOv5 source is pinned to commit
`915bbf294bb74c859f0b41f1c23bc395014ea679`. Its original license is preserved
in `runtime/third_party/yolov5/LICENSE`.

Obtain M3FD and YOLO-format labels separately. With aligned files in
`/path/to/M3FD/{vi,ir,labels}`, materialize the **committed sample IDs**:

```bash
python tools/make_split_manifests.py --data-root /path/to/M3FD
```

The split IDs are included under `splits/m3fd_811_seed2026`. This is our own
fixed split, not an official M3FD split or the exact split of another paper.
The command writes `data/m3fd_uod_811_seed2026/{train,val,test}.jsonl` by default;
custom manifest locations must retain the directory name `m3fd_uod_811_seed2026`.
The loader checks label range, dimensions and geometric alignment assumptions.
The full original duplicate-audit splitter is retained in `tools/prepare_split.py`
for use with an audited source manifest; it is not needed to reuse these IDs.

Provide a trusted official COCO `yolov5l.pt`. Reuse the original common initial
checkpoint if available, or construct a fresh CPU initial state:

```bash
python tools/prepare_initial.py \
  --weights /path/to/yolov5l.pt \
  --output "$PWD/weights/common_initial_continuous_nc6.pt" \
  --seed 2026 --threads 2
```

This CPU-only command creates a six-class continuous-box checkpoint before the
25-bin distribution head is installed. It disables NFE, separates the two necks,
records `epoch=-1` and `global_update=0`, and refuses to replace an existing file.
Regenerating the initial state does not guarantee the original checkpoint's
file hash or bitwise identity. Use one fixed initial state for matched GT-only
and JSD experiments, and record its hash. No existing M3FD-trained weights are
used as the initial detector in this protocol.

Resolve machine-local paths without committing data or weights:

```bash
python tools/configure.py \
  --weights /path/to/yolov5l.pt \
  --initial-checkpoint "$PWD/weights/common_initial_continuous_nc6.pt" \
  --manifest-root "$PWD/data/m3fd_uod_811_seed2026"
```

This generates an ignored `config.json` at the repository root. Do not edit it
after starting a run: checkpoint and cache identities include its SHA256.
Use a separate checkout for another experiment, since outputs and the resolved
configuration are rooted in this checkout.

## Training and viewing results

```bash
python tools/supervisor.py
python tools/live_best.py --watch
```

The supervisor runs the two-GPU detector and then frozen-detector calibration
and Test evaluation. It writes `training.log` and `calibration.log`. Ctrl+C in
the viewer closes only the viewer. Resume training with:

```bash
python tools/supervisor.py --resume "$PWD/training/last.pt"
```

An empty workspace requires data, weights and a configured initial state before
training. The portable copy has CPU checks; a new-machine GPU run still requires
a forward/backward and memory preflight. Training is not started by setup tools.

## Provenance

`snapshot_manifest.json` records source and exported file hashes and identifies
the portability edits.
Only local dependency paths and standalone self-test fixtures are adapted;
the detector architecture, losses and training decisions are preserved.
Generated reports identify which checks were run. No trained performance is
implied by those CPU checks.
