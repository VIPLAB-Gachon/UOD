#!/usr/bin/env python3
"""Create a CPU-only common M3FD initialization from trusted COCO YOLOv5l weights.

This saves the continuous-box model before installing the 25-bin head. Training
loads this model-only state and initializes the distribution head separately.
The file records epoch=-1 and global_update=0; no training is performed. A newly
generated checkpoint is not guaranteed to reproduce the historical file's hash.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["People", "Car", "Bus", "Lamp", "Motorcycle", "Truck"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path,
                        help="trusted official 80-class COCO yolov5l.pt checkpoint")
    parser.add_argument("--output", required=True, type=Path,
                        help="new common initialization .pt file; existing files are never replaced")
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--threads", default=2, type=int)
    args = parser.parse_args()
    weights = args.weights.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not weights.is_file():
        parser.error(f"Weights not found: {weights}")
    if output.exists():
        parser.error(f"Refusing to replace an existing checkpoint: {output}")
    if args.threads < 1:
        parser.error("--threads must be positive")

    # Set before importing Torch. All model creation and serialization stay on
    # CPU, even when this helper is invoked on a GPU training server.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
    cache_root = output.parent / ".cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("YOLOV5_CONFIG_DIR", str(cache_root / "yolo"))
    sys.path.insert(0, str(ROOT / "runtime"))

    import torch
    from uod.engine import atomic_save, seed_all
    from uod.reproducibility import configure_determinism, file_sha256
    from uod.yolov5_official import OfficialYOLOv5UOD

    torch.set_num_threads(args.threads)
    configure_determinism(True)
    seed_all(args.seed)
    model = OfficialYOLOv5UOD.from_pretrained(
        weights, CLASSES, use_nfe=False, separate_ife=True,
        feature_objective="hinge",
    )
    source = model.pretrained_source
    if source["original_classes"] != 80:
        raise ValueError("The source checkpoint must be an original 80-class COCO detector")
    if (float(model.yaml.get("depth_multiple", 0)),
            float(model.yaml.get("width_multiple", 0))) != (1.0, 1.0):
        raise ValueError("The source checkpoint must use the current YOLOv5l width/depth")
    if (source["transferred_tensors"], source["target_tensors"]) != (607, 613):
        raise ValueError("Expected the current YOLOv5l six-class 607/613 tensor transfer")
    if torch.cuda.is_initialized() or any(p.device.type != "cpu" for p in model.parameters()):
        raise RuntimeError("Initialization must not use CUDA")

    atomic_save(dict(
        state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
        epoch=-1, global_update=0, seed=args.seed, class_names=CLASSES,
        origin="M3FD six-class untrained shared initialization from COCO yolov5l.pt",
        pretrained_source=source,
    ), output)
    print(json.dumps(dict(
        status="ready", checkpoint=str(output), sha256=file_sha256(output),
        class_names=CLASSES, epoch=-1, global_update=0, seed=args.seed,
        pretrained_source=source, gpu_used=False,
        historical_checkpoint_hash_reproduction_guaranteed=False,
    ), indent=2))


if __name__ == "__main__":
    main()
