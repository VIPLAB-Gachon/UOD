"""Prepare a fresh six-class YOLOv5s initialization for Ours-P2 training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))

import torch
from uod.engine import atomic_save, seed_all
from uod.reproducibility import file_sha256
from uod.yolov5_official import OfficialYOLOv5UOD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Initialization already exists: {args.output}")

    config = json.loads((ROOT / "configs/m3fd_640_811.template.json").read_text())
    if file_sha256(args.weights) != config["model"]["pretrained_weights_sha256"]:
        parser.error("Expected the pinned official COCO YOLOv5s v7.0 weights")

    torch.set_num_threads(2)
    seed_all(config["seed"])
    names = config["data"]["class_names"]
    model = OfficialYOLOv5UOD.from_pretrained(
        args.weights, names, use_nfe=False, separate_ife=True, feature_objective="hinge"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(
        {
            "state_dict": model.state_dict(),
            "epoch": -1,
            "global_update": 0,
            "seed": config["seed"],
            "class_names": names,
            "pretrained_source": model.pretrained_source,
        },
        args.output,
    )
    print(f"{args.output}  SHA-256 {file_sha256(args.output)}")


if __name__ == "__main__":
    main()
