"""Evaluate the bundled Ours-P2 EMA on a local paired manifest."""
import argparse
import importlib.util
import json
from pathlib import Path

from train_ddp import (
    ROOT, DetectionAP, DistributionDetect, OfficialYOLOv5UOD, build_dataset,
    candidate_pool, install_p2, make_loader, move_batch, soft_postprocess, torch,
    undo_letterbox,
)
from models.yolo import DetectionModel
from uod.reproducibility import configure_determinism


def load_detector(checkpoint, device):
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if artifact.get("format") != "ours-p2-ema-v1":
        raise ValueError("Expected an Ours-P2 EMA export")
    config = artifact["config"]
    settings = config["model"]
    names = config["data"]["class_names"]
    base = DetectionModel(artifact["yolov5_yaml"], ch=3, nc=len(names))
    model = OfficialYOLOv5UOD(
        base, names, source={}, use_nfe=False, separate_ife=True
    )
    model.ifd = DistributionDetect(model.ifd, **settings["localization_distribution"])
    install_p2(model, **settings["p2"])
    model.stride = model.ifd.stride
    model.load_state_dict(artifact["ema_state_dict"], strict=True)
    return model.eval().requires_grad_(False).to(device), config, artifact["epoch"]


def load_cp(path):
    spec = importlib.util.spec_from_file_location("ours_p2_cp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cp, _, checks, _ = module.load_existing_cp()
    if checks.get("status") != "passed":
        raise RuntimeError("CP-Cluster backend did not pass its self-test")
    return cp


@torch.inference_mode()
def evaluate(args):
    torch.set_num_threads(2)
    configure_determinism(True)
    device = torch.device(args.device)
    model, config, epoch = load_detector(args.checkpoint, device)
    config["data"]["test_manifest"] = str(args.manifest.resolve())
    dataset = build_dataset(config, "test")
    loader = make_loader(dataset, args.batch_size, 0, False, device)
    cp = load_cp(args.cp_loader) if args.cp_loader else None
    modes = ("rgb", "thermal", "rgbt")
    metrics = {mode: DetectionAP(model.nc) for mode in modes}
    settings = config["postprocess"]

    def predict(decoded, meta):
        if cp is None:
            return soft_postprocess(decoded, meta, settings)
        pool = candidate_pool(decoded.cpu(), settings)
        result = cp(pool, iou_threshold=settings["iou_threshold"],
                    min_score=settings["score_threshold"], max_det=settings["max_detections"],
                    wfa_thresh=.8, tune_coords=1, opt_id=3)
        return {
            "boxes": undo_letterbox(result[:, :4], meta),
            "scores": result[:, 4],
            "labels": result[:, 5].long(),
        }

    for raw in loader:
        batch = move_batch(raw, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model.forward_pair(batch["rgb"], batch["thermal"])
        rgb = output["decoded_rgb"].float()
        thermal = output["decoded_thermal"].float()
        decoded = (rgb, thermal, torch.cat((rgb, thermal), dim=1))
        for mode, images in zip(modes, decoded, strict=True):
            predictions = [predict(image, meta) for image, meta in zip(images, raw["meta"], strict=True)]
            metrics[mode].update(predictions, raw["original_targets"])

    return {
        "checkpoint_epoch": epoch,
        "postprocess": "cp_cluster" if cp is not None else "linear_soft_nms",
        "metrics": {mode: meter.compute() for mode, meter in metrics.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/ours-p2-e184.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cp-loader", type=Path, help="Trusted local evaluator that exports load_existing_cp()")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2, allow_nan=False))
