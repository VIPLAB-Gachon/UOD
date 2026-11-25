#!/usr/bin/env python3
"""Train UOD with a pinned official YOLOv5 v7.0 detector and loss."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch

from torch.utils.data import Subset

from map50_tracking import Map50Tracker
from uod.alignment_metrics import AlignmentDiagnostics, ForegroundCentroidProbe, evenly_spaced_indices
from uod.data import assert_disjoint
from uod.engine import (atomic_save, build_dataset, make_loader, move_batch,
                        read_config, seed_all, select_device)
from uod.geometry import undo_letterbox
from uod.localization_distribution import DistributionDetect
from uod.localization_loss import DistributionUODLoss, EDGE_GATE_STATS
from uod.metrics import DetectionAP
from uod.reproducibility import (configure_determinism, numpy_rng_state, restore_rng,
                                 validate_fork_config, file_sha256, batch_fingerprint,
                                 verify_restored_fork)
from uod.yolov5_official import (ModelEMA, OfficialYOLOv5UOD,
    OfficialYOLOv5UODLoss, non_max_suppression, smart_optimizer)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    checkpoint_args = parser.add_mutually_exclusive_group()
    checkpoint_args.add_argument("--resume", help="Resume a checkpoint produced by this script")
    checkpoint_args.add_argument("--fork-from", help="Start a new matched arm from a common complete checkpoint")
    parser.add_argument("--expected-fork-sha256", help="Required content hash for a common fork")
    parser.add_argument("--micro-batch-size", type=int,
                        help="Override physical pair batch; effective batch stays in config")
    parser.add_argument("--workers", type=int)
    return parser.parse_args()


def resolve_config(path: str, device: str | None, micro_batch: int | None,
                   workers: int | None):
    config_path = Path(path).expanduser().resolve()
    cfg = read_config(config_path)
    weights = Path(cfg["model"]["weights"]).expanduser()
    cfg["model"]["weights"] = str(
        (config_path.parent / weights).resolve() if not weights.is_absolute() else weights.resolve())
    if device:
        cfg["device"] = device
    if micro_batch is not None:
        cfg["training"]["micro_batch_size"] = micro_batch
    if workers is not None:
        cfg["num_workers"] = workers
    return cfg


def scaled_yolov5_hyp(cfg: dict, model: OfficialYOLOv5UOD):
    hyp = dict(cfg["loss"]["detection"])
    required = {"box", "cls", "cls_pw", "obj", "obj_pw", "anchor_t", "fl_gamma"}
    missing = required - hyp.keys()
    if missing:
        raise ValueError(f"Missing YOLOv5 loss hyperparameters: {sorted(missing)}")
    levels = int(model.ifd.nl)
    image_size = int(cfg["data"]["image_size"])
    hyp["box"] *= 3 / levels
    hyp["cls"] *= model.nc / 80 * 3 / levels
    hyp["obj"] *= (image_size / 640) ** 2 * 3 / levels
    return hyp


def validation_nms(prediction, confidence, iou, max_det):
    """Process every image even when the upstream batch NMS time limit expires."""
    # YOLOv5 NMS can return empty results for the remaining batch on timeout.
    # Each call contains one image so its deadline cannot skip another image.
    return [non_max_suppression(
        image.unsqueeze(0), confidence, iou, classes=None, agnostic=False,
        multi_label=False, max_det=max_det)[0] for image in prediction]


@torch.no_grad()
def evaluate(model: OfficialYOLOv5UOD, loader, device: torch.device,
             postprocess: dict, amp: bool, diagnostics: AlignmentDiagnostics | None = None,
             diagnostic_indices: set[int] | None = None):
    model.eval()
    metrics = {mode: DetectionAP(model.nc) for mode in ("rgb", "thermal", "rgbt")}
    confidence = float(postprocess.get("score_threshold", 0.001))
    iou = float(postprocess.get("iou_threshold", 0.6))
    max_det = int(postprocess.get("max_detections", 300))
    offset = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        selected = [i for i in range(len(batch["targets"]))
                    if diagnostics is not None and
                    (diagnostic_indices is None or offset + i in diagnostic_indices)]
        previous_offset = offset
        offset += len(batch["targets"])
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model.forward_pair(batch["rgb"], batch["thermal"], return_features=bool(selected))
        if selected:
            diagnostics.update(
                [value[selected] for value in output["intrinsic_rgb"]],
                [value[selected] for value in output["intrinsic_thermal"]],
                batch["valid_mask"][selected], [batch["targets"][i] for i in selected],
                batch["rgb"].shape[-2:])
        rgb = output["decoded_rgb"].float()
        thermal = output["decoded_thermal"].float()
        candidates = {
            "rgb": rgb.clone(),
            "thermal": thermal.clone(),
            "rgbt": torch.cat((rgb, thermal), dim=1),
        }
        for mode, prediction in candidates.items():
            detections = validation_nms(prediction, confidence, iou, max_det)
            formatted = []
            for detection, meta in zip(detections, batch["meta"], strict=True):
                formatted.append({
                    "boxes": undo_letterbox(detection[:, :4], meta),
                    "scores": detection[:, 4],
                    "labels": detection[:, 5].long(),
                })
            metrics[mode].update(formatted, raw_batch["original_targets"])
        if offset // 500 > previous_offset // 500 or offset == len(loader.dataset):
            print(f"validation pairs {offset}/{len(loader.dataset)}", flush=True)
    return {mode: metric.compute() for mode, metric in metrics.items()}



@torch.no_grad()
def evaluate_with_diagnostics(model, val_loader, probe_loader, device, cfg, amp,
                              out_dir: Path, epoch: int):
    diagnostic_cfg = cfg.get("diagnostics", {})
    diagnostics, selected = None, None
    if diagnostic_cfg.get("enabled", False):
        model.eval()
        probe = ForegroundCentroidProbe(int(diagnostic_cfg.get("samples_per_region", 128)))
        print(f"epoch {epoch}: fitting object/background probe on training subset", flush=True)
        for raw_batch in probe_loader:
            batch = move_batch(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                output = model.forward_pair(batch["rgb"], batch["thermal"], return_features=True)
            probe.update(output["intrinsic_rgb"], output["intrinsic_thermal"],
                         batch["valid_mask"], batch["targets"], batch["rgb"].shape[-2:])
        diagnostics = AlignmentDiagnostics(
            temperature=float(diagnostic_cfg.get("temperature", 1.0)), probe=probe,
            samples_per_region=int(diagnostic_cfg.get("samples_per_region", 128)))
        selected = set(evenly_spaced_indices(len(val_loader.dataset),
                       int(diagnostic_cfg.get("val_pairs", 256))))
        atomic_save(probe.state_dict(), out_dir / f"probe_epoch_{epoch:03d}.pt")
    print(f"epoch {epoch}: validating all {len(val_loader.dataset)} pairs", flush=True)
    results = evaluate(model, val_loader, device, cfg.get("postprocess", {}), amp,
                       diagnostics, selected)
    if diagnostics is not None:
        report = diagnostics.compute()
        report.update(epoch=epoch, weights="EMA", train_probe_pairs=len(probe_loader.dataset))
        (out_dir / f"alignment_epoch_{epoch:03d}.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        path = out_dir / "alignment_history.csv"
        fields = ["epoch", "level", "object_count", "object_js", "object_cosine",
                  "foreground_js", "background_js", "all_js",
                  "rgb_probe_auc", "thermal_probe_auc", "rgb_variance", "thermal_variance",
                  "rgb_entropy", "thermal_entropy", "rgb_rms", "thermal_rms"]
        is_new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if is_new:
                writer.writeheader()
            for level, values in report["levels"].items():
                row = {"epoch": epoch, "level": level,
                       "object_count": values.get("objects", {}).get("count", 0),
                       "object_js": values.get("objects", {}).get("js"),
                       "object_cosine": values.get("objects", {}).get("cosine"),
                       **{f"{region}_js": values[region]["js"] for region in ("foreground", "background", "all")}}
                for mode in ("rgb", "thermal"):
                    row.update({f"{mode}_probe_auc": values[f"{mode}_probe"]["auc"],
                                f"{mode}_variance": values["all"][f"{mode}_channel_variance"],
                                f"{mode}_entropy": values["all"][f"{mode}_entropy"],
                                f"{mode}_rms": values["all"][f"{mode}_rms"]})
                writer.writerow(row)
        print("alignment:", {level: {"object_js": values.get("objects", {}).get("js"),
            "rgb_probe_auc": values["rgb_probe"]["auc"],
            "thermal_probe_auc": values["thermal_probe"]["auc"]}
            for level, values in report["levels"].items()}, flush=True)
    return results


def checkpoint_state(model, ema, optimizer, scheduler, scaler, cfg, epoch,
                     best_score, global_update, loader_generator, val_generator,
                     probe_generator=None, best_scores=None, backend=None):
    cpu_state = lambda module: {key: value.detach().cpu() for key, value in module.state_dict().items()}
    return {
        "format_version": 2,
        "kind": "uod_yolov5_training",
        "state_dict": cpu_state(model),
        "ema_state_dict": cpu_state(ema.ema),
        "ema_updates": ema.updates,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "global_update": global_update,
        "config": cfg,
        "pretrained_source": model.pretrained_source,
        "best_scores": dict(best_scores or {}),
        "backend": backend or {},
        "numpy_rng": numpy_rng_state(),
        "probe_loader_rng": probe_generator.get_state() if probe_generator is not None else None,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "loader_rng": loader_generator.get_state(),
        "val_loader_rng": val_generator.get_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def metric_value(row: dict, name: str):
    if name in {"val/unimodal_mean/map50_95", "val/unimodal_mean/map50"}:
        key = name.rsplit("/", 1)[1]
        return .5 * (float(row["val/rgb/" + key]) + float(row["val/thermal/" + key]))
    if name not in row or row[name] in (None, ""):
        raise ValueError(f"best_metric {name!r} is absent from validation results")
    return float(row[name])


def scheduled_feature_alpha(loss_cfg: dict, epoch_progress: float) -> float:
    """Linearly introduce feature loss after an optional detector-only phase."""
    target = float(loss_cfg.get("alpha", 1.0))
    start = float(loss_cfg.get("alpha_start_epoch", 0.0))
    ramp = float(loss_cfg.get("alpha_ramp_epochs", 0.0))
    if target < 0 or start < 0 or ramp < 0:
        raise ValueError("alpha, alpha_start_epoch and alpha_ramp_epochs must be non-negative")
    if target == 0 or epoch_progress <= start:
        return 0.0
    if ramp == 0:
        return target
    return target * min(1.0, (epoch_progress - start) / ramp)


def copy_fork_history(source_dir: Path, out_dir: Path, completed_epoch: int):
    """Keep inherited evaluation history with its original epoch numbers."""
    source_history = source_dir / "history.csv"
    if source_history.exists():
        with source_history.open(newline="") as handle:
            if any(int(row["epoch"]) > completed_epoch for row in csv.DictReader(handle)):
                raise ValueError("Fork source must be frozen at the common checkpoint; future history/best cannot be inherited")
    for name in ("history.csv", "alignment_history.csv"):
        path = source_dir / name
        if path.exists():
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fields = list(reader.fieldnames)
                rows = [r for r in reader if int(r["epoch"]) <= completed_epoch]
            if name == "history.csv":
                # These diagnostics did not exist in the shared warmup. Leave
                # inherited rows blank rather than inventing measured zeros.
                fields += [key for key in EDGE_GATE_STATS if key not in fields]
            with (out_dir / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    for pattern in ("val_epoch_*.json", "alignment_epoch_*.json", "probe_epoch_*.pt"):
        for path in source_dir.glob(pattern):
            if int(path.stem.rsplit("_", 1)[1]) <= completed_epoch:
                shutil.copy2(path, out_dir / path.name)
    for path in source_dir.glob("best*.pt"):
        shutil.copy2(path, out_dir / path.name)


def main():
    args = parse_args()
    cfg = resolve_config(args.config, args.device, args.micro_batch_size, args.workers)
    seed = int(cfg.get("seed", 2026))
    backend = configure_determinism(bool(cfg.get("deterministic", False)))
    seed_all(seed)
    device = select_device(cfg.get("device", "cuda:0"))
    tr = cfg["training"]
    if tr.get("optimizer") != "SGD":
        raise ValueError("This reproduction run requires optimizer: SGD")
    epochs = int(tr["epochs"])
    effective_batch = int(tr["batch_size"])
    micro_batch = int(tr.get("micro_batch_size", effective_batch))
    if epochs < 1 or micro_batch < 1 or effective_batch < micro_batch:
        raise ValueError("Invalid epochs/effective/micro batch size")
    if effective_batch % micro_batch:
        raise ValueError("Effective pair batch must be divisible by micro_batch_size")
    accumulate = effective_batch // micro_batch
    workers = int(cfg.get("num_workers", 4))
    out_dir = Path(tr["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.fork_from and any((out_dir / name).exists() for name in ("last.pt", "history.csv", "fork_initial.pt")):
        raise FileExistsError("A matched fork needs an unused output directory")
    if not args.resume and (out_dir / "last.pt").exists():
        raise FileExistsError(f"Existing run: {out_dir}. Use --resume or change out_dir")

    train_ds = build_dataset(cfg, "train", True)
    val_ds = build_dataset(cfg, "val", False)
    assert_disjoint(train_ds, val_ds)
    loader_generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(seed + 1)
    train_loader = make_loader(train_ds, micro_batch, workers, True, device, loader_generator)
    val_loader = make_loader(
        val_ds, int(tr.get("val_batch_size", micro_batch)), workers,
        False, device, val_generator)

    probe_loader = None
    probe_generator = torch.Generator().manual_seed(seed + 2)
    diagnostic_cfg = cfg.get("diagnostics", {})
    if diagnostic_cfg.get("enabled", False):
        probe_ds = build_dataset(cfg, "train", False)
        probe_indices = evenly_spaced_indices(len(probe_ds), int(diagnostic_cfg.get("train_pairs", 128)))
        probe_loader = make_loader(
            Subset(probe_ds, probe_indices), int(tr.get("val_batch_size", 1)), workers,
            False, device, probe_generator)
        protocol = {
            "training_probe_indices": probe_indices,
            "training_probe_ids": [probe_ds.samples[i]["id"] for i in probe_indices],
            "validation_indices": evenly_spaced_indices(len(val_ds), int(diagnostic_cfg.get("val_pairs", 256))),
            "training_augmentation": False,
            "foreground_definition": "feature-cell center inside any GT box; excludes padding",
            "model_updates": False,
            "probe": "class centroids fit on training features; held-out object/background ROC AUC",
            "samples_per_image_per_region": int(diagnostic_cfg.get("samples_per_region", 128)),
        }
        (out_dir / "diagnostic_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    model = OfficialYOLOv5UOD.from_pretrained(
        cfg["model"]["weights"], cfg["data"]["class_names"],
        float(cfg["model"].get("nfe_noise", 0.05)),
        use_nfe=bool(cfg["model"].get("use_nfe", True)),
        feature_objective=cfg["loss"].get("feature_objective", "hinge"),
        task_loss=cfg["loss"].get("task", {}),
        modality_loss=cfg["loss"].get("js_orth_modality", {}),
        separate_ife=bool(cfg["model"].get("separate_ife", False)))
    distribution_cfg = cfg["model"].get("localization_distribution")
    if not distribution_cfg:
        raise ValueError("This isolated trainer requires a localization distribution head")
    initial_path = Path(cfg["model"]["initial_checkpoint"]).resolve()
    if file_sha256(initial_path) != cfg["model"]["initial_checkpoint_sha256"]:
        raise ValueError("Initial shared detector checkpoint hash differs")
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    if int(initial["epoch"]) != -1 or int(initial["global_update"]) != 0:
        raise ValueError("Expected the fixed, untrained common detector checkpoint")
    model.load_state_dict(initial["state_dict"], strict=True)
    del initial
    model.ifd = DistributionDetect(model.ifd, **distribution_cfg)
    model.stride = model.ifd.stride
    model.pretrained_source = {**model.pretrained_source,
        "initial_checkpoint": str(initial_path),
        "initial_checkpoint_sha256": cfg["model"]["initial_checkpoint_sha256"],
        "localization_distribution": distribution_cfg,
        "distribution_initialization": "zero weights; Gaussian bias at .5 anchor distance; object/class rows copied"}
    hyp = scaled_yolov5_hyp(cfg, model)
    model.hyp = hyp
    model = model.to(device)
    criterion = DistributionUODLoss(
        model, alpha=float(cfg["loss"].get("alpha", 0.)),
        **cfg["loss"]["localization"])


    nominal_batch = int(tr.get("nominal_batch_size", 64))
    scaled_decay = float(tr["weight_decay"]) * effective_batch / nominal_batch
    optimizer = smart_optimizer(
        model, "SGD", float(tr["lr0"]), float(tr["momentum"]), scaled_decay)
    lrf = float(tr.get("lrf", 0.01))
    if not 0 < lrf <= 1:
        raise ValueError("lrf must be in (0,1]")
    schedule_epochs = int(tr.get("lr_schedule_epochs", epochs))
    if schedule_epochs < epochs:
        raise ValueError("lr_schedule_epochs must be greater than or equal to epochs")
    lr_factor = lambda epoch: (1 - epoch / schedule_epochs) * (1 - lrf) + lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    amp = bool(tr.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = ModelEMA(model)

    start_epoch, best_score, global_update = 0, -1.0, 0
    best_metric = tr.get("best_metric", "val/rgbt/map50")
    tracked_best = list(dict.fromkeys([best_metric, *tr.get("additional_best_metrics", [])]))
    best_scores = {metric: -1.0 for metric in tracked_best}
    checkpoint_path = args.fork_from or args.resume
    fork_record = None
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path).resolve()
        if args.fork_from:
            if not args.expected_fork_sha256 or file_sha256(checkpoint_path) != args.expected_fork_sha256:
                raise ValueError("Common checkpoint hash is missing or does not match")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("kind") != "uod_yolov5_training":
            raise ValueError("Not a checkpoint produced by this trainer")
        if args.fork_from:
            validate_fork_config(checkpoint["config"], cfg)
            if checkpoint.get("backend") != backend:
                raise ValueError("Backend settings differ from the common checkpoint")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        ema.ema.load_state_dict(checkpoint["ema_state_dict"], strict=True)
        ema.updates = int(checkpoint["ema_updates"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_scores.update(checkpoint.get("best_scores", {}))
        best_scores[best_metric] = best_score
        global_update = int(checkpoint["global_update"])
        restore_rng(checkpoint, loader_generator, val_generator, probe_generator,
                    strict=bool(args.fork_from))
        if args.fork_from:
            copy_fork_history(checkpoint_path.parent, out_dir, start_epoch)
            fork_record = {"source": str(checkpoint_path), "sha256": args.expected_fork_sha256,
                           "completed_epoch": start_epoch, "global_update": global_update,
                           "ema_updates": ema.updates, "backend": backend,
                           "alignment_region": cfg["loss"].get("js_alignment", {}).get("region", "all"),
                           "target_alpha": cfg["loss"].get("alpha", 0.)}
            (out_dir / "fork.json").write_text(json.dumps(fork_record, indent=2))
        del checkpoint
    if start_epoch >= epochs:
        raise ValueError(f"Run already reached {start_epoch} epochs (target={epochs})")

    cfg["runtime"] = {
        "effective_pair_batch": effective_batch,
        "micro_pair_batch": micro_batch,
        "accumulation_steps": accumulate,
        "scaled_weight_decay": scaled_decay,
        "lr_schedule_epochs": schedule_epochs,
        "yolov5_hyp_after_scaling": hyp,
        "pretrained_source": model.pretrained_source,
        "nfe_enabled": model.use_nfe,
        "separate_ife": model.separate_ife,
        "backend": backend,
    }
    (out_dir / "config.resolved.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    deploy_parameters = sum(parameter.numel() for name, parameter in model.named_parameters()
                            if not name.startswith(("nfe_rgb.", "nfe_thermal.", "task_features.", "modality_features.")))
    print(f"Pairs train/val={len(train_ds)}/{len(val_ds)}, device={device}, image={cfg['data']['image_size']}")
    print(f"Pair batch effective/micro={effective_batch}/{micro_batch}, accumulate={accumulate}")
    print(f"Parameters train/deploy={parameters:,}/{deploy_parameters:,}; AMP={amp}")
    print(
        f"NFE enabled={model.use_nfe}; localization distribution/JSD enabled; "
        f"separate IFE={model.separate_ife}; "
        f"objective={model.feature_objective}; "
        f"mode={cfg['loss'].get('negative_mode', 'hinge') if model.feature_objective == 'hinge' else model.feature_objective}; "
        f"alpha target={cfg['loss'].get('alpha', 1.0)}; "
        f"start/ramp={cfg['loss'].get('alpha_start_epoch', 0.0)}/"
        f"{cfg['loss'].get('alpha_ramp_epochs', 0.0)} epochs")
    print(f"Pretrained transfer: {model.pretrained_source}")
    print(f"Optimizer=SGD, lr0={tr['lr0']}, momentum={tr['momentum']}, decay={scaled_decay}")
    print(f"Training epochs={epochs}; LR schedule horizon={schedule_epochs} epochs")

    if not checkpoint_path and diagnostic_cfg.get("initial_evaluation", False):
        initial = evaluate_with_diagnostics(ema.ema, val_loader, probe_loader, device, cfg,
                                            amp, out_dir, 0)
        (out_dir / "val_epoch_000.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
        print("Initial evaluation saved", flush=True)
    if fork_record is not None or (not checkpoint_path and
            (diagnostic_cfg.get("initial_evaluation", False) or tr.get("save_initial_checkpoint", False))):
        filename = "fork_initial.pt" if fork_record else "initial.pt"
        starting_state = checkpoint_state(model, ema, optimizer, scheduler, scaler, cfg, start_epoch - 1,
                    best_score, global_update, loader_generator, val_generator,
                    probe_generator, best_scores, backend)
        if fork_record is not None:
            verification = verify_restored_fork(checkpoint_path, starting_state)
            (out_dir / "fork_verification.json").write_text(json.dumps(verification, indent=2))
        atomic_save(starting_state, out_dir / filename)
        del starting_state
        print(f"Saved {filename}; starting optimizer updates at epoch {start_epoch + 1}", flush=True)

    log_path = out_dir / "history.csv"
    # This supplemental selection is outside checkpoint state, so the common
    # fork and its legacy mAP50-95 bookkeeping remain exactly reproducible.
    map50_tracker = Map50Tracker(out_dir, log_path, start_epoch, warmup_epochs=3)
    batches_per_epoch = len(train_loader)
    warmup_iterations = max(round(float(tr.get("warmup_epochs", 3.0)) * batches_per_epoch), 100)
    audited_batches = 0
    for epoch in range(start_epoch, epochs):
        begin = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running: dict[str, float] = {}
        seen = 0
        support_min, support_max = float("inf"), -float("inf")
        augmentation_audit = {"epoch": epoch + 1, "pairs": 0, "boxes_before": 0,
                              "boxes_after": 0, "boxes_clipped": 0,
                              "scale_min": 1., "scale_max": 1.}
        for step, raw_batch in enumerate(train_loader):
            iteration = epoch * batches_per_epoch + step
            epoch_progress = (iteration + 1) / batches_per_epoch
            criterion.alpha = scheduled_feature_alpha(cfg["loss"], epoch_progress)
            if iteration <= warmup_iterations:
                progress = iteration / max(1, warmup_iterations)
                target_lr = float(tr["lr0"]) * lr_factor(epoch)
                for group_index, group in enumerate(optimizer.param_groups):
                    start_lr = float(tr.get("warmup_bias_lr", 0.1)) if group_index == 0 else 0.0
                    group["lr"] = start_lr + progress * (target_lr - start_lr)
                    if "momentum" in group:
                        start_momentum = float(tr.get("warmup_momentum", 0.8))
                        group["momentum"] = start_momentum + progress * (
                            float(tr["momentum"]) - start_momentum)
            if audited_batches < int(tr.get("audit_batches", 0)):
                audit = {"epoch": epoch + 1, "step": step + 1, **batch_fingerprint(raw_batch)}
                with (out_dir / "batch_audit.jsonl").open("a") as handle:
                    handle.write(json.dumps(audit) + "\n")
                audited_batches += 1
            for meta in raw_batch["meta"]:
                aug = meta.get("affine")
                if aug:
                    augmentation_audit["pairs"] += 1
                    for key in ("boxes_before", "boxes_after", "boxes_clipped"):
                        augmentation_audit[key] += aug[key]
                    augmentation_audit["scale_min"] = min(augmentation_audit["scale_min"], aug["scale"])
                    augmentation_audit["scale_max"] = max(augmentation_audit["scale_max"], aug["scale"])
            batch = move_batch(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                outputs = model.forward_pair(batch["rgb"], batch["thermal"])
                loss, stats = criterion(
                    outputs, batch["targets"], batch["rgb"].shape[-2:], batch["valid_mask"])
                stats["feature_alpha"] = loss.new_tensor(criterion.alpha).detach()
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch + 1}, step={step}; "
                    f"ids={[meta['id'] for meta in batch['meta']]}")
            scaler.scale(loss).backward()
            boundary = (step + 1) % accumulate == 0 or step + 1 == batches_per_epoch
            if boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(tr.get("grad_clip", 10.0)),
                    error_if_nonfinite=not scaler.is_enabled())
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.is_enabled() and scaler.get_scale() < old_scale
                if not skipped:
                    ema.update(model)
                    global_update += 1
                else:
                    print(f"AMP overflow: optimizer/EMA update skipped epoch={epoch + 1} "
                          f"step={step + 1} scale={old_scale:g}->{scaler.get_scale():g}", flush=True)
                if tr.get("audit_batches", 0) and (skipped or step < int(tr["audit_batches"])):
                    event = {"epoch": epoch + 1, "step": step + 1, "skipped": skipped,
                             "scale_before": old_scale, "scale_after": scaler.get_scale(),
                             "global_update": global_update,
                             "lr": [group["lr"] for group in optimizer.param_groups]}
                    with (out_dir / "optimizer_events.jsonl").open("a") as handle:
                        handle.write(json.dumps(event) + "\n")
                optimizer.zero_grad(set_to_none=True)
            batch_size = batch["rgb"].shape[0]
            support_min = min(support_min, float(stats["target_support_min"]))
            support_max = max(support_max, float(stats["target_support_max"]))
            for key, value in stats.items():
                running[key] = running.get(key, 0.0) + float(value) * batch_size
            seen += batch_size
            log_every = int(tr.get("log_every", 200))
            if log_every and ((step + 1) % log_every == 0 or step + 1 == batches_per_epoch):
                print(
                    f"epoch {epoch + 1}/{epochs} step {step + 1}/{batches_per_epoch} "
                    f"loss={float(stats['loss']):.4f} det={float(stats['det_sum']):.4f} "
                    f"jsd={float(stats['fs']):.5f} dfl={float(stats['dfl']):.5f} alpha={criterion.alpha:.5g} "
                    f"Rteach/pair={float(stats['selected_rgb_teacher_per_pair']):.3f} "
                    f"Tteach/pair={float(stats['selected_thermal_teacher_per_pair']):.3f} "
                    f"edges/pair={float(stats['selected_edges_per_pair']):.3f}/"
                    f"{float(stats['eligible_edges_per_pair']):.3f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6g}",
                    flush=True)
        if augmentation_audit["pairs"]:
            (out_dir / f"augmentation_epoch_{epoch + 1:03d}.json").write_text(
                json.dumps(augmentation_audit, indent=2))
            print("augmentation:", augmentation_audit, flush=True)
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            **{key: value / seen for key, value in running.items()},
        }
        selected_count = row["selected_rgb_teacher_per_pair"] + row["selected_thermal_teacher_per_pair"]
        row["teacher_iou_selected"] = row["teacher_iou_sum_per_pair"] / selected_count if selected_count else 0.
        row["student_iou_selected"] = row["student_iou_sum_per_pair"] / selected_count if selected_count else 0.
        eligible = row["eligible_edges_per_pair"]
        row["edge_gate_blocked_fraction"] = row["blocked_edges_per_pair"] / eligible if eligible else 0.
        row["target_support_min"] = support_min
        row["target_support_max"] = support_max
        improved = False
        improved_metrics = []
        if (epoch + 1) % int(tr.get("eval_every", 1)) == 0 or epoch + 1 == epochs:
            results = evaluate_with_diagnostics(ema.ema, val_loader, probe_loader, device,
                                                cfg, amp, out_dir, epoch + 1)
            (out_dir / f"val_epoch_{epoch + 1:03d}.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8")
            for mode, values in results.items():
                for key in ("map50", "map50_95"):
                    row[f"val/{mode}/{key}"] = values[key]
            score = metric_value(row, best_metric)
            improved = score > best_score
            if improved:
                best_score = score
            for metric in tracked_best:
                value = metric_value(row, metric)
                if value > best_scores.get(metric, -1.0):
                    best_scores[metric] = value
                    improved_metrics.append(metric)
            print("validation:", {
                mode: {key: values[key] for key in ("map50", "map50_95")}
                for mode, values in results.items()}, flush=True)
        for mode in ("rgb", "thermal", "rgbt"):
            for key in ("map50", "map50_95"):
                row.setdefault(f"val/{mode}/{key}", "")
        row["seconds"] = time.monotonic() - begin
        fields = list(row)
        if log_path.exists():
            with log_path.open(newline="", encoding="utf-8") as handle:
                existing = next(csv.reader(handle))
            if set(existing) != set(fields):
                raise ValueError("history.csv schema mismatch; select a new out_dir")
            fields = existing
        new_log = not log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if new_log:
                writer.writeheader()
            writer.writerow(row)

        payload = checkpoint_state(
            model, ema, optimizer, scheduler, scaler, cfg, epoch, best_score,
            global_update, loader_generator, val_generator,
            probe_generator, best_scores, backend)
        atomic_save(payload, out_dir / "last.pt")
        if row["val/rgb/map50"] != "":
            map50_tracker.update(row, saved_checkpoint_epoch=epoch)
        if improved:
            shutil.copy2(out_dir / "last.pt", out_dir / "best.pt")
        for metric in improved_metrics:
            filename = "best_" + metric.replace("val/", "").replace("/", "_") + ".pt"
            shutil.copy2(out_dir / "last.pt", out_dir / filename)
        if ((epoch + 1) % int(tr.get("save_every", 5)) == 0 or
                epoch + 1 in tr.get("checkpoint_epochs", [])):
            shutil.copy2(out_dir / "last.pt", out_dir / f"epoch_{epoch + 1:03d}.pt")
        if epoch + 1 in tr.get("review_epochs", []):
            (out_dir / f"review_epoch_{epoch + 1:03d}.json").write_text(
                json.dumps({"epoch": epoch + 1, "metrics": row,
                            "best_scores": best_scores, "checkpoint": str(out_dir / "last.pt")}, indent=2))
        print(
            f"epoch {epoch + 1} saved; elapsed={row['seconds']:.1f}s; "
            f"best {best_metric}={best_score:.4f}", flush=True)
    print(f"Finished. Training checkpoint: {out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
