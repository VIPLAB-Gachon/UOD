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

from uod.data import assert_disjoint
from uod.engine import (atomic_save, build_dataset, make_loader, move_batch,
                        read_config, seed_all, select_device)
from uod.geometry import undo_letterbox
from uod.metrics import DetectionAP
from uod.yolov5_official import (ModelEMA, OfficialYOLOv5UOD,
    OfficialYOLOv5UODLoss, non_max_suppression, smart_optimizer)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--resume", help="Resume a checkpoint produced by this script")
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


@torch.no_grad()
def evaluate(model: OfficialYOLOv5UOD, loader, device: torch.device,
             postprocess: dict, amp: bool):
    model.eval()
    metrics = {mode: DetectionAP(model.nc) for mode in ("rgb", "thermal", "rgbt")}
    confidence = float(postprocess.get("score_threshold", 0.001))
    iou = float(postprocess.get("iou_threshold", 0.6))
    max_det = int(postprocess.get("max_detections", 300))
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model.forward_pair(batch["rgb"], batch["thermal"])
        rgb = output["decoded_rgb"].float()
        thermal = output["decoded_thermal"].float()
        candidates = {
            "rgb": rgb.clone(),
            "thermal": thermal.clone(),
            "rgbt": torch.cat((rgb, thermal), dim=1),
        }
        for mode, prediction in candidates.items():
            detections = non_max_suppression(
                prediction, confidence, iou, classes=None, agnostic=False,
                multi_label=False, max_det=max_det)
            formatted = []
            for detection, meta in zip(detections, batch["meta"], strict=True):
                formatted.append({
                    "boxes": undo_letterbox(detection[:, :4], meta),
                    "scores": detection[:, 4],
                    "labels": detection[:, 5].long(),
                })
            metrics[mode].update(formatted, raw_batch["original_targets"])
    return {mode: metric.compute() for mode, metric in metrics.items()}


def checkpoint_state(model, ema, optimizer, scheduler, scaler, cfg, epoch,
                     best_score, global_update, loader_generator, val_generator):
    cpu_state = lambda module: {key: value.detach().cpu() for key, value in module.state_dict().items()}
    return {
        "format_version": 1,
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
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "loader_rng": loader_generator.get_state(),
        "val_loader_rng": val_generator.get_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def metric_value(row: dict, name: str):
    if name not in row or row[name] in (None, ""):
        raise ValueError(f"best_metric {name!r} is absent from validation results")
    return float(row[name])


def main():
    args = parse_args()
    cfg = resolve_config(args.config, args.device, args.micro_batch_size, args.workers)
    seed = int(cfg.get("seed", 2026))
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

    model = OfficialYOLOv5UOD.from_pretrained(
        cfg["model"]["weights"], cfg["data"]["class_names"],
        float(cfg["model"].get("nfe_noise", 0.05)))
    hyp = scaled_yolov5_hyp(cfg, model)
    model.hyp = hyp
    model = model.to(device)
    criterion = OfficialYOLOv5UODLoss(
        model, alpha=float(cfg["loss"].get("alpha", 1.0)),
        margin=float(cfg["loss"].get("margin", 1.0)),
        reduction=cfg["loss"].get("reduction", "pixel"),
        mask_padding=bool(cfg["loss"].get("mask_padding", False)))

    nominal_batch = int(tr.get("nominal_batch_size", 64))
    scaled_decay = float(tr["weight_decay"]) * effective_batch / nominal_batch
    optimizer = smart_optimizer(
        model, "SGD", float(tr["lr0"]), float(tr["momentum"]), scaled_decay)
    lrf = float(tr.get("lrf", 0.01))
    if not 0 < lrf <= 1:
        raise ValueError("lrf must be in (0,1]")
    lr_factor = lambda epoch: (1 - epoch / epochs) * (1 - lrf) + lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    amp = bool(tr.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = ModelEMA(model)

    start_epoch, best_score, global_update = 0, -1.0, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("kind") != "uod_yolov5_training":
            raise ValueError("Not a checkpoint produced by this trainer")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        ema.ema.load_state_dict(checkpoint["ema_state_dict"], strict=True)
        ema.updates = int(checkpoint["ema_updates"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        global_update = int(checkpoint["global_update"])
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        loader_generator.set_state(checkpoint["loader_rng"])
        val_generator.set_state(checkpoint["val_loader_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
            saved = checkpoint["cuda_rng"]
            if len(saved) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(saved)
    if start_epoch >= epochs:
        raise ValueError(f"Run already reached {start_epoch} epochs (target={epochs})")

    cfg["runtime"] = {
        "effective_pair_batch": effective_batch,
        "micro_pair_batch": micro_batch,
        "accumulation_steps": accumulate,
        "scaled_weight_decay": scaled_decay,
        "yolov5_hyp_after_scaling": hyp,
        "pretrained_source": model.pretrained_source,
    }
    (out_dir / "config.resolved.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    deploy_parameters = sum(parameter.numel() for name, parameter in model.named_parameters()
                            if not name.startswith(("nfe_rgb.", "nfe_thermal.")))
    print(f"Pairs train/val={len(train_ds)}/{len(val_ds)}, device={device}, image=1280")
    print(f"Pair batch effective/micro={effective_batch}/{micro_batch}, accumulate={accumulate}")
    print(f"Parameters train/deploy={parameters:,}/{deploy_parameters:,}; AMP={amp}")
    print(f"Pretrained transfer: {model.pretrained_source}")
    print(f"Optimizer=SGD, lr0={tr['lr0']}, momentum={tr['momentum']}, decay={scaled_decay}")

    log_path = out_dir / "history.csv"
    batches_per_epoch = len(train_loader)
    warmup_iterations = max(round(float(tr.get("warmup_epochs", 3.0)) * batches_per_epoch), 100)
    best_metric = tr.get("best_metric", "val/rgbt/map50")
    for epoch in range(start_epoch, epochs):
        begin = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running: dict[str, float] = {}
        seen = 0
        for step, raw_batch in enumerate(train_loader):
            iteration = epoch * batches_per_epoch + step
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
            batch = move_batch(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                outputs = model.forward_pair(batch["rgb"], batch["thermal"])
                loss, stats = criterion(
                    outputs, batch["targets"], batch["rgb"].shape[-2:], batch["valid_mask"])
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
                    print("AMP overflow: optimizer/EMA update skipped", flush=True)
                optimizer.zero_grad(set_to_none=True)
            batch_size = batch["rgb"].shape[0]
            for key, value in stats.items():
                running[key] = running.get(key, 0.0) + float(value) * batch_size
            seen += batch_size
            log_every = int(tr.get("log_every", 200))
            if log_every and ((step + 1) % log_every == 0 or step + 1 == batches_per_epoch):
                print(
                    f"epoch {epoch + 1}/{epochs} step {step + 1}/{batches_per_epoch} "
                    f"loss={float(stats['loss']):.4f} det={float(stats['det_sum']):.4f} "
                    f"fs={float(stats['fs']):.4f} lr={optimizer.param_groups[0]['lr']:.6g}",
                    flush=True)
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            **{key: value / seen for key, value in running.items()},
        }
        improved = False
        if (epoch + 1) % int(tr.get("eval_every", 1)) == 0 or epoch + 1 == epochs:
            results = evaluate(ema.ema, val_loader, device, cfg.get("postprocess", {}), amp)
            (out_dir / f"val_epoch_{epoch + 1:03d}.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8")
            for mode, values in results.items():
                for key in ("map50", "map50_95"):
                    row[f"val/{mode}/{key}"] = values[key]
            score = metric_value(row, best_metric)
            improved = score > best_score
            if improved:
                best_score = score
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
            global_update, loader_generator, val_generator)
        atomic_save(payload, out_dir / "last.pt")
        if improved:
            shutil.copy2(out_dir / "last.pt", out_dir / "best.pt")
        if (epoch + 1) % int(tr.get("save_every", 5)) == 0:
            shutil.copy2(out_dir / "last.pt", out_dir / f"epoch_{epoch + 1:03d}.pt")
        print(
            f"epoch {epoch + 1} saved; elapsed={row['seconds']:.1f}s; "
            f"best {best_metric}={best_score:.4f}", flush=True)
    print(f"Finished. Training checkpoint: {out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
