"""Matched 200-epoch GT/JSD detector with RGB People/Lamp/Motorcycle ranking."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
from datetime import datetime, timedelta, timezone
import functools
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
COMMON_TOOLS = ROOT / "tools"
sys.path.insert(0, str(COMMON_TOOLS))

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, Subset, get_worker_info

from uod.data import assert_disjoint, collate_pairs
from uod.engine import atomic_save, build_dataset, make_loader, move_batch, seed_all, seed_worker
from uod.geometry import undo_letterbox
from uod.localization_distribution import DistributionDetect
from uod.localization_loss import DistributionUODLoss
from focus_objective import scheduled_ranking_beta
from p2_objective import P2FocusObjective as FocusObjective, active_gpu_check
from uod.p2_coupled import install_p2, matched_hyp
from uod.metrics import DetectionAP
from uod.reproducibility import (batch_fingerprint, configure_determinism,
    file_sha256, numpy_rng_state, restore_rng, state_equal)
from uod.soft_nms import linear_soft_nms
from uod.yolov5_official import ModelEMA, OfficialYOLOv5UOD, smart_optimizer
from train_yolov5_uod import checkpoint_state, metric_value, scaled_yolov5_hyp, scheduled_feature_alpha
from utils.general import xywh2xyxy


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stamp():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def validate_config(cfg, world=2):
    tr = cfg["training"]
    assert world == cfg["distributed"]["world_size"] == 2
    assert tr["epochs"] == tr["lr_schedule_epochs"] > 0
    assert cfg["data"]["image_size"] == 640
    assert tr["micro_batch_size"] > 0 and tr["batch_size"] >= world * tr["micro_batch_size"]
    assert tr["batch_size"] % (world * tr["micro_batch_size"]) == 0
    assert tr["batch_size"] // (world * tr["micro_batch_size"]) == cfg["distributed"]["accumulation_steps"]
    assert cfg["experiment"]["train_pairs"] % tr["batch_size"] == 0
    assert tr["optimizer"] == "SGD" and tr["lr0"] == .01
    assert tr["eval_every"] == 1 and tr["retain_periodic"] == 2
    assert tr["best_metric"] in {"val/unimodal_mean/map50_95", "val/rgbt/map50_95", "val/rgbt/map50", "val/unimodal_mean/map50"}
    assert float(cfg["loss"]["alpha"]) >= 0
    assert cfg["loss"]["alpha_start_epoch"] == 3.
    assert cfg["loss"]["alpha_ramp_epochs"] == 5.
    ranking = cfg["loss"]["ranking"]
    assert ranking['class_ids'] == [0, 3, 4] and ranking['modality'] == 'rgb'
    assert ranking['beta'] == .1 and ranking['start_epoch'] == 3. and ranking['ramp_epochs'] == 5.
    assert not cfg['calibration']['enabled'] and tr['epochs'] == 200
    loc = cfg["loss"]["localization"]
    assert loc["min_teacher_iou"] == .7 and loc["min_iou_gap"] == .1
    assert loc["edge_gate"] == "gt_better"
    assert cfg['model']['p2'] == dict(enabled=True, channels=128, initialization_seed=2026, anchor_scale=.5)
    assert not cfg["model"]["use_nfe"] and cfg["model"]["separate_ife"]
    assert cfg["postprocess"]["method"] == "soft_nms"
    assert cfg["postprocess"]["soft_nms_method"] == "linear"
    if float(cfg["augmentation"].get("mosaic_probability", 0.)) > 0:
        assert (COMMON_TOOLS / "paired_mosaic.py").is_file(), "Configured paired Mosaic module is not prepared; training must not silently omit it"


class OriginalMicroBatches:
    """Alternate whole local batches without duplicating or dropping pairs."""
    def __init__(self, dataset, generator, micro, rank, world):
        self.dataset, self.generator, self.micro = dataset, generator, micro
        self.rank, self.world = rank, world
        assert len(dataset) % (micro * world) == 0

    def __iter__(self):
        indices = list(RandomSampler(self.dataset, generator=self.generator))
        for start in range(self.rank * self.micro, len(indices), self.world * self.micro):
            yield indices[start:start + self.micro]

    def __len__(self):
        return len(self.dataset) // (self.micro * self.world)


class PreflightMicroBatches:
    """Finite probe batches so worker IPC drains through normal exhaustion."""
    def __init__(self, sampler, steps):
        self.sampler, self.steps = sampler, int(steps)
        assert 0 < self.steps <= len(sampler)

    def __iter__(self):
        return itertools.islice(iter(self.sampler), self.steps)

    def __len__(self):
        return self.steps


def original_worker_seed(worker, rank, world):
    base = get_worker_info().seed - worker
    torch.manual_seed(base + worker * world + rank)
    seed_worker(worker)


def summed_gradient_hook(state, bucket):
    # Criterion returns an image SUM, so rank gradients must SUM as well.
    # Avoid a loss*world multiplier that increases AMP activation gradients.
    return dist.all_reduce(bucket.buffer(), op=dist.ReduceOp.SUM,
        async_op=True).get_future().then(lambda future: future.value()[0])


class OriginalBNHistory:
    """Local BN for each four-pair forward; merge running updates in rank order."""
    def __init__(self, model, world):
        self.modules = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        assert self.modules and all(m.momentum is not None and m.track_running_stats for m in self.modules)
        self.world = world
        self.momenta = torch.cat([torch.full((2 * m.num_features,), float(m.momentum),
            device=m.running_mean.device) for m in self.modules])
        self.checked = False

    def pack(self):
        return torch.cat([v for m in self.modules for v in (m.running_mean, m.running_var)])

    @torch.no_grad()
    def before(self):
        self.previous = self.pack().clone()
        self.counts = torch.stack([m.num_batches_tracked.clone() for m in self.modules])

    @torch.no_grad()
    def after_backward(self):
        current = self.pack()
        if not self.checked:
            counts = torch.stack([m.num_batches_tracked for m in self.modules])
            assert torch.equal(counts, self.counts + 1), "Each BN must run once per paired local forward"
            self.checked = True
        gathered = [torch.empty_like(current) for _ in range(self.world)]
        dist.all_gather(gathered, current)
        decay = 1 - self.momenta
        merged = self.previous.clone()
        for value in gathered:
            contribution = value - decay * self.previous
            merged = decay * merged + contribution
        offset = 0
        for i, module in enumerate(self.modules):
            for buffer in (module.running_mean, module.running_var):
                size = buffer.numel()
                buffer.copy_(merged[offset:offset + size])
                offset += size
            module.num_batches_tracked.copy_(self.counts[i] + self.world)


class PairLoss(nn.Module):
    def __init__(self, detector, criterion):
        super().__init__()
        self.detector, self.criterion = detector, criterion

    def forward(self, batch):
        output = self.detector.forward_pair(batch["rgb"], batch["thermal"])
        return self.criterion(output, batch["targets"], batch["rgb"].shape[-2:], batch["valid_mask"])


def rank_rng(loader, val, probe):
    return dict(python_rng=random.getstate(), numpy_rng=numpy_rng_state(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
        loader_rng=loader.get_state(), val_loader_rng=val.get_state(), probe_loader_rng=probe.get_state())


def candidate_pool(decoded, postprocess):
    """Same uncorrected-score pool as the recent VFL Test: one class per box."""
    scores, labels = (decoded[:, 4:5].float() * decoded[:, 5:].float()).max(1)
    keep = scores > float(postprocess["score_threshold"])
    pool = torch.cat((xywh2xyxy(decoded[keep, :4].float()),
        scores[keep, None], labels[keep, None].float()), 1)
    if len(pool):
        pool = pool[pool[:, 4].argsort(descending=True)[:int(postprocess["max_nms"])]]
    return pool


@torch.no_grad()
def soft_postprocess(decoded, meta, postprocess):
    # Match the frozen-detector calibration/Test path, including score ties.
    pool = candidate_pool(decoded.cpu(), postprocess).numpy()
    result = torch.from_numpy(linear_soft_nms(pool, float(postprocess["iou_threshold"]),
        float(postprocess["score_threshold"]), int(postprocess["max_detections"])))
    return dict(boxes=undo_letterbox(result[:, :4], meta), scores=result[:, 4], labels=result[:, 5].long())


@torch.no_grad()
def distributed_evaluate(model, dataset, cfg, rank, world, device):
    model.eval()
    batch_size = int(cfg["training"]["val_batch_size"])
    indices = [i for block in range(math.ceil(len(dataset) / batch_size)) if block % world == rank
        for i in range(block * batch_size, min((block + 1) * batch_size, len(dataset)))]
    loader = make_loader(Subset(dataset, indices), batch_size, 0, False, device,
        torch.Generator().manual_seed(int(cfg["seed"]) + 1))
    metrics = {mode: DetectionAP(model.nc) for mode in ("rgb", "thermal", "rgbt")}
    for raw in loader:
        batch = move_batch(raw, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=bool(cfg["training"]["amp"])):
            output = model.forward_pair(batch["rgb"], batch["thermal"])
        rgb, thermal = output["decoded_rgb"].float(), output["decoded_thermal"].float()
        for mode, decoded in dict(rgb=rgb, thermal=thermal, rgbt=torch.cat((rgb, thermal), 1)).items():
            formatted = [soft_postprocess(image, meta, cfg["postprocess"])
                for image, meta in zip(decoded, raw["meta"], strict=True)]
            metrics[mode].update(formatted, raw["original_targets"])
        del output, batch, rgb, thermal
    payload = dict(indices=indices, predictions={m: v.predictions for m, v in metrics.items()},
        targets=metrics["rgb"].targets)
    gathered = [None] * world
    dist.all_gather_object(gathered, payload)
    if rank:
        return None
    ordered = sorted((index, r, j) for r, part in enumerate(gathered) for j, index in enumerate(part["indices"]))
    assert [i for i, _, _ in ordered] == list(range(len(dataset)))
    results = {}
    for mode in metrics:
        merged = DetectionAP(model.nc)
        merged.predictions = [gathered[r]["predictions"][mode][j] for _, r, j in ordered]
        merged.targets = [gathered[r]["targets"][j] for _, r, j in ordered]
        results[mode] = merged.compute()
    return results


def copy_checkpoint(source, destination):
    """Atomic hard link avoids storing identical full checkpoints twice."""
    source, destination = Path(source), Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".linktmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    temporary.replace(destination)


def build_training_components(cfg, device):
    model = OfficialYOLOv5UOD.from_pretrained(cfg["model"]["weights"], cfg["data"]["class_names"],
        use_nfe=False, separate_ife=True, feature_objective="hinge")
    source = Path(cfg["model"]["initial_checkpoint"])
    assert file_sha256(source) == cfg["model"]["initial_checkpoint_sha256"]
    initial = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    assert initial["epoch"] == -1 and initial["global_update"] == 0
    # Deliberately MODEL ONLY: old optimizer/scheduler/EMA must never override
    # this experiment's explicit batch, LR, actual WD and schedule.
    model.load_state_dict(initial["state_dict"], strict=True)
    del initial
    distribution = cfg["model"]["localization_distribution"]
    model.ifd = DistributionDetect(model.ifd, **distribution)
    install_p2(model, **cfg['model']['p2'])
    model.stride = model.ifd.stride
    model.pretrained_source = {**model.pretrained_source,
        "initial_checkpoint": str(source),
        "initial_checkpoint_sha256": cfg["model"]["initial_checkpoint_sha256"],
        "localization_distribution": distribution,
        "distribution_initialization": "zero regression weights; Gaussian bias centered at .5; object/class rows copied",
        "initialization_policy": "common pretrained MODEL ONLY; optimizer/EMA/scaler/scheduler freshly constructed"}
    model.hyp = matched_hyp(cfg, model)
    model.to(device)
    tr = cfg["training"]
    criterion = FocusObjective(model, cfg)
    decay = float(tr["weight_decay"]) * int(tr["batch_size"]) / int(tr["nominal_batch_size"])
    optimizer = smart_optimizer(model, "SGD", float(tr["lr0"]), float(tr["momentum"]), decay)
    horizon = int(tr["lr_schedule_epochs"])
    lr_factor = lambda epoch: (1 - epoch / horizon) * (1 - float(tr["lrf"])) + float(tr["lrf"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(tr["amp"]))
    ema = ModelEMA(model)
    return model, criterion, optimizer, scheduler, scaler, ema, lr_factor, decay


@torch.no_grad()
def sync_constant_geometry(model, ema):
    """Keep fixed decoding geometry exact across EMA updates and reloads."""
    for key in ("anchors", "support"):
        getattr(ema.ema.ifd, key).copy_(getattr(model.ifd, key))


def train(args):
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    cfg = read(args.config)
    validate_config(cfg, world)
    config_hash = file_sha256(args.config)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.set_num_threads(2)
    backend = configure_determinism(bool(cfg["deterministic"]))
    seed_all(int(cfg["seed"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    tr = cfg["training"]
    out = Path(tr["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    preflight = int(args.preflight_steps or 0) > 0
    if not args.resume and not preflight:
        assert not (out / "last.pt").exists() and not (out / "initial.pt").exists(), "Never overwrite an existing run"
    generators = [torch.Generator().manual_seed(int(cfg["seed"]) + i) for i in range(3)]
    loader_gen, val_gen, probe_gen = generators
    train_ds, val_ds = build_dataset(cfg, "train", True), build_dataset(cfg, "val", False)
    assert len(train_ds) == cfg["experiment"]["train_pairs"] == 3360
    assert len(val_ds) == cfg["experiment"]["val_pairs"] == 420
    assert_disjoint(train_ds, val_ds)
    if float(cfg["augmentation"].get("mosaic_probability", 0.)) > 0:
        from paired_mosaic import wrap_dataset
        train_ds = wrap_dataset(train_ds, cfg)
        assert callable(getattr(train_ds, "set_epoch", None))
    sampler = OriginalMicroBatches(train_ds, loader_gen, int(tr["micro_batch_size"]), rank, world)
    sampled_batches = PreflightMicroBatches(sampler, args.preflight_steps) if preflight else sampler
    workers = int(cfg["num_workers"]) // world
    loader_kwargs = dict(batch_sampler=sampled_batches, num_workers=workers, collate_fn=collate_pairs,
        pin_memory=True, generator=loader_gen,
        worker_init_fn=functools.partial(original_worker_seed, rank=rank, world=world),
        persistent_workers=False)
    if workers:
        loader_kwargs.update(multiprocessing_context="spawn", timeout=120, prefetch_factor=1)
    loader = DataLoader(train_ds, **loader_kwargs)
    model, criterion, optimizer, scheduler, scaler, ema, lr_factor, decay = build_training_components(cfg, device)
    primary = tr["best_metric"]
    tracked = list(dict.fromkeys([primary, *tr.get("additional_best_metrics", [])]))
    best_scores = {name: -1. for name in tracked}
    start, updates = 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)
        assert state["kind"] == "uod_yolov5_training" and state["distributed_world_size"] == world
        assert state["config_sha256"] == config_hash, "Resume requires the identical frozen experiment config"
        assert state["backend"] == backend
        model.load_state_dict(state["state_dict"], strict=True)
        ema.ema.load_state_dict(state["ema_state_dict"], strict=True)
        ema.updates = int(state["ema_updates"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        restore_rng(dict(state["rank_rng"][rank], backend=backend), *generators, strict=True)
        start, updates = int(state["epoch"]) + 1, int(state["global_update"])
        best_scores = dict(state["best_scores"])
        checks = dict(model=state_equal(model.state_dict(), state["state_dict"]),
            ema=state_equal(ema.ema.state_dict(), state["ema_state_dict"]),
            optimizer=state_equal(optimizer.state_dict(), state["optimizer"]),
            scheduler=state_equal(scheduler.state_dict(), state["scheduler"]),
            scaler=state_equal(scaler.state_dict(), state["scaler"]),
            rank_rng=state_equal(rank_rng(*generators), state["rank_rng"][rank]))
        assert all(checks.values()), str(checks)
        write(ROOT / "reports" / f"resume_epoch{start:03d}_rank{rank}.json",
            dict(status="passed", checks=checks, source=str(args.resume), sha256=file_sha256(args.resume)))
        del state
    cfg["runtime"] = dict(effective_pair_batch=int(tr["batch_size"]),
        micro_pair_batch_per_gpu=int(tr["micro_batch_size"]), world_size=world,
        accumulation_steps=int(cfg["distributed"]["accumulation_steps"]), scaled_weight_decay=decay, lr_schedule_epochs=int(tr["lr_schedule_epochs"]),
        yolov5_hyp_after_scaling=model.hyp, backend=backend,
        pretrained_source=model.pretrained_source, ddp_gradient_reduction="SUM",
        batchnorm=f"local {tr['micro_batch_size']}-pair statistics; ordered sequential running-buffer merge",
        ema_constant_geometry="fixed anchors/support copied exactly, never EMA-averaged")
    wrapper = DDP(PairLoss(model, criterion), device_ids=[rank], output_device=rank,
        broadcast_buffers=False, gradient_as_bucket_view=True, find_unused_parameters=False)
    wrapper.register_comm_hook(None, summed_gradient_hook)
    bn = OriginalBNHistory(model, world)
    global_batches = len(sampler) * world
    assert global_batches * int(tr["micro_batch_size"]) == len(train_ds)
    accumulation = int(cfg["distributed"]["accumulation_steps"])
    assert len(loader) % accumulation == 0
    if preflight:
        assert int(args.preflight_steps) % accumulation == 0
    warmup_iterations = max(round(float(tr["warmup_epochs"]) * global_batches), 100)
    if rank == 0:
        print(f"M3FD JSD | fresh initialization | split3360/420/420 | SGD lr0={tr['lr0']} "
            f"actualWD={decay:.8g} | {tr['micro_batch_size']} pairs/GPU x{world} GPUs x{accumulation}accum="
            f"{tr['batch_size']} global pairs | {tr['epochs']} epochs/schedule | EMA linear Soft-NMS", flush=True)
        print("Pretrained provenance:", json.dumps(model.pretrained_source), flush=True)
        print("Optimizer initial groups:", [dict(lr=g["lr"], initial_lr=g["initial_lr"],
            weight_decay=g["weight_decay"], momentum=g["momentum"], nesterov=g["nesterov"])
            for g in optimizer.param_groups], flush=True)
        print(f"Original JSD: alpha={cfg['loss']['alpha']} start3 ramp5, object Teacher IoU>=.7 gap>=.1, gt_better edges, full-network gradients; alpha0 means no cross-modal update", flush=True)
        print('P2-P5 coupled anchor-based25; original P3-P5 paths/gains/balances retained; P2 obj balance4', flush=True)
        print('RGB ranking: People/Lamp/Motorcycle, beta=.1 start3 ramp5; uniform per GT; full model trainable; no calibrator or extra detector epochs', flush=True)
        if not preflight:
            write(out / "config.resolved.json", cfg)
    if not args.resume and not preflight:
        rank_states = [None] * world
        dist.all_gather_object(rank_states, rank_rng(*generators))
        if rank == 0:
            assert not optimizer.state and ema.updates == 0
            assert all(g["lr"] == .01 for g in optimizer.param_groups)
            assert scheduler.last_epoch == 0
            payload = checkpoint_state(model, ema, optimizer, scheduler, scaler, cfg, -1,
                -1., 0, *generators, best_scores=best_scores, backend=backend)
            payload.update(rank_rng=rank_states, distributed_world_size=world, config_sha256=config_hash)
            atomic_save(payload, out / "initial.pt")
            del payload
            write(ROOT / "reports" / "initial_state_verification.json", dict(status="passed",
                completed_epoch=0, global_update=0, fresh_optimizer=True, fresh_ema=True,
                optimizer_empty=True, initial_lrs=[g["lr"] for g in optimizer.param_groups],
                scaled_weight_decay=decay, scheduler_horizon=int(tr["lr_schedule_epochs"]),
                initial_checkpoint=str(out / "initial.pt"), sha256=file_sha256(out / "initial.pt"),
                common_model_only_source=cfg["model"]["initial_checkpoint"], config_sha256=config_hash))
        dist.barrier()
    history = out / "history.csv"
    audited = 0
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start, int(tr["epochs"])):
        begin = time.monotonic()
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch + 1)
        wrapper.train()
        optimizer.zero_grad(set_to_none=True)
        running, seen = {}, 0
        lower, upper = float("inf"), -float("inf")
        loader_iterator = iter(loader)
        for step, raw in enumerate(loader_iterator):
            original_step = step * world + rank
            original_iteration = epoch * global_batches + original_step
            criterion.alpha = scheduled_feature_alpha(cfg["loss"], (original_iteration + 1) / global_batches)
            criterion.beta = scheduled_ranking_beta(cfg['loss']['ranking'], (original_iteration + 1) / global_batches)
            final_iteration = epoch * global_batches + (step + 1) * world - 1
            if final_iteration <= warmup_iterations:
                progress = final_iteration / warmup_iterations
                target = float(tr["lr0"]) * lr_factor(epoch)
                for i, group in enumerate(optimizer.param_groups):
                    initial_lr = float(tr["warmup_bias_lr"]) if i == 0 else 0.
                    group["lr"] = initial_lr + progress * (target - initial_lr)
                    group["momentum"] = float(tr["warmup_momentum"]) + progress * (
                        float(tr["momentum"]) - float(tr["warmup_momentum"]))
            if not preflight and audited < int(tr.get("audit_batches", 0)) // world:
                with (out / f"batch_audit_rank{rank}.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(epoch=epoch + 1, step=original_step + 1,
                        **batch_fingerprint(raw))) + "\n")
                audited += 1
            batch = move_batch(raw, device)
            bn.before()
            update_now = (step + 1) % accumulation == 0
            sync = contextlib.nullcontext() if update_now else wrapper.no_sync()
            with sync:
                with torch.autocast("cuda", dtype=torch.float16, enabled=bool(tr["amp"])):
                    loss, stats = wrapper(batch)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite loss E{epoch + 1} rank{rank} step{step + 1}")
                scaler.scale(loss).backward()
            bn.after_backward()
            if update_now:
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr["grad_clip"]),
                    error_if_nonfinite=not bool(tr["amp"]))
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < previous_scale
                if not skipped:
                    ema.update(model)
                    # Fixed decoding geometry must not slowly drift due to FP32
                    # EMA roundoff. This also makes a fresh EMA reload reproduce
                    # the exact grid used during Validation.
                    sync_constant_geometry(model, ema)
                    updates += 1
                if rank == 0 and (step < 8 or skipped) and not preflight:
                    with (out / "optimizer_events.jsonl").open("a") as stream:
                        stream.write(json.dumps(dict(epoch=epoch + 1, step=(step + 1) * world,
                            global_update=updates, skipped=skipped, scale_before=previous_scale,
                            scale_after=scaler.get_scale(), gradient_norm=float(norm) if bool(torch.isfinite(norm)) else None,
                            lr=[g["lr"] for g in optimizer.param_groups])) + "\n")
                optimizer.zero_grad(set_to_none=True)
            count = len(raw["meta"])
            seen += count
            for key, value in stats.items():
                running[key] = running.get(key, 0.) + float(value) * count
            running["feature_alpha"] = running.get("feature_alpha", 0.) + criterion.alpha * count
            lower, upper = min(lower, float(stats["target_support_min"])), max(upper, float(stats["target_support_max"]))
            if rank == 0 and (step == 0 or (step + 1) % int(tr["log_every"]) == 0 or step + 1 == len(loader)):
                print(f"epoch {epoch + 1}/{tr['epochs']} step {(step + 1) * world}/{global_batches} "
                    f"loss={float(stats['loss']):.4f} det={float(stats['det_sum']):.4f} "
                    f"jsd={float(stats['fs']):.5f} dfl={float(stats['dfl']):.5f} alpha={criterion.alpha:.5g} "
                    f"rank={float(stats['ranking']):.5f} beta={criterion.beta:.5g} "
                    f"RGB->T edges/pair={float(stats['selected_rgb_teacher_edges_per_pair']):.3f} "
                    f"T->RGB edges/pair={float(stats['selected_thermal_teacher_edges_per_pair']):.3f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6g} updates={updates}", flush=True)
            if preflight and update_now and step + 1 >= int(args.preflight_steps):
                # No checkpoint or detector output is committed by this probe.
                # The probe sampler has exactly the requested batch count.
                # Public iterator exhaustion drains worker/pin-memory IPC
                # instead of interrupting a full epoch's prefetched tensors.
                try:
                    next(loader_iterator)
                except StopIteration:
                    pass
                else:
                    raise AssertionError("Preflight loader must end at the requested step")
                active_ranking = active_gpu_check(model, criterion, batch, cfg)
                new_groups = {"RGB_P2": list(model.p2_ife_rgb.parameters()), "Thermal_P2": list(model.p2_ife_thermal.parameters()), "shared_P2_predictor": list(model.ifd.m[0].parameters())}
                momentum = {}
                for label, parameters in new_groups.items():
                    buffers = [optimizer.state[p]["momentum_buffer"] for p in parameters]
                    assert all(torch.isfinite(x).all() for x in buffers)
                    momentum[label] = sum(float(x.double().square().sum()) for x in buffers) ** .5
                    assert momentum[label] > 0
                active_ranking["P2_native_SGD_momentum_norms"] = momentum
                module = model.ifd
                module.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=bool(tr["amp"])):
                    model.eval()
                    prediction = model.forward_pair(batch["rgb"][:1], batch["thermal"][:1])
                decoded = torch.cat((prediction["decoded_rgb"][0].float(),
                    prediction["decoded_thermal"][0].float()), 0)
                post = soft_postprocess(decoded, raw["meta"][0], cfg["postprocess"])
                assert len(post["scores"]) <= cfg["postprocess"]["max_detections"]
                assert bool(torch.isfinite(post["boxes"]).all())
                assert updates > 0, "GPU preflight must include an actual successful SGD/EMA update"
                report = dict(status="passed", rank=rank, real_paired_steps=step + 1,
                    successful_optimizer_updates=updates, local_pairs=tr["micro_batch_size"], global_pairs=tr["batch_size"],
                    finite_training_loss=True, finite_soft_nms=True, probe_saved_checkpoint=False,
                    active_ranking=active_ranking,
                    loader_workers_shutdown_cleanly=True,
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024 ** 3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024 ** 3,
                    config_sha256=config_hash,
                    trainer_sha256=file_sha256(Path(__file__).resolve()),
                    paired_mosaic_sha256=file_sha256(COMMON_TOOLS / "paired_mosaic.py"),
                    objective_sha256=file_sha256(COMMON_TOOLS / 'focus_objective.py'),
                    ranking_sha256=file_sha256(ROOT / 'runtime/uod/focus_ranking.py'),
                    p2_sha256=file_sha256(ROOT / 'runtime/uod/p2_coupled.py'),
                    p2_objective_sha256=file_sha256(ROOT / 'tools/p2_objective.py'))
                write(ROOT / "reports" / f"gpu_preflight_rank{rank}.json", report)
                dist.barrier()
                if rank == 0:
                    write(ROOT / "reports" / "gpu_preflight.json", dict(status="passed",
                        ranks=[read(ROOT / "reports" / f"gpu_preflight_rank{r}.json") for r in range(world)]))
                    print(f"GPU preflight passed: real local{tr['micro_batch_size']}/global{tr['batch_size']} forward, SUM backward, SGD/AMP/EMA, BN, Soft-NMS; no training state saved", flush=True)
                dist.destroy_process_group()
                return
            del loss, stats, batch
        scheduler.step()
        keys = sorted(running)
        totals = torch.tensor([running[k] for k in keys] + [seen], dtype=torch.float64, device=device)
        dist.all_reduce(totals)
        bounds = torch.tensor([lower, -upper], device=device)
        dist.all_reduce(bounds, op=dist.ReduceOp.MIN)
        assert int(totals[-1]) == len(train_ds)
        row = dict(epoch=epoch + 1, lr=optimizer.param_groups[0]["lr"],
            **{k: float(totals[i] / totals[-1]) for i, k in enumerate(keys)})
        row.update(target_support_min=float(bounds[0]), target_support_max=-float(bounds[1]))
        if rank == 0:
            print(f"epoch {epoch + 1}: validating all {len(val_ds)} pairs with EMA and linear Soft-NMS", flush=True)
        results = distributed_evaluate(ema.ema, val_ds, cfg, rank, world, device)
        rank_states = [None] * world
        dist.all_gather_object(rank_states, rank_rng(*generators))
        if rank == 0:
            write(out / f"val_epoch_{epoch + 1:03d}.json", results)
            for mode, metrics in results.items():
                for key in ("map50", "map75", "map50_95"):
                    row[f"val/{mode}/{key}"] = metrics[key]
            improved = []
            for name in tracked:
                score = metric_value(row, name)
                if score > best_scores[name]:
                    best_scores[name] = score
                    improved.append(name)
            row["seconds"] = time.monotonic() - begin
            new = not history.exists()
            with history.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if new:
                    writer.writeheader()
                writer.writerow(row)
            payload = checkpoint_state(model, ema, optimizer, scheduler, scaler, cfg, epoch,
                best_scores[primary], updates, *generators, best_scores=best_scores, backend=backend)
            payload.update(rank_rng=rank_states, distributed_world_size=world, config_sha256=config_hash)
            atomic_save(payload, out / "last.pt")
            del payload
            for name in improved:
                filename = "best_" + name.replace("val/", "").replace("/", "_") + ".pt"
                copy_checkpoint(out / "last.pt", out / filename)
                if name == primary:
                    copy_checkpoint(out / "last.pt", out / "best.pt")
            if (epoch + 1) % int(tr["save_every"]) == 0 or epoch + 1 in tr.get("checkpoint_epochs", []):
                copy_checkpoint(out / "last.pt", out / f"epoch_{epoch + 1:03d}.pt")
                periodic = sorted(out.glob("epoch_[0-9][0-9][0-9].pt"))
                for obsolete in periodic[:-int(tr["retain_periodic"])]:
                    obsolete.unlink()
            print("validation:", {m: {k: v[k] for k in ("map50", "map75", "map50_95")}
                for m, v in results.items()}, flush=True)
            print(f"epoch {epoch + 1} saved; elapsed={row['seconds']:.1f}s; Best {primary}={best_scores[primary] * 100:.3f}%", flush=True)
            write(out / "status.json", dict(status="training" if epoch + 1 < tr["epochs"] else "completed",
                completed_epoch=epoch + 1, target_epochs=tr["epochs"], global_update=updates,
                best_metric=primary, best_score=best_scores[primary], best_scores=best_scores,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024 ** 3, updated_at_kst=stamp()))
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--preflight-steps", type=int, default=0,
        help="Run disposable real GPU training steps, postprocess once, write only preflight reports")
    train(parser.parse_args())
