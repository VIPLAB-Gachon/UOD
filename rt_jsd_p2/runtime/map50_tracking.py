"""Supplemental mAP50 selection, independent of training checkpoint state."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil

import torch


class Map50Tracker:
    """Select one epoch by mean RGB/Thermal mAP50; keep the earliest tie."""

    filenames = {
        "global": "best_unimodal_mean_map50.pt",
        "post_warmup": "best_post_warmup_unimodal_mean_map50.pt",
    }

    def __init__(self, out_dir: Path, history_path: Path, completed_epoch: int,
                 warmup_epochs: int = 3):
        self.out_dir = Path(out_dir)
        self.warmup_epochs = warmup_epochs
        self.best = {scope: None for scope in self.filenames}
        self.available = {scope: False for scope in self.filenames}
        if history_path.exists():
            with history_path.open(newline="", encoding="utf-8") as handle:
                rows = sorted(csv.DictReader(handle), key=lambda row: int(row["epoch"]))
            for row in rows:
                if 1 <= int(row["epoch"]) <= completed_epoch:
                    self._select(row)

        # A legacy best.pt can initialize the mAP50 best only if its actual
        # checkpoint epoch matches the winner computed from inherited history.
        epoch_cache = {}
        for scope, filename in self.filenames.items():
            record = self.best[scope]
            if record is None:
                continue
            candidates = [self.out_dir / filename,
                          self.out_dir / f"epoch_{record['epoch']:03d}.pt",
                          self.out_dir / "last.pt", self.out_dir / "best.pt",
                          *sorted(self.out_dir.glob("best*.pt"))]
            for path in dict.fromkeys(candidates):
                if not path.exists():
                    continue
                if path not in epoch_cache:
                    state = torch.load(path, map_location="cpu", weights_only=True)
                    epoch_cache[path] = int(state["epoch"]) + 1
                    del state
                if epoch_cache[path] == record["epoch"]:
                    self._copy_checkpoint(path, self.out_dir / filename)
                    epoch_cache[self.out_dir / filename] = record["epoch"]
                    self.available[scope] = True
                    break
        self._write_selection()

    @staticmethod
    def _record(row: dict):
        values = {}
        for mode in ("rgb", "thermal", "rgbt"):
            value = row.get(f"val/{mode}/map50")
            values[f"{mode}_map50"] = None if value in (None, "") else float(value)
        if any(values[f"{mode}_map50"] is None for mode in ("rgb", "thermal")):
            return None
        if any(value is not None and not math.isfinite(value) for value in values.values()):
            raise ValueError("mAP50 selection requires finite validation metrics")
        return {"epoch": int(row["epoch"]),
                "score": .5 * (values["rgb_map50"] + values["thermal_map50"]),
                **values}

    def _select(self, row: dict):
        record = self._record(row)
        if record is None:
            return []
        improved = []
        for scope in self.filenames:
            if scope == "post_warmup" and record["epoch"] <= self.warmup_epochs:
                continue
            current = self.best[scope]
            if current is None or record["score"] > current["score"]:
                self.best[scope] = dict(record)
                improved.append(scope)
        return improved

    @staticmethod
    def _copy_checkpoint(source: Path, destination: Path):
        if source == destination:
            return
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)

    def update(self, row: dict, saved_checkpoint_epoch: int):
        """Call after atomic_save(last.pt); its epoch is zero-based, as in trainer."""
        if int(row["epoch"]) != saved_checkpoint_epoch + 1:
            raise ValueError("Validation row and saved checkpoint must have the same epoch")
        checkpoint_path = self.out_dir / "last.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        for scope in self._select(row):
            self._copy_checkpoint(checkpoint_path, self.out_dir / self.filenames[scope])
            self.available[scope] = True
        self._write_selection()

    def _write_selection(self):
        selection = {
            "metric": "val/unimodal_mean/map50",
            "formula": "(val/rgb/map50 + val/thermal/map50) / 2",
            "units": "fraction",
            "tie_break": "earliest epoch",
            "warmup_epochs": self.warmup_epochs,
        }
        for scope, filename in self.filenames.items():
            selection[scope] = {
                "start_epoch": 1 if scope == "global" else self.warmup_epochs + 1,
                "best": self.best[scope],
                "checkpoint_available": self.available[scope],
                "checkpoint": filename if self.available[scope] else None,
            }
        path = self.out_dir / "best_map50_selection.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
        temporary.replace(path)
