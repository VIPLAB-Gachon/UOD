"""Display the latest validation, AP50 Best and primary Best separately."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]


MODES = (("rgb", "RGB"), ("thermal", "Thermal"), ("rgbt", "RGBT"))
METRICS = ("map50", "map50_95")


def read_state(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        # A status update may be in progress while the viewer reads it.
        return {}


def metric_value(row, metric):
    if metric.startswith("val/unimodal_mean/"):
        key = metric.rsplit("/", 1)[1]
        return .5 * (float(row[f"val/rgb/{key}"]) + float(row[f"val/thermal/{key}"]))
    return float(row[metric])


def metric_label(metric):
    _, mode, key = metric.split("/", 2)
    mode = {"rgb": "RGB", "thermal": "Thermal", "rgbt": "RGBT",
            "unimodal_mean": "RGB/T mean"}.get(mode, mode)
    key = {"map50": "AP50", "map75": "AP75", "map50_95": "AP50-95"}.get(key, key)
    return f"{mode} {key}"


def validated_rows(history, metric):
    if not history.exists():
        return []
    valid = []
    with history.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                epoch = int(row["epoch"])
                values = [float(row[f"val/{mode}/{key}"])
                          for mode, _ in MODES for key in METRICS]
                values.append(metric_value(row, metric))
                if epoch > 0 and all(math.isfinite(value) for value in values):
                    valid.append(row)
            except (KeyError, TypeError, ValueError):
                # Do not display a partially appended validation row.
                continue
    return valid


def result_table(title, row, epochs):
    label = f"epoch {row['epoch']}/{epochs}" if row else "waiting for validation"
    lines = [f"{title} | {label}",
             f"  {'Mode':<9} {'AP50':>9} {'AP50-95':>11}"]
    for mode, name in MODES:
        values = [f"{100 * float(row[f'val/{mode}/{key}']):.2f}" if row else "--"
                  for key in METRICS]
        lines.append(f"  {name:<9} {values[0]:>9} {values[1]:>11}")
    return lines


def render(root=ROOT):
    cfg = json.loads((root / "config.json").read_text())
    metric = cfg["training"]["best_metric"]
    ap50_metric = metric.rsplit("/", 1)[0] + "/map50"
    epochs = cfg["training"]["epochs"]
    metric_name = metric_label(metric)
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")
    lines = [f"M3FD 8:1:1 | 640x640 | alpha {cfg['loss']['alpha']} | {stamp}",
             f"Detector EMA Validation (%); primary Best = {metric_name}"]
    state = read_state(root / "pipeline_status.json")
    valid = validated_rows(root / "training/history.csv", metric)
    progress = ""
    log = root / "training.log"
    if log.exists():
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 32768))
            matches = re.findall(r"epoch \d+/\d+ step \d+/\d+", stream.read().decode(errors="replace"))
        progress = matches[-1] if matches else ""
    lines.append(f"Stage: {state.get('stage', 'preparing')} | {state.get('status', 'preparing')} | {progress or 'waiting for first step'}")
    if valid:
        latest = max(valid, key=lambda row: int(row["epoch"]))
        best = max(valid, key=lambda row: (metric_value(row, metric), -int(row["epoch"])))
        best_ap50 = max(valid, key=lambda row: (metric_value(row, ap50_metric), -int(row["epoch"])))
    else:
        latest = best = best_ap50 = None
    lines += [""] + result_table("Current result (latest completed validation)", latest, epochs)
    lines += [""] + result_table(f"Best result ({metric_label(ap50_metric)})", best_ap50, epochs)
    lines += [""] + result_table(f"Best result ({metric_name})", best, epochs)
    if progress and state.get("stage") == "detector":
        running_epoch = int(re.search(r"epoch (\d+)", progress).group(1))
        if latest is None or running_epoch > int(latest["epoch"]):
            lines += ["", f"Epoch {running_epoch} AP: pending validation."]
    lines += ["", "Updates every 2s; Ctrl+C closes viewer only."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            result = render()
            print(("\033[2J\033[H" if args.watch else "") + result, flush=True)
            if not args.watch:
                break
            time.sleep(2)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
