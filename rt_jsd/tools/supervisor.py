"""Run the detector and its separately trained calibrator in the same pinned container."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def stamp():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat()


def state(status, **fields):
    p = ROOT / "pipeline_status.json"
    content = json.loads(p.read_text()) if p.exists() else {}
    content.update(status=status, updated_at=stamp(), **fields)
    temporary = p.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n")
    temporary.replace(p)


def execute(command, log_name):
    with (ROOT / log_name).open("a", buffering=1) as log:
        log.write(f"\n[{stamp()}] Starting {command!r}\n")
        process = subprocess.Popen(command, cwd=ROOT / "runtime", env=os.environ.copy(),
                                   stdout=log, stderr=subprocess.STDOUT)
        state("running", child_pid=process.pid, current_log=str(ROOT / log_name))
        code = process.wait()
        log.write(f"[{stamp()}] Exit code {code}\n")
    if code:
        raise subprocess.CalledProcessError(code, command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume")
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()
    config = ROOT / "config.json"
    try:
        if not args.calibration_only:
            state("running", stage="detector", started_at=stamp(), target_epochs=json.loads(config.read_text())["training"]["epochs"])
            command = [sys.executable, "-u", "-B", "-m", "torch.distributed.run",
                       "--nnodes=1", "--nproc_per_node=2", "--rdzv_backend=static",
                       "--master_addr=127.0.0.1", "--master_port=29611", "--max_restarts=0",
                       str(ROOT / "tools/train_ddp.py"), "--config", str(config)]
            if args.resume:
                command.extend(["--resume", args.resume])
            execute(command, "training.log")
        if not (ROOT / "training/best.pt").is_file():
            raise FileNotFoundError("Detector's validation-selected best.pt is required")
        cfg = json.loads(config.read_text())
        if cfg.get("calibration", {}).get("enabled", True):
            state("running", stage="calibrator_and_test", detector_completed_at=stamp())
            execute([sys.executable, "-u", "-B", str(ROOT / "tools/calibrate.py"),
                     "--config", str(config), "--device", "cuda:0", "--stage", "all"],
                    "calibration.log")
        else:
            assert not args.calibration_only, "Calibration is disabled in this frozen configuration"
            state("running", stage="detector_only_test", detector_completed_at=stamp())
            execute([sys.executable, "-u", "-B", str(ROOT / "tools/test_detector.py"),
                     "--config", str(config), "--device", "cuda:0"], "test.log")
        state("completed", stage="completed", completed_at=stamp(), child_pid=None)
    except BaseException as error:
        state("failed", error=repr(error), child_pid=None)
        raise


if __name__ == "__main__":
    main()
