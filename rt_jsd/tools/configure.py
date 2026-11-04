"""Resolve a portable template against local weights and fixed M3FD manifests."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    weights, initial, manifests = [p.expanduser().resolve() for p in (
        args.weights, args.initial_checkpoint, args.manifest_root)]
    for path in (weights, initial):
        if not path.is_file():
            parser.error("Required weights do not exist: " + str(path))
    if manifests.name != "m3fd_uod_811_seed2026":
        parser.error("Manifest directory must be named m3fd_uod_811_seed2026")
    all_ids = set()
    for split, count in (("train", 3360), ("val", 420), ("test", 420)):
        rows = [json.loads(line) for line in (manifests / (split + ".jsonl")).read_text().splitlines()
                if line.strip()]
        expected = (ROOT / "splits/m3fd_811_seed2026" / (split + ".txt")).read_text().splitlines()
        ids = [str(row["id"]) for row in rows]
        if len(ids) != count or len(set(ids)) != count or ids != expected:
            parser.error("Manifest IDs/order differ from the committed split: " + split)
        if all_ids.intersection(ids):
            parser.error("Manifest splits overlap")
        all_ids.update(ids)
    text = (ROOT / "configs/m3fd_640_811.template.json").read_text()
    text = text.replace("${PROJECT_ROOT}", str(ROOT)).replace("${MANIFEST_ROOT}", str(manifests))
    cfg = json.loads(text)
    cfg["model"]["weights"] = str(weights)
    cfg["model"]["initial_checkpoint"] = str(initial)
    cfg["model"]["initial_checkpoint_sha256"] = sha(initial)
    if args.check_only:
        print("PASS: fixed split IDs, local paths and initial checkpoint SHA verified; no files written.")
        return
    if (ROOT / "training/status.json").exists():
        parser.error("An experiment already exists here; do not reconfigure an active/resumed run")
    output = ROOT / "config.json"
    encoded = json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
    if output.exists() and output.read_text() != encoded:
        parser.error("config.json already exists with different contents; use a separate checkout")
    output.write_text(encoded)
    print("Configured:", output)


if __name__ == "__main__":
    main()
