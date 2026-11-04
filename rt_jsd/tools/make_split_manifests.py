"""Materialize the committed M3FD split IDs using local images and YOLO labels."""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def index(folder, suffixes):
    result = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            if path.stem in result:
                raise ValueError("Duplicate sample ID in " + str(folder))
            result[path.stem] = path.resolve()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="M3FD directory containing vi/, ir/ and labels/")
    parser.add_argument("--output", type=Path, default=ROOT / "data/m3fd_uod_811_seed2026")
    args = parser.parse_args()
    source, output = args.data_root.expanduser().resolve(), args.output.expanduser().resolve()
    if output.name != "m3fd_uod_811_seed2026":
        parser.error("Output directory must be named m3fd_uod_811_seed2026")
    images = {".png", ".jpg", ".jpeg", ".bmp"}
    pools = {"rgb": index(source / "vi", images),
             "thermal": index(source / "ir", images),
             "label": index(source / "labels", {".txt"})}
    payloads, seen = {}, set()
    for split, count in (("train", 3360), ("val", 420), ("test", 420)):
        ids = (ROOT / "splits/m3fd_811_seed2026" / (split + ".txt")).read_text().splitlines()
        if len(ids) != count or len(set(ids)) != count or seen.intersection(ids):
            raise ValueError("Committed split IDs are invalid: " + split)
        seen.update(ids)
        rows = [{"id": stem, **{key: str(pool[stem]) for key, pool in pools.items()}} for stem in ids]
        payloads[split] = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    # Check every destination before writing any manifest.
    for split, text in payloads.items():
        path = output / (split + ".jsonl")
        if path.exists() and path.read_text() != text:
            raise ValueError("Refusing to replace a different manifest: " + str(path))
    output.mkdir(parents=True, exist_ok=True)
    for split, text in payloads.items():
        (output / (split + ".jsonl")).write_text(text)
    print("Wrote fixed 3360/420/420 manifests:", output)


if __name__ == "__main__":
    main()
