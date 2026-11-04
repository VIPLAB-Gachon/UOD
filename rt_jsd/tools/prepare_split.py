#!/usr/bin/env python3
"""Build a separate, fixed M3FD 80/10/10 split without altering its source.

Seed-2026 duplicate-group shuffle matches the existing preparation algorithm;
only the target counts change to 3360/420/420.  This is an internal split,
not a claim to reproduce the exact UOD paper split IDs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


CLASSES = ["People", "Car", "Bus", "Lamp", "Motorcycle", "Truck"]
TARGETS = {"train": 3360, "val": 420, "test": 420}
PROJECT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_fixed(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace a different fixed artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def write_json(path: Path, payload: dict) -> None:
    write_fixed(path, (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT / "data/m3fd_uod")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/m3fd_uod_811_seed2026")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate directory trees")
    if args.seed != 2026:
        raise ValueError("This frozen protocol is specifically seed 2026")

    source_paths = sorted(path for path in source.rglob("*") if path.is_file())
    source_snapshot = {str(path.relative_to(source)): {
        "sha256": sha256(path), "bytes": path.stat().st_size,
    } for path in source_paths}
    records = read_jsonl(source / "all.jsonl")
    details = read_jsonl(source / "pair_audit.jsonl")
    source_report = json.loads((source / "preparation_report.json").read_text())
    by_id = {row["id"]: row for row in records}
    detail_by_id = {row["id"]: row for row in details}
    ids = sorted(by_id)
    if len(records) != 4200 or len(by_id) != 4200 or len(details) != 4200 or set(detail_by_id) != set(ids):
        raise ValueError("Require exactly 4200 distinct audited paired IDs")
    if source_report["class_names"] != CLASSES:
        raise ValueError("Class ordering differs from the frozen dataset")

    old_sets = {split: {row["id"] for row in read_jsonl(source / f"{split}.jsonl")}
                for split in TARGETS}
    if set.union(*old_sets.values()) != set(ids) or sum(map(len, old_sets.values())) != 4200:
        raise ValueError("Source partitions do not form a disjoint full-pool cover")

    parent = {stem: stem for stem in ids}
    seen = {}
    checks = Counter()

    def find(stem: str) -> str:
        while parent[stem] != stem:
            parent[stem] = parent[parent[stem]]
            stem = parent[stem]
        return stem

    for number, stem in enumerate(ids, 1):
        row, detail = by_id[stem], detail_by_id[stem]
        for modality in ("rgb", "thermal"):
            path = Path(row[modality])
            if not path.is_file() or sha256(path) != detail[f"{modality}_sha256"]:
                raise ValueError(f"Image missing or changed since full decode/pixel audit: {path}")
            checks[f"{modality}_file_sha256_matches_full_decode_audit"] += 1
            key = (modality, detail[f"{modality}_pixels_sha256"])
            if key in seen:
                parent[find(stem)] = find(seen[key])
            else:
                seen[key] = stem
        label_path = Path(row["label"])
        if not label_path.is_file() or sha256(label_path) != detail["label_sha256"]:
            raise ValueError(f"Label missing or changed: {label_path}")
        checks["label_sha256_matches_full_annotation_audit"] += 1
        label_rows = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
        observed = Counter()
        for values in label_rows:
            if len(values) != 5:
                raise ValueError(f"Invalid YOLO label row: {stem}")
            cls, cx, cy, width, height = map(float, values)
            if not all(math.isfinite(v) for v in (cls, cx, cy, width, height)):
                raise ValueError(f"Nonfinite label value: {stem}")
            if cls != int(cls) or not 0 <= cls < 6 or width <= 0 or height <= 0:
                raise ValueError(f"Invalid class or box size: {stem}")
            if min(cx - width / 2, cy - height / 2) < -1e-4 or max(cx + width / 2, cy + height / 2) > 1.0001:
                raise ValueError(f"Box outside normalized image bounds: {stem}")
            observed[CLASSES[int(cls)]] += 1
        if len(label_rows) != detail["objects"] or dict(observed) != detail["class_objects"]:
            raise ValueError(f"Label count differs from frozen audit: {stem}")
        checks["label_class_coordinates_and_counts_valid"] += 1
        if number % 500 == 0 or number == len(ids):
            print(f"Verified images, labels, and cached full-decode audit {number}/{len(ids)}", flush=True)

    grouped = defaultdict(list)
    for stem in ids:
        grouped[find(stem)].append(stem)
    groups = sorted(grouped.values(), key=lambda group: group[0])
    random.Random(args.seed).shuffle(groups)
    selected = {split: [] for split in TARGETS}
    for group in groups:
        remaining = {split: TARGETS[split] - len(selected[split]) for split in selected}
        fits = [split for split in selected if remaining[split] >= len(group)]
        split = fits[0] if fits else max(remaining, key=remaining.get)
        selected[split].extend(group)
    selected = {split: sorted(stems) for split, stems in selected.items()}
    sets = {split: set(stems) for split, stems in selected.items()}
    if any(len(selected[split]) != TARGETS[split] for split in selected):
        raise ValueError("Target partition cardinality not reached")
    if set.union(*sets.values()) != set(ids) or sum(map(len, sets.values())) != 4200:
        raise ValueError("New split contains overlap, unknown IDs, or missing IDs")
    membership = {stem: split for split, stems in sets.items() for stem in stems}
    for group in groups:
        if len({membership[stem] for stem in group}) != 1:
            raise ValueError(f"Exact duplicates cross split: {group}")

    output.mkdir(parents=True, exist_ok=True)
    split_reports = {}
    for split, stems in {"all": ids, **selected}.items():
        manifest = output / f"{split}.jsonl"
        write_fixed(manifest, "".join(json.dumps(by_id[stem]) + "\n" for stem in stems).encode())
        txt = ("\n".join(stems) + "\n").encode()
        write_fixed(output / "splits" / f"{split}.txt", txt)
        write_fixed(output / f"{split}.txt", txt)
        counts, class_images = Counter(), Counter()
        for stem in stems:
            counts.update(detail_by_id[stem]["class_objects"])
            class_images.update(detail_by_id[stem]["class_objects"].keys())
        split_reports[split] = {
            "pairs": len(stems), "objects": sum(counts.values()),
            "class_objects": {name: counts[name] for name in CLASSES},
            "class_present_images": {name: class_images[name] for name in CLASSES},
            "manifest": str(manifest), "manifest_sha256": sha256(manifest),
            "id_list": str(output / "splits" / f"{split}.txt"),
            "id_list_sha256": sha256(output / "splits" / f"{split}.txt"),
        }
    write_fixed(output / "classes.txt", ("\n".join(CLASSES) + "\n").encode())
    write_fixed(output / "pair_audit.jsonl", (source / "pair_audit.jsonl").read_bytes())
    dataset = {
        "name": "M3FD", "class_names": CLASSES, "num_classes": 6,
        **{split + "_manifest": report["manifest"] for split, report in split_reports.items()},
    }
    write_json(output / "dataset.json", dataset)
    old_relationship = {
        split: {old_split: len(stems & old_ids) for old_split, old_ids in old_sets.items()}
        for split, stems in sets.items()
    }
    source_unchanged = all(path.is_file() and sha256(path) == source_snapshot[str(path.relative_to(source))]["sha256"]
                           for path in source_paths)
    source_files_unchanged = sorted(path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()) == sorted(source_snapshot)
    if not source_unchanged or not source_files_unchanged:
        raise ValueError("Source dataset metadata changed during preparation")
    duplicate_groups = sorted([group for group in groups if len(group) > 1])
    report = {
        "status": "ready", "pairs": 4200, "class_names": CLASSES,
        "splits": split_reports, "split_seed": args.seed,
        "split_ratio": [8, 1, 1], "split_is_official": False,
        "split_policy": "internal 80/10/10; deterministic exact-duplicate-group shuffle; first fitting partition",
        "paper_exact_split_ids_available": False,
        "split_source": str(source), "split_disjoint": True,
        "all_4200_pairs_covered_once": True, "exact_duplicate_groups": duplicate_groups,
        "exact_duplicates_cross_split": False, "scene_sequence_separation_verified": False,
        "source_metadata_files_unchanged": True,
        "old_split_membership_in_new": old_relationship,
        "old_split_relation_note": "New train is old train+val; new val and test each contain 420 pairs from old test. Treat as a separate protocol and reinitialize the detector from common pretrained initialization.",
        "annotations_modified": False, "image_files_modified": False,
        "all_images_fully_decoded": source_report["all_images_fully_decoded"],
        "full_decode_audit_reused_after_encoded_sha256_identity_check": True,
        "all_thermal_grayscale_uint8": source_report["all_thermal_grayscale_uint8"],
        "all_yolo_voc_classes_and_coordinates_match": source_report["all_yolo_voc_classes_and_coordinates_match"],
        "observed_objects": split_reports["all"]["objects"],
        "class_id_voc_validation": source_report["class_id_voc_validation"],
        "source_metadata_snapshot": source_snapshot,
        "provenance": source_report["provenance"],
        "builder": str(Path(__file__).resolve()), "builder_sha256": sha256(Path(__file__)),
        "gpu_used": False, "training_started": False,
    }
    write_json(output / "preparation_report.json", report)
    audit = {
        "status": "passed", "checks": dict(checks),
        "all_source_metadata_sha256_unchanged": True,
        "partition_sizes_exact": True, "all_ids_covered_once": True,
        "no_cross_split_id_overlap": True, "no_cross_split_exact_duplicate_group": True,
        "all_splits_contain_all_six_classes": all(
            all(split_reports[split]["class_objects"][name] > 0 for name in CLASSES)
            for split in TARGETS),
        "source": str(source), "output": str(output),
        "split_seed": args.seed, "split_ratio": [8, 1, 1],
        "splits": split_reports, "old_split_membership_in_new": old_relationship,
        "duplicate_groups": duplicate_groups,
        "builder_sha256": report["builder_sha256"],
        "preparation_report_sha256": sha256(output / "preparation_report.json"),
        "dataset_json_sha256": sha256(output / "dataset.json"),
        "pair_audit_sha256": sha256(output / "pair_audit.jsonl"),
    }
    write_json(output / "split_audit.json", audit)
    print(json.dumps({"status": "passed", "output": str(output), "splits": split_reports,
                      "old_split_membership_in_new": old_relationship,
                      "split_audit_sha256": sha256(output / "split_audit.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
