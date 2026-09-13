"""Complete selected-image audit, without changing raw data or candidate selection."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
from PIL import Image
from .common import (
    REVISION,
    annotation_paths,
    load_coco,
    paths,
    sha256,
    save_json,
    unique_index,
    convert_box,
)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit_one(row, raw, metadata, annotations, categories):
    result = dict(
        row,
        errors=[],
        invalid_metadata_boxes=[],
        invalid_actual_boxes=[],
        unknown_categories=[],
    )
    path = raw / row["source"]
    meta = metadata.get(row["image_id"])
    if meta is None:
        result["errors"].append("missing_annotation_entry")
        return result
    result.update(annotation_width=meta["width"], annotation_height=meta["height"])
    try:
        result["bytes"] = path.stat().st_size
        if not result["bytes"]:
            raise ValueError("empty_file")
        result["sha256"] = sha256(path)
        with Image.open(path) as im:
            im.load()
            result.update(
                width=im.width, height=im.height, mode=im.mode, format=im.format
            )
            if im.format != "PNG":
                result["errors"].append("not_png")
            if im.mode not in ("RGB", "RGBA", "L", "P"):
                result["errors"].append("unsupported_channels")
        if (result["width"], result["height"]) != (meta["width"], meta["height"]):
            result["errors"].append("dimension_mismatch")
    except (OSError, ValueError) as exc:
        result["errors"].append("decode_or_file_failure:" + type(exc).__name__)
    anns = annotations.get(row["image_id"], [])
    result["objects"] = len(anns)
    for ann in anns:
        if ann["category_id"] not in categories:
            result["unknown_categories"].append(ann["id"])
        for target, width, height in [
            ("metadata", meta["width"], meta["height"]),
            ("actual", result.get("width"), result.get("height")),
        ]:
            if width is None:
                continue
            try:
                convert_box(ann["bbox"], width, height)
            except ValueError as exc:
                result[f"invalid_{target}_boxes"].append(
                    {"annotation_id": ann["id"], "reason": str(exc)}
                )
    for field in [
        "invalid_metadata_boxes",
        "invalid_actual_boxes",
        "unknown_categories",
    ]:
        if result[field]:
            result["errors"].append(field)
    return result


def leakage(rows):
    output = {}
    for field in ["image_id", "filename", "sha256"]:
        groups = defaultdict(set)
        for row in rows:
            value = Path(row["source"]).name if field == "filename" else row.get(field)
            if value is not None:
                groups[value].add(row["split"])
        output["shared_" + field] = sorted(
            k for k, splits in groups.items() if len(splits) > 1
        )
    output["passed"] = not any(output.values())
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        default="data/interim/subset_plans/uvh26_mv_baseline_subset_v1/selection.json",
    )
    parser.add_argument("--prefix", default="baseline_subset")
    args = parser.parse_args()
    cfg = paths()
    source = annotation_paths(cfg["raw"])
    datasets = {s: load_coco(p) for s, p in source.items()}
    metadata = {s: unique_index(d["images"]) for s, d in datasets.items()}
    annotations = {}
    for s, d in datasets.items():
        annotations[s] = defaultdict(list)
        for ann in d["annotations"]:
            annotations[s][ann["image_id"]].append(ann)
    selection = json.loads(Path(args.selection).read_text())

    def check(row):
        return audit_one(
            row,
            cfg["raw"],
            metadata[row["split"]],
            annotations[row["split"]],
            {c["id"] for c in datasets[row["split"]]["categories"]},
        )

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(check, selection), 1):
            results.append(result)
            if i % 500 == 0:
                print(
                    f"Audited {i}/{len(selection)}; failed images {sum(bool(r['errors']) for r in results)}",
                    flush=True,
                )
    out = cfg["reports"] / "audit"
    split_check = leakage(results)
    summary = dict(
        scope="Selected candidate subset pixel audit only",
        revision=REVISION,
        selection_sha256=sha256(args.selection),
        annotation_sha256={s: sha256(p) for s, p in source.items()},
        counts=dict(Counter(r["split"] for r in results)),
        images=len(results),
        failed_images=sum(bool(r["errors"]) for r in results),
        error_counts=dict(Counter(e for r in results for e in r["errors"])),
        passed=not any(r["errors"] for r in results) and split_check["passed"],
        records=results,
    )
    save_json(out / f"{args.prefix}_image_audit.json", summary)
    save_json(out / f"{args.prefix}_leakage.json", split_check)
    fields = [
        "split",
        "image_id",
        "source",
        "annotation_width",
        "annotation_height",
        "width",
        "height",
        "mode",
        "bytes",
        "sha256",
        "errors",
    ]
    write_csv(
        out / f"{args.prefix}_dimension_mismatches.csv",
        [r for r in results if "dimension_mismatch" in r["errors"]],
        fields,
    )
    write_csv(
        out / f"{args.prefix}_decode_failures.csv",
        [
            r
            for r in results
            if any(e.startswith("decode_or_file_failure") for e in r["errors"])
        ],
        fields,
    )
    write_csv(
        out / f"{args.prefix}_content_hashes.csv",
        results,
        ["split", "image_id", "source", "sha256"],
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
