"""Quarantine invalid candidates and freeze an independently audited MV subset."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import yaml
from .audit_baseline_subset import audit_one, leakage, write_csv
from .common import (
    ROOT,
    REVISION,
    paths,
    annotation_paths,
    load_coco,
    unique_index,
    category_mapping,
    sha256,
    save_json,
    convert_box,
    validate_line,
    validate_manifest,
)

PLAN = ROOT / "data/interim/subset_plans/baseline_seed42_recovery_v1"
DEST = ROOT / "data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42"


def load_inputs():
    cfg = paths()
    sources = annotation_paths(cfg["raw"])
    datasets = {s: load_coco(p) for s, p in sources.items()}
    metadata = {s: unique_index(d["images"]) for s, d in datasets.items()}
    anns = {s: defaultdict(list) for s in datasets}
    for s, d in datasets.items():
        unique_index(d["annotations"])
        for a in d["annotations"]:
            anns[s][a["image_id"]].append(a)
    return cfg, sources, datasets, metadata, anns


def replacement_rank(row, target, annotations, full_counts):
    counts = Counter(a["category_id"] for a in annotations.get(row["image_id"], []))
    # Prefer exact class-exposure matches, then rare-class-weighted differences.
    distance = sum(
        abs(counts[c] - target[c]) / max(full_counts[c], 1)
        for c in set(counts) | set(target)
    )
    tie = hashlib.sha256(f"42:{row['split']}:{row['image_id']}".encode()).hexdigest()
    return distance, tie


def plan():
    if PLAN.exists():
        raise ValueError("Recovery plan exists; preserve it and use freeze after audit")
    cfg, sources, datasets, metadata, anns = load_inputs()
    audit = json.loads(
        (ROOT / "reports/audit/baseline_subset_image_audit.json").read_text()
    )
    candidate = (
        ROOT / "data/interim/subset_plans/uvh26_mv_baseline_subset_v1/selection.json"
    )
    if audit["selection_sha256"] != sha256(candidate):
        raise ValueError("Candidate changed after audit")
    selection = json.loads(candidate.read_text())
    bad = {(r["split"], r["image_id"]): r for r in audit["records"] if r["errors"]}
    if not leakage(audit["records"])["passed"]:
        raise ValueError("Content/ID/filename leakage needs separate investigation")
    used_ids = {r["image_id"] for r in selection}
    used_names = {Path(r["source"]).name for r in selection}
    hashes = {r["sha256"] for r in audit["records"] if r.get("sha256")}
    inventory = json.loads((ROOT / "data/interim/hub_metadata.json").read_text())
    if inventory["sha"] != REVISION:
        raise ValueError("Revision mismatch")
    files = unique_index(
        [
            {"name": Path(r["rfilename"]).name, "source": r["rfilename"]}
            for r in inventory["siblings"]
            if r["rfilename"].endswith(".png")
        ],
        "name",
    )
    replacements = []
    rejected_candidates = []
    final = []
    for row in selection:
        key = (row["split"], row["image_id"])
        if key not in bad:
            final.append(row)
            continue
        s = row["split"]
        target = Counter(a["category_id"] for a in anns[s][row["image_id"]])
        full = Counter(a["category_id"] for a in datasets[s]["annotations"])
        candidates = [
            dict(split=s, image_id=im["id"], source=files[im["file_name"]]["source"])
            for im in datasets[s]["images"]
            if im["id"] not in used_ids
            and im["file_name"] not in used_names
            and (cfg["raw"] / files[im["file_name"]]["source"]).is_file()
        ]
        candidates.sort(key=lambda r: replacement_rank(r, target, anns[s], full))
        for replacement in candidates:
            checked = audit_one(
                replacement,
                cfg["raw"],
                metadata[s],
                anns[s],
                {c["id"] for c in datasets[s]["categories"]},
            )
            if checked["errors"] or checked.get("sha256") in hashes:
                rejected_candidates.append(
                    dict(
                        **replacement,
                        errors=checked["errors"],
                        duplicate_content=checked.get("sha256") in hashes,
                    )
                )
                continue
            final.append(replacement)
            used_ids.add(replacement["image_id"])
            used_names.add(Path(replacement["source"]).name)
            hashes.add(checked["sha256"])
            replacements.append(
                dict(
                    quarantined=row,
                    replacement=replacement,
                    reason=bad[key]["errors"],
                    ranking_distance=replacement_rank(
                        replacement, target, anns[s], full
                    )[0],
                    original_class_counts=dict(target),
                    replacement_class_counts=dict(
                        Counter(
                            a["category_id"] for a in anns[s][replacement["image_id"]]
                        )
                    ),
                )
            )
            break
        else:
            raise ValueError("No audited local replacement available")
    validate_manifest(final)
    save_json(PLAN / "selection.json", final)
    save_json(
        PLAN / "recovery.json",
        dict(
            candidate_sha256=sha256(candidate),
            algorithm="Preserve valid candidate rows. Rank unselected locally present same-split candidates by inverse-full-class-frequency-weighted L1 exposure distance; SHA256(42:split:id) breaks ties. Audit before acceptance; forbid reused IDs, filenames and content.",
            seed=42,
            quarantined=list(bad.values()),
            replacements=replacements,
            rejected_replacement_candidates=rejected_candidates,
        ),
    )
    save_json(
        ROOT / "reports/audit/baseline_subset_recovery.json",
        json.loads((PLAN / "recovery.json").read_text()),
    )
    print(json.dumps(replacements, indent=2))


def freeze(audit_prefix="baseline_subset_final"):

    if DEST.exists() or DEST.with_name(DEST.name + ".building").exists():
        raise ValueError("Version exists; do not overwrite")
    cfg, sources, datasets, metadata, anns = load_inputs()
    audit_path = ROOT / f"reports/audit/{audit_prefix}_image_audit.json"
    audit = json.loads(audit_path.read_text())
    selected = json.loads((PLAN / "selection.json").read_text())
    if not audit["passed"] or audit["selection_sha256"] != sha256(
        PLAN / "selection.json"
    ):
        raise ValueError("Final audit did not pass this selection")
    if audit["annotation_sha256"] != {s: sha256(p) for s, p in sources.items()}:
        raise ValueError("Annotations changed after audit")
    checked = {(r["split"], r["image_id"]): r for r in audit["records"]}
    if Counter(r["split"] for r in selected) != Counter(train=8000, val=2000):
        raise ValueError("Wrong subset size")
    mapping = category_mapping(datasets["train"]["categories"])
    if mapping != category_mapping(datasets["val"]["categories"]):
        raise ValueError("Class mapping differs")
    ids = {c["original_id"]: c["yolo_id"] for c in mapping}
    temp = DEST.with_name(DEST.name + ".building")
    temp.mkdir(parents=True)
    manifest = []
    frequencies = Counter()
    backgrounds = []
    for row in selected:
        s = row["split"]
        meta = metadata[s][row["image_id"]]
        check = checked[(s, row["image_id"])]
        source = cfg["raw"] / row["source"]
        image = f"images/{s}/{source.name}"
        label = f"labels/{s}/{source.stem}.txt"
        if sha256(source) != check["sha256"]:
            raise ValueError("Image changed after audit")
        (temp / image).parent.mkdir(parents=True, exist_ok=True)
        (temp / label).parent.mkdir(parents=True, exist_ok=True)
        (temp / image).symlink_to(os.path.relpath(source, (temp / image).parent))
        lines = []
        for ann in sorted(anns[s][row["image_id"]], key=lambda a: a["id"]):
            cid = ids[ann["category_id"]]
            box = convert_box(ann["bbox"], meta["width"], meta["height"])
            line = f"{cid} " + " ".join(f"{v:.10f}" for v in box)
            validate_line(line, len(mapping))
            lines.append(line)
            frequencies[f"{s}:{cid}"] += 1
        (temp / label).write_text("\n".join(lines) + ("\n" if lines else ""))
        manifest.append(
            dict(
                **row,
                image=image,
                label=label,
                objects=len(lines),
                label_sha256=sha256(temp / label),
                source_sha256=check["sha256"],
            )
        )
        if not lines:
            backgrounds.append(row)
    validate_manifest(manifest)
    save_json(temp / "manifest.json", manifest)
    for s in ["train", "val"]:
        save_json(temp / f"{s}_manifest.json", [r for r in manifest if r["split"] == s])
    recovery = json.loads((PLAN / "recovery.json").read_text())
    distribution = []
    for s in ["train", "val"]:
        full = Counter(a["category_id"] for a in datasets[s]["annotations"])
        total = sum(frequencies[f"{s}:{c['yolo_id']}"] for c in mapping)
        for c in mapping:
            count = frequencies[f"{s}:{c['yolo_id']}"]
            full_share = full[c["original_id"]] / sum(full.values())
            sub_share = count / total
            distribution.append(
                dict(
                    split=s,
                    class_id=c["yolo_id"],
                    name=c["name"],
                    full_instances=full[c["original_id"]],
                    subset_instances=count,
                    full_share=full_share,
                    subset_share=sub_share,
                    share_difference_pp=100 * (sub_share - full_share),
                )
            )
    provenance = dict(
        version=str(DEST.relative_to(cfg["processed"])),
        scope="UVH-26 MV 8000/2000 subset; no full image-integrity claim",
        revision=REVISION,
        seed=42,
        selection_algorithm=recovery["algorithm"],
        candidate_selection_algorithm="Original coverage-aware seed42 selection retained except documented quarantines",
        annotation_sha256={s: sha256(p) for s, p in sources.items()},
        manifest_sha256={
            s: sha256(temp / f"{s}_manifest.json") for s in ["train", "val"]
        },
        combined_manifest_sha256=sha256(temp / "manifest.json"),
        image_audit_sha256=sha256(audit_path),
        image_audit_status="passed",
        class_mapping_sha256=sha256(ROOT / "configs/class_mapping.yaml"),
    )
    ledger = dict(
        version=provenance["version"],
        class_frequencies=dict(frequencies),
        accepted_boxes=sum(frequencies.values()),
        rejected_boxes=[],
        rejected=0,
        quarantined_images=recovery["quarantined"],
        replacement_images=recovery["replacements"],
        background_images=backgrounds,
        unknown_categories=[],
        clipped=0,
        repaired=0,
        images=len(manifest),
        manifest_sha256=provenance["combined_manifest_sha256"],
    )
    save_json(temp / "conversion.json", ledger)
    save_json(temp / "provenance.json", provenance)
    save_json(temp / "version.json", provenance)
    save_json(
        temp / "subset_summary.json",
        dict(
            train=8000,
            val=2000,
            objects=dict(frequencies),
            distribution=distribution,
            **{"image_audit_status": "passed"},
        ),
    )
    (temp / "dataset.yaml").write_text(
        yaml.safe_dump(
            dict(
                path=str(DEST),
                train="images/train",
                val="images/val",
                names={c["yolo_id"]: c["name"] for c in mapping},
            ),
            sort_keys=False,
        )
    )
    temp.rename(DEST)
    save_json(ROOT / "reports/audit/baseline_subset_conversion.json", ledger)
    save_json(ROOT / "reports/audit/baseline_subset_frozen_provenance.json", provenance)
    write_csv(
        ROOT / "reports/tables/baseline_subset_final_distribution.csv",
        distribution,
        list(distribution[0]),
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["plan", "freeze"])
    p.add_argument("--plan-dir")
    p.add_argument("--destination")
    p.add_argument("--audit-prefix", default="baseline_subset_final")
    a = p.parse_args()
    if a.plan_dir:
        PLAN = (ROOT / a.plan_dir).resolve()
    if a.destination:
        DEST = (ROOT / a.destination).resolve()
    plan() if a.action == "plan" else freeze(a.audit_prefix)
