"""Conservative recovery of visually degraded images; preserve prior frozen candidate."""

from collections import Counter
import json
from pathlib import Path
from PIL import Image
import numpy as np
from . import recover_baseline as recovery
from .audit_baseline_subset import audit_one
from .common import ROOT, REVISION, save_json


def gray_fraction(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((320, 180))
        a = np.asarray(im, dtype=np.int16)
    return float(
        ((a.max(2) - a.min(2) <= 2) & (a.mean(2) >= 120) & (a.mean(2) <= 135)).mean()
    )


def main():
    cfg, sources, datasets, metadata, anns = recovery.load_inputs()
    old_plan = recovery.PLAN
    plan = ROOT / "data/interim/subset_plans/baseline_seed42_recovery_v2"
    if plan.exists():
        raise ValueError("Plan exists")
    selection = json.loads((old_plan / "selection.json").read_text())
    previous = json.loads((old_plan / "recovery.json").read_text())
    rejected = {("train", 5235), ("val", 20260)}
    audited = json.loads(
        (ROOT / "reports/audit/baseline_subset_final_image_audit.json").read_text()
    )
    bykey = {(r["split"], r["image_id"]): r for r in audited["records"]}
    usedids = {r["image_id"] for r in selection} | {21818}
    usednames = {Path(r["source"]).name for r in selection} | {"803489.png"}
    hashes = {r["sha256"] for r in audited["records"]}
    inventory = json.loads((ROOT / "data/interim/hub_metadata.json").read_text())
    assert inventory["sha"] == REVISION
    files = {
        Path(r["rfilename"]).name: r["rfilename"]
        for r in inventory["siblings"]
        if r["rfilename"].endswith(".png")
    }
    final = []
    decisions = []
    rejected_candidates = []
    for row in selection:
        if (row["split"], row["image_id"]) not in rejected:
            final.append(row)
            continue
        s = row["split"]
        target = Counter(a["category_id"] for a in anns[s][row["image_id"]])
        full = Counter(a["category_id"] for a in datasets[s]["annotations"])
        candidates = [
            dict(split=s, image_id=im["id"], source=files[im["file_name"]])
            for im in datasets[s]["images"]
            if im["id"] not in usedids and im["file_name"] not in usednames
        ]
        candidates.sort(
            key=lambda r: recovery.replacement_rank(r, target, anns[s], full)
        )
        for rep in candidates:
            if not (cfg["raw"] / rep["source"]).is_file():
                from huggingface_hub import hf_hub_download

                print("Acquire replacement candidate", rep["source"], flush=True)
                hf_hub_download(
                    "iisc-aim/UVH-26",
                    rep["source"],
                    repo_type="dataset",
                    revision=REVISION,
                    local_dir=cfg["raw"],
                )
            check = audit_one(
                rep,
                cfg["raw"],
                metadata[s],
                anns[s],
                {c["id"] for c in datasets[s]["categories"]},
            )
            gray = (
                gray_fraction(cfg["raw"] / rep["source"])
                if not check["errors"]
                else None
            )
            if check["errors"] or check.get("sha256") in hashes or gray > 0.25:
                rejected_candidates.append(
                    dict(**rep, errors=check["errors"], gray_fraction=gray)
                )
                continue
            final.append(rep)
            usedids.add(rep["image_id"])
            usednames.add(Path(rep["source"]).name)
            hashes.add(check["sha256"])
            decisions.append(
                dict(
                    quarantined=row,
                    replacement=rep,
                    reason=["severe_visual_gray_artifacts"],
                    ranking_distance=recovery.replacement_rank(
                        rep, target, anns[s], full
                    )[0],
                    original_class_counts=dict(target),
                    replacement_class_counts=dict(
                        Counter(a["category_id"] for a in anns[s][rep["image_id"]])
                    ),
                    replacement_gray_fraction=gray,
                )
            )
            break
        else:
            raise ValueError("No replacement")
    save_json(plan / "selection.json", final)
    save_json(
        plan / "recovery.json",
        dict(
            **{
                k: v
                for k, v in previous.items()
                if k
                not in [
                    "quarantined",
                    "replacements",
                    "rejected_replacement_candidates",
                ]
            },
            supersedes="baseline_seed42",
            quarantined=previous["quarantined"]
            + [
                dict(
                    **bykey[k],
                    visual_quarantine_reason="severe gray artifacts observed during manual review",
                )
                for k in sorted(rejected)
            ],
            replacements=previous["replacements"] + decisions,
            rejected_replacement_candidates=previous["rejected_replacement_candidates"]
            + rejected_candidates,
            visual_policy="Reject these two visually observed severely degraded images. For replacement candidates reject low-resolution near-mid-gray fraction above 25%; this heuristic is a candidate filter, not a validated full-dataset corruption detector. Final replacements require manual review.",
        ),
    )
    save_json(
        ROOT / "reports/audit/baseline_subset_recovery_v2.json",
        json.loads((plan / "recovery.json").read_text()),
    )
    p = ROOT / "reports/audit/visual_review.json"
    v = json.loads(p.read_text())
    v.update(
        status="failed",
        decision="Superseded before training: severe source visual artifacts in train5235 and val20260 require conservative replacement.",
    )
    save_json(ROOT / "reports/audit/visual_review_superseded_v1.json", v)
    save_json(p, v)
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
