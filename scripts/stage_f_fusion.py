"""Stage F1-F5: guarded cached-calibration fusion cross-validation only."""

import csv
import importlib.metadata
import io
import json
import subprocess
import sys
import time
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.common_metrics import measure
from src.evaluation.comparison_protocol import (
    CALIBRATION,
    CALIBRATION_SHA,
    MAPPING_SHA,
    NAMES,
    canonical_hash,
    guard_path,
    load_manifest,
    seal,
    sha,
    validate_bundle,
    verify_seal,
)
from src.evaluation.complementarity import threshold_curve
from src.evaluation.fusion import (
    OPERATING_GRID,
    candidates,
    fuse,
    improvement_gate,
    make_folds,
    split_fold,
)
from src.experiments.accuracy_data import convert_labels

OUT = ROOT / "reports/accuracy_stage_f"
BUNDLE = ROOT / "runs/E4_stageF_v1"
E = ROOT / "reports/comparisons/E1_E3_stageE_v2"
DATA = ROOT / "data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2"
FAMILIES = ["baseline", "union", "wbf", "recovery"]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def note(stage, work, commands, inputs, measurements, remaining, deviations="None"):
    path = ROOT / f"docs/progress/2026-09-23_STAGE_{stage}.md"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        f"# Stage {stage} progress — {datetime.now(UTC).isoformat()}\n\nWork completed: {work}\n\nExact commands: `{commands}`\n\nInputs and hashes: {inputs}\n\nMeasurements: {measurements}\n\nFailures or deviations: {deviations}\n\nRemaining work: {remaining}\n\nGit: base commit `{subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}`; delivery commit is the commit introducing this note.\n"
    )


def load_inputs():
    receipt = json.loads((E / "freeze_receipt.json").read_text())
    if sha(ROOT / receipt["protocol_file"]) != receipt["protocol_file_sha256"]:
        raise ValueError("Parent protocol file hash mismatch")
    parent = verify_seal(json.loads((ROOT / receipt["protocol_file"]).read_text()))
    if canonical_hash(parent) != receipt["protocol_canonical_sha256"]:
        raise ValueError("Parent protocol canonical hash mismatch")
    for file, expected in receipt["artifact_sha256"].items():
        if sha(ROOT / file) != expected:
            raise ValueError(f"Stage E artifact hash mismatch: {file}")
    for file, expected in parent["source_sha256"].items():
        if sha(ROOT / file) != expected:
            raise ValueError(f"Frozen evaluator/source mismatch: {file}")
    if sha(ROOT / "configs/class_mapping.yaml") != MAPPING_SHA:
        raise ValueError("Mapping checksum mismatch")
    if (
        parent["AP"]["maxDets"] != [1, 10, 300]
        or parent["AP"]["score_floor"] != 0.001
        or parent["class_names"] != NAMES
    ):
        raise ValueError("Unexpected common evaluator settings")
    if not np.allclose(parent["AP"]["iou_thresholds"], np.arange(0.5, 0.96, 0.05)):
        raise ValueError("Unexpected AP IoU range")
    rows = load_manifest()
    ids = [r["image_id"] for r in rows]
    if len(ids) != 500 or sha(CALIBRATION) != CALIBRATION_SHA:
        raise ValueError("Calibration mismatch")
    records = {}
    hashes = {
        "parent_protocol": receipt["protocol_canonical_sha256"],
        "calibration": CALIBRATION_SHA,
        "mapping": MAPPING_SHA,
    }
    for model in ["E1", "E3"]:
        config = parent["models"][model]
        checkpoint = ROOT / config["checkpoint"]
        if sha(checkpoint) != config["sha256"]:
            raise ValueError(f"{model} checkpoint mismatch")
        r = json.loads((E / f"{model}_prediction_receipt.json").read_text())
        path = ROOT / "runs" / r["id"] / "predictions.jsonl"
        if (
            r["manifest_sha256"] != CALIBRATION_SHA
            or r["mapping_sha256"] != MAPPING_SHA
        ):
            raise ValueError("Prediction provenance mismatch")
        if r["checkpoint_sha256"] != config["sha256"]:
            raise ValueError("Prediction checkpoint mismatch")
        validate_bundle(path, r, ids)
        hashes[model + "_checkpoint"] = config["sha256"]
        hashes[model + "_predictions"] = r["prediction_sha256"]
        stored = [json.loads(line) for line in path.open()]
        converted = []
        for row, pred in zip(rows, stored):
            if pred["image_key"] != row["image"]:
                raise ValueError("Stable image key mismatch")
            label = guard_path(DATA / row["label"], DATA, rows)
            if sha(label) != row["label_sha256"]:
                raise ValueError("Calibration annotation hash mismatch")
            gt, labels = convert_labels(
                label.read_text(), pred["width"], pred["height"]
            )
            converted.append(
                {
                    "image_id": row["image_id"],
                    "width": pred["width"],
                    "height": pred["height"],
                    "gt_boxes": gt.tolist(),
                    "gt_classes": (labels - 1).tolist(),
                    "boxes": [d["xyxy"] for d in pred["detections"]],
                    "scores": [d["confidence"] for d in pred["detections"]],
                    "classes": [d["class_id"] for d in pred["detections"]],
                }
            )
        records[model] = converted
    for a, b in zip(records["E1"], records["E3"]):
        if a["gt_boxes"] != b["gt_boxes"] or a["gt_classes"] != b["gt_classes"]:
            raise ValueError("E1/E3 annotation mismatch")
    # Explicit historical report/source allowlist; never enumerate data or reserved manifests.
    preservation = {}
    for base in [
        "reports/evaluations",
        "reports/comparisons/E1_E3_stageE_v2",
        "reports/comparisons/E1_vs_E0",
        "reports/comparisons/E2_gateA_v2",
    ]:
        for p in sorted((ROOT / base).rglob("*")):
            if p.is_file():
                preservation[str(p.relative_to(ROOT))] = sha(p)
    save(
        OUT / "integrity.json",
        {
            "passed": True,
            "hashes": hashes,
            "images": 500,
            "objects": sum(len(r["gt_classes"]) for r in records["E1"]),
            "preservation": preservation,
            "reserved_accessed": False,
        },
    )
    return rows, records, parent, hashes


def protocol(rows, parent, hashes):
    files = [
        "src/evaluation/fusion.py",
        "scripts/stage_f_fusion.py",
        "src/evaluation/common_metrics.py",
        "src/evaluation/complementarity.py",
        "src/evaluation/comparison_protocol.py",
    ]
    folds = make_folds([r["image_id"] for r in rows])
    save(
        OUT / "folds.json",
        {
            "seed": 42,
            "folds": folds,
            "allocation": "sorted image IDs; random.Random(42).shuffle then stride-five slices; heldout100/train400",
        },
    )
    value = {
        "id": "E4_stageF_calibration_CV_v1",
        "parent_protocol_hash": hashes["parent_protocol"],
        "input_hashes": hashes,
        "seed": 42,
        "scope": "cached calibration500 only; no inference except final latency",
        "candidate_grid": candidates(),
        "operating_grid": OPERATING_GRID,
        "algorithms": {
            "baseline": "Unchanged E1 low-floor detections; operating threshold selected on training fold from same grid as fusion. Stage E .34 is not used to choose OOF threshold.",
            "union": "Concatenate E1 then E3 scores unchanged; stable class-aware NMS, suppress IoU>grid threshold, top300. No score calibration.",
            "wbf": "E1-anchored asymmetric two-model WBF. Process E1 descending score, pair at most one unused same-class E3 with highest IoU>=threshold. Box=(w1*s1*b1+w3*s3*b3)/(w1*s1+w3*s3); score=w1*s1+w3*s3. Unmatched E1 unchanged; unmatched E3 not added.",
            "recovery": "Preserve all E1 values; add high-confidence E3 only when max ANY-class IoU against E1 anchors with score>=.20 and previously accepted additions is <grid IoU. Never evict E1 at max300. Training-fold GT support<100: E1-only. Among eligible classes, E3 training AP50:95<E1 class AP: require max(grid addition threshold,.55); otherwise grid addition threshold. Class rules fitted only on training folds.",
        },
        "selection_rule": "For each family/train fold maximize macro class F1 over parameter x operating grid; round12 tie then prefer higher operating threshold, larger E1 weight, higher addition threshold, lower IoU, lexical candidate ID. No AP-based parameter search beyond fixed weak-class rule.",
        "cross_validation": "Five deterministic image folds. Training labels only select params/threshold/support/weak-class rules; selected model evaluated once per held-out fold. Pooled low-floor OOF predictions determine AP; fold-specific threshold-filtered predictions separately determine fixed P/R/F1/confusion. Full OOF pool used to compare families, so this is model-selection evidence, not an unbiased final test.",
        "gate": {
            "AP50_95_delta_min": -0.005,
            "precision_delta_min": -0.01,
            "recall_delta_min": -0.01,
            "gain": "macro class F1 >=.02 OR AP50:95 >=.01",
            "distributed": "For the metric meeting gain gate, at least two classes with >=100 GT improve by >=.01; mean per-class gain excluding largest-GT class is positive.",
            "duplicates": "Increase in duplicate count <=max(10,25% baseline duplicate count) AND duplicate/prediction-rate increase <=.02.",
            "reproduction": "Bitwise-equivalent canonical prediction hashes from repeated deterministic fusion with same configuration.",
            "target": "Report P>=.70 and R>=.70; never tune to force target.",
        },
        "family_selection": "Among passing families maximize pooled OOF macro F1, then AP50:95, then fixed family order. If none pass reject E4; strongest nonbaseline OOF macro-F1 family is benchmarked only.",
        "final_fit": "If pass, tune winning family on all calibration500 using exactly same grid/rule, freeze final code/config and evaluate once. If reject, no extra search/full-set score: benchmark fold0 training-selected configuration of strongest OOF family, explicitly unselected.",
        "AP": parent["AP"],
        "matching_iou": 0.5,
        "PR_curves": "COCO low-floor interpolated PR at IoU .50, distinct from fold-specific operating points.",
        "latency": "Existing Stage D seed42 calibration100 sample; MPS float32 batch1,10warmups,one timed pass; actual E1 then E3 inference then fusion. Include file decode, both detector preprocess/inference/NMS, coordinate conversion, CPU transfer and fusion/filter; report load excluded. No video FPS.",
        "forbidden": [
            "reports/accuracy_stage_a/protocol_v2/final_evaluation_1500.json",
            "data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2/val_manifest.json",
            "Any image/label path not in hash-verified calibration500 allowlist; do not enumerate or hash reserved paths",
        ],
        "source_sha256": {f: sha(ROOT / f) for f in files},
        "packages": {
            n: importlib.metadata.version(n)
            for n in ["torch", "torchvision", "ultralytics", "numpy", "pycocotools"]
        },
        "reserved_accessed": False,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    save(OUT / "protocol.json", seal(value))
    return folds, value


def evaluate(records, threshold):
    with redirect_stdout(io.StringIO()):
        return measure(records, NAMES, confidence=threshold)


def errors_summary(errors):
    return {
        "tp": sum(e["correct"] for e in errors),
        "fp": sum(e["class_aware_fp"] for e in errors),
        "fn": sum(e["class_aware_fn"] for e in errors),
        "duplicates": sum(len(e["duplicate_predictions"]) for e in errors),
        "localization": sum(len(e["localization_candidates"]) for e in errors),
        "background": sum(len(e["background_candidates"]) for e in errors),
    }


def training_context(a, b):
    _, ac, *_ = evaluate(a, 0.25)
    _, bc, *_ = evaluate(b, 0.25)
    eligible = [i for i, c in enumerate(ac) if c["gt_count"] >= 100]
    return {
        "eligible_classes": eligible,
        "weak_classes": [
            i for i in eligible if (bc[i]["ap50_95"] or 0) < (ac[i]["ap50_95"] or 0)
        ],
        "fit_image_ids": [r["image_id"] for r in a],
    }


def rank(point, cfg):
    return (
        round(point["macro_class_f1"], 12),
        point["threshold"],
        cfg.get("weights", [1])[0],
        cfg.get("addition_threshold", 0),
        -cfg.get("iou", 0),
        cfg["id"],
    )


def choose(family, training_ids, all_records, context, cache, trials, fold):
    best = None
    ids = set(training_ids)
    for cfg in [c for c in candidates() if c["algorithm"] == family]:
        key = canonical_hash(
            {"cfg": cfg, "context": context if family == "recovery" else None}
        )
        if key not in cache:
            cache[key] = [
                fuse(a, b, cfg, context)
                for a, b in zip(all_records["E1"], all_records["E3"])
            ]
        curve = threshold_curve(
            [r for r in cache[key] if r["image_id"] in ids], OPERATING_GRID
        )
        for point in curve:
            trials.append({"fold": fold, "candidate": cfg["id"], **point})
            value = (rank(point, cfg), cfg, point, key)
            if best is None or value[0] > best[0]:
                best = value
    return {
        "configuration": best[1],
        "threshold": best[2]["threshold"],
        "training_macro_f1": best[2]["macro_class_f1"],
        "cache_key": best[3],
        "context": context if family == "recovery" else None,
    }


def mixed_evaluation(records, thresholds):
    raw_summary, raw_classes, _, pr, _ = evaluate(records, 0.001)
    fixed = []
    for r in records:
        keep = [i for i, s in enumerate(r["scores"]) if s >= thresholds[r["image_id"]]]
        fixed.append(
            {
                **r,
                **{k: [r[k][i] for i in keep] for k in ["boxes", "scores", "classes"]},
            }
        )
    summary, classes, matrix, _, errors = evaluate(fixed, 0.001)
    summary.update(
        map50=raw_summary["map50"],
        map50_95=raw_summary["map50_95"],
        predictions_raw=raw_summary["predictions_raw"],
        confidence="training-selected threshold per held-out fold; see fold selections",
        scope="Calibration500 pooled out-of-fold model-selection evidence",
    )
    for c, raw in zip(classes, raw_classes):
        for key in ["ap50", "ap50_95", "prediction_count_raw"]:
            c[key] = raw[key]
    return summary, classes, matrix, pr, errors


def export(name, records, result):
    summary, classes, matrix, pr, errors = result
    save(OUT / f"{name}_metrics.json", summary)
    table(OUT / f"{name}_per_class.csv", classes)
    save(
        OUT / f"{name}_confusion.json",
        {"labels": NAMES + ["background"], "matrix": matrix.tolist()},
    )
    save(OUT / f"{name}_errors.json", errors_summary(errors))
    np.savez_compressed(
        OUT / f"{name}_pr_curves.npz", precision=pr, recall=np.linspace(0, 1, 101)
    )
    table(
        OUT / f"{name}_image_counts.csv",
        [
            {
                "image_id": r["image_id"],
                "raw_detections": len(r["scores"]),
                "operating_detections": e["predictions_fixed"],
                "tp": e["correct"],
                "fp": e["class_aware_fp"],
                "fn": e["class_aware_fn"],
                "duplicates": len(e["duplicate_predictions"]),
                "localization": len(e["localization_candidates"]),
            }
            for r, e in zip(records, errors)
        ],
    )
    with (BUNDLE / f"{name}_predictions.jsonl").open("x") as stream:
        for r in records:
            stream.write(json.dumps(r, separators=(",", ":")) + "\n")


def main():
    if OUT.exists() or BUNDLE.exists():
        raise FileExistsError("Immutable Stage F output already exists")
    OUT.mkdir(parents=True)
    BUNDLE.mkdir()
    try:
        start = time.perf_counter()
        rows, records, parent, hashes = load_inputs()
        folds, plan = protocol(rows, parent, hashes)
        note(
            "F1",
            "Verified all checkpoints, parent, calibration, mapping and cached prediction hashes; frozen bounded protocol before scoring.",
            " .venv/bin/python scripts/stage_f_fusion.py",
            json.dumps(hashes),
            "500 allowed images;19 parameter configurations;9 operating thresholds;five folds",
            "Cached CV, benchmark and closeout",
        )
        cache = {}
        trials = []
        selections = []
        oof = {f: [] for f in FAMILIES}
        thresholds = {f: {} for f in FAMILIES}
        fold_metrics = {f: [] for f in FAMILIES}
        repro = []
        for fold in range(5):
            a, held = split_fold(records["E1"], folds, fold)
            b, _ = split_fold(records["E3"], folds, fold)
            context = training_context(a, b)
            train_ids = [r["image_id"] for r in a]
            held_ids = {r["image_id"] for r in held}
            for family in FAMILIES:
                chosen = choose(
                    family, train_ids, records, context, cache, trials, fold
                )
                selected = [
                    r for r in cache[chosen["cache_key"]] if r["image_id"] in held_ids
                ]
                repeated = [
                    fuse(a, b, chosen["configuration"], chosen["context"])
                    for a, b in zip(records["E1"], records["E3"])
                    if a["image_id"] in held_ids
                ]
                equal = canonical_hash(selected) == canonical_hash(repeated)
                assert equal
                repro.append(
                    {
                        "fold": fold,
                        "family": family,
                        "prediction_hash": canonical_hash(selected),
                        "equal": equal,
                    }
                )
                metrics, *_ = evaluate(selected, chosen["threshold"])
                fold_metrics[family].append(metrics)
                oof[family].extend(selected)
                thresholds[family].update(
                    {r["image_id"]: chosen["threshold"] for r in selected}
                )
                selections.append(
                    {
                        "fold": fold,
                        "family": family,
                        **chosen,
                        "training_images": 400,
                        "heldout_images": 100,
                    }
                )
            save(
                OUT / "progress.json",
                {
                    "completed_folds": fold + 1,
                    "total_folds": 5,
                    "elapsed_seconds": time.perf_counter() - start,
                },
            )
        save(OUT / "fold_selections.json", selections)
        table(OUT / "training_search.csv", trials)
        save(OUT / "reproducibility.json", repro)
        results = {}
        per_class = []
        for family in FAMILIES:
            oof[family].sort(key=lambda r: r["image_id"])
            result = mixed_evaluation(oof[family], thresholds[family])
            results[family] = result
            export(family + "_oof", oof[family], result)
            per_class.extend([{"family": family, **c} for c in result[1]])
        table(OUT / "per_class_results.csv", per_class)
        base = results["baseline"]
        gates = {
            f: improvement_gate(
                base[0],
                results[f][0],
                base[1],
                results[f][1],
                errors_summary(base[4]),
                errors_summary(results[f][4]),
                all(r["equal"] for r in repro if r["family"] == f),
            )
            for f in FAMILIES[1:]
        }
        passing = [f for f in FAMILIES[1:] if gates[f]["passed"]]
        order = lambda f: (
            results[f][0]["macro_class_f1"],
            results[f][0]["map50_95"],
            -FAMILIES.index(f),
        )
        strongest = max(FAMILIES[1:], key=order)
        selected = max(passing, key=order) if passing else None
        aggregate = {
            f: {
                key: {
                    "mean": float(np.mean([x[key] for x in fold_metrics[f]])),
                    "sample_std": float(
                        np.std([x[key] for x in fold_metrics[f]], ddof=1)
                    ),
                }
                for key in [
                    "precision",
                    "recall",
                    "harmonic_aggregate_f1",
                    "macro_class_f1",
                    "map50",
                    "map50_95",
                ]
            }
            for f in FAMILIES
        }
        report = {
            "scope": "calibration-only five-fold model selection; preselected checkpoints and historical calibration exposure limit independence",
            "pooled_oof": {f: results[f][0] for f in FAMILIES},
            "fold_metrics": fold_metrics,
            "fold_mean_std": aggregate,
            "gates": gates,
            "selected_fusion": selected,
            "strongest_candidate": strongest,
            "decision": "select_E4" if selected else "reject_E4_retain_E1",
            "elapsed_seconds": time.perf_counter() - start,
            "reserved_accessed": False,
        }
        save(OUT / "cross_validation_results.json", report)
        note(
            "F2_F3",
            "Completed deterministic5fold training-only search and held-out evaluation for four families; cached fusion only.",
            " .venv/bin/python scripts/stage_f_fusion.py",
            f"Protocol {canonical_hash(plan)}; folds {sha(OUT / 'folds.json')}",
            f"{len(trials)} training-only grid rows;20 heldout evaluations;repeated prediction hashes match",
            "Gate/final configuration, latency and delivery",
        )
        if selected:
            context = training_context(records["E1"], records["E3"])
            fit_trials = []
            chosen = choose(
                selected,
                [r["image_id"] for r in rows],
                records,
                context,
                cache,
                fit_trials,
                "full_fit",
            )
            table(OUT / "final_fit_search.csv", fit_trials)
            final_config = {
                "purpose": "selected_E4_calibration_only",
                "family": selected,
                "configuration": chosen["configuration"],
                "threshold": chosen["threshold"],
                "context": chosen["context"],
            }
        else:
            chosen = next(
                s for s in selections if s["fold"] == 0 and s["family"] == strongest
            )
            final_config = {
                "purpose": "rejected_fusion_latency_only_not_selected",
                "family": strongest,
                "configuration": chosen["configuration"],
                "threshold": chosen["threshold"],
                "context": chosen["context"],
                "rule": "Fold0 training-selected representative; no additional all-calibration parameter search",
            }
        final_config.update(
            parent_protocol_hash=hashes["parent_protocol"],
            stage_f_protocol_hash=canonical_hash(plan),
            input_hashes=hashes,
            source_sha256=plan["source_sha256"],
            reserved_accessed=False,
        )
        config_path = ROOT / "configs/accuracy/E4_fusion.yaml"
        if config_path.exists():
            raise FileExistsError(config_path)
        config_path.write_text(yaml.safe_dump(seal(final_config), sort_keys=False))
        save(OUT / "final_amendment.json", seal(final_config))
        if selected:
            final = cache[chosen["cache_key"]]
            export("frozen_calibration", final, evaluate(final, chosen["threshold"]))
        note(
            "F4_F5",
            "Applied preregistered improvement gates; froze final/benchmark-only configuration without changing any search rule.",
            " .venv/bin/python scripts/stage_f_fusion.py",
            f"Final amendment {canonical_hash(final_config)}",
            json.dumps(
                {"decision": report["decision"], "strongest": strongest, "gates": gates}
            ),
            "Actual sequential-inference benchmark, tests, preservation and Git",
        )
        save(
            OUT / "CV_COMPLETE.json",
            {
                "passed": True,
                "decision": report["decision"],
                "reserved_accessed": False,
                "seconds": time.perf_counter() - start,
            },
        )
        print(
            json.dumps(
                {
                    "completed": True,
                    "decision": report["decision"],
                    "strongest": strongest,
                    "metrics": report["pooled_oof"],
                }
            )
        )
    except BaseException as error:
        save(
            OUT / "FAILURE.json",
            {
                "stage": "F1-F5",
                "error_type": type(error).__name__,
                "message": str(error),
                "utc": datetime.now(UTC).isoformat(),
                "reserved_accessed": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
