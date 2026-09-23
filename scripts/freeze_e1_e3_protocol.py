"""Freeze the future comparison contract from completed calibration evidence."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.comparison_protocol import seal, sha, verify_seal

REPORT = ROOT / "reports/comparisons/E1_E3_stageE_v2"
OUTPUT = ROOT / "configs/accuracy/E1_E3_comparison_v1.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError("Frozen protocol already exists")
    plan = verify_seal(json.loads((REPORT / "calibration_plan.json").read_text()))
    thresholds = verify_seal(
        json.loads((REPORT / "operating_thresholds.json").read_text())
    )
    if not json.loads((REPORT / "COMPLETE.json").read_text())["passed"]:
        raise ValueError("Calibration incomplete")
    protocol = {
        "id": "E1_E3_E4_matched_comparison_v1",
        "status": "base_protocol_frozen_reserved_gate_closed_pending_E4_freeze_and_separate_authorization",
        "schema_version": plan["schema_version"],
        "class_names": plan["class_names"],
        "class_mapping_sha256": plan["mapping_sha256"],
        "reserved_manifest_sha256": plan["reserved_sha256_recorded_only"],
        "reserved_manifest_hash_source": "Previously recorded Stage A protocol; manifest contents not opened in Stage E",
        "reserved_scope": "1500 historically exposed validation images, not pristine independent test data",
        "calibration_manifest_sha256": plan["calibration_sha256"],
        "models": plan["models"],
        "common_evaluator_source_sha256": plan["source_sha256"][
            "src/evaluation/common_metrics.py"
        ],
        "source_sha256": plan["source_sha256"],
        "packages": plan["packages"],
        "AP": plan["AP"],
        "coordinate_rule": plan["coordinates"],
        "fixed_threshold_matching": plan["threshold_rule"],
        "operating_thresholds": thresholds["thresholds"],
        "threshold_selection_seal": json.loads(
            (REPORT / "operating_thresholds.json").read_text()
        )["canonical_sha256"],
        "statistical_analysis": {
            "unit": "image; paired sampling with replacement",
            "seed": 42,
            "replicates": 1000,
            "resample_size": 1500,
            "shared_draws": "identical sampled image indices for E1/E3/E4 in each replicate; duplicate images receive distinct COCO IDs with copied GT and predictions",
            "metrics": ["AP50", "AP50:95", "macro_class_F1", "harmonic_aggregate_F1"],
            "contrasts": ["E3-E1", "E4-E1", "E4-E3"],
            "interval": "95% percentile bootstrap, numpy percentile [2.5,97.5], linear interpolation; recompute full pooled metrics each replicate; frozen thresholds never retuned",
            "per_class_support_gate": "Report support for all classes; per-class CIs only if >=50 GT objects across >=20 distinct images in original reserved set; report number of valid bootstrap replicates",
            "absent_class_bootstrap": "Absent GT classes excluded by the common evaluator in each replicate; label intervals as unstable if represented class set changes in >5% of replicates",
            "interpretation": "Paired descriptive uncertainty; correlated frames may underestimate uncertainty. No formal statistical-significance claim, including for intervals excluding zero; no model tuning from intervals. Low support or unstable intervals reported as inconclusive.",
            "multiple_comparisons": "All three preregistered contrasts reported; 95% intervals are pointwise, not simultaneous family-wise intervals",
        },
        "benchmark": {
            "sample": "Sort authorized reserved rows by integer image_id; random.Random(42).shuffle; take first100; freeze IDs/hash before loading images or viewing difficulty",
            "images": 100,
            "batch": 1,
            "device": "mps",
            "dtype": "float32",
            "warmups": 10,
            "timed_passes": 3,
            "order": "same sample order each pass; rotate model execution order E1,E3,E4 / E3,E4,E1 / E4,E1,E3",
            "synchronization": "torch.mps.synchronize at all stage boundaries",
            "preprocessing": "Disk decode and RGB conversion, CPU tensor/device transfer, detector-specific normalization/resize/padding",
            "inference": "Neural network plus proposal/box decoding and detector-internal NMS for BOTH models; E4 includes both detectors and frozen fusion arithmetic, no cached detector outputs",
            "postprocessing": "Map to original coordinates, CPU transfer, common clipping and frozen operating-confidence filter",
            "end_to_end": "Sum of synchronized stages; includes local file decode; excludes model loading, artifact hashing, recording, metrics, video capture/display",
            "model_loading": "Measure/report separately, excluded from timed samples",
            "summaries": "Per-image and pooled mean/median/p95 per stage; throughput = 1000/mean_ms. Also report pass-level summaries; no live-video FPS claim",
            "hardware": "Same Mac, power mode and AC power; record CPU/GPU/OS/packages, input tensor shapes and memory samples with peak-memory limitations",
            "failure": "Stop on non-finite outputs, timing synchronization failure, unexpected settings or missing images; do not silently drop samples",
        },
        "family_gate": {
            "members": ["E1", "E3", "E4"],
            "E4": "Not implemented or tuned in Stage E. Develop/tune only calibration500; freeze exact algorithm, model/checkpoint hashes, all fusion parameters, operating threshold, code hashes and calibration evidence BEFORE reserved access.",
            "required_amendment": "Immutable E4-completion amendment referencing this protocol hash, committed before reserved access; changes to E1/E3 thresholds/checkpoints or evaluation settings require explicit preregistered protocol revision before access.",
            "authorization": "A separate future --authorize-reserved-evaluation flag plus user authorization and a complete family seal are required; Stage E commands intentionally expose no flag enabling reserved inference.",
            "one_family_policy": "One frozen family comparison; cache predictions once. Bootstrap and reporting use cached outputs, not additional model selection. On technical failure retain evidence, diagnose without tuning and preregister any necessary rerun. No favorable-run cherry-picking.",
            "if_E4_not_pursued": "Explicitly amend the family to E1/E3 before opening reserved data; do not silently drop E4 after viewing reserved results.",
        },
        "permitted_outputs": [
            "Common prediction bundles in ignored local runs",
            "AP/frozen-threshold aggregate and per-class metrics with GT/prediction support",
            "Confusion matrices and predefined error diagnostics",
            "Paired bootstrap intervals and valid-replicate counts",
            "Matched latency samples/summaries",
            "Protocol/provenance hashes and technical/faculty reports",
        ],
        "failure_rules": [
            "Reject wrong checkpoint/manifest/mapping/source/schema/package hashes",
            "Reject unauthorized image/annotation paths before loading",
            "Reject duplicate/missing images, non-finite scores/boxes, invalid classes or degenerate boxes",
            "Reject outputs with >300 predictions/image or predictions below .001",
            "Stop on inference/NMS warnings indicating truncation or timeout, OOM or incomplete bundles",
            "Never change annotations, operating thresholds, fusion settings or checkpoint selection using reserved evidence",
        ],
        "reserved_accessed": False,
        "fusion_started": False,
    }
    value = seal(protocol)
    OUTPUT.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    receipts = {
        str(p.relative_to(ROOT)): sha(p)
        for p in sorted(REPORT.glob("*"))
        if p.is_file()
    }
    (REPORT / "freeze_receipt.json").write_text(
        json.dumps(
            {
                "protocol_file": str(OUTPUT.relative_to(ROOT)),
                "protocol_file_sha256": sha(OUTPUT),
                "protocol_canonical_sha256": value["canonical_sha256"],
                "artifact_sha256": receipts,
                "reserved_accessed": False,
            },
            indent=2,
        )
        + "\n"
    )
    print("Frozen future protocol:", value["canonical_sha256"])


if __name__ == "__main__":
    main()
