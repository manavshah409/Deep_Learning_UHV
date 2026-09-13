"""Read saved subset evidence and require a committed, pushed baseline checkpoint."""

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NAME = "yolov8n_uvh26_mv_baseline_seed42_v1"


def load(path):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else {}


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)


def main():
    a = load("reports/audit/baseline_closeout_integrity.json")
    v = load("reports/audit/closeout_validation.json")
    f = load("reports/audit/baseline_subset_frozen_provenance.json")
    run = load(f"reports/tables/{NAME}_provenance.json")
    m = load(f"reports/tables/{NAME}_validation_metrics.json")
    t = load(f"reports/tables/{NAME}_latency.json")
    d = load("reports/audit/closeout_delivery.json")
    critical = [
        "configs/baseline_yolov8n.yaml",
        "configs/class_mapping.yaml",
        "src/training/train_baseline.py",
        "src/evaluation/evaluate_baseline.py",
        "src/evaluation/benchmark_inference.py",
        "docs/phase_reports/PHASE_1_BASELINE.md",
        "output/pdf/UVH26_Faculty_Progress_Report.pdf",
        "reports/audit/closeout_validation.json",
        "reports/audit/closeout_pytest.txt",
        f"reports/tables/{NAME}_provenance.json",
        f"reports/tables/{NAME}_validation_metrics.json",
        f"reports/tables/{NAME}_validation_per_class.csv",
        f"reports/tables/{NAME}_latency.json",
        "reports/error_analysis/baseline_prediction_review.json",
    ]
    committed = all(
        git("cat-file", "-e", f"HEAD:{p}").returncode == 0 for p in critical
    )
    committed = (
        committed and git("diff", "--quiet", "HEAD", "--", *critical).returncode == 0
    )
    checkpoint = d.get("baseline_commit")
    ancestor = (
        bool(checkpoint)
        and git("merge-base", "--is-ancestor", checkpoint, "HEAD").returncode == 0
    )
    checks = {
        "selected_10000_image_audit": a.get("status") == "passed"
        and a.get("images_and_labels_verified") == 10000,
        "frozen_manifests": f.get("image_audit_status") == "passed"
        and m.get("validation_manifest_sha256")
        == f.get("manifest_sha256", {}).get("val"),
        "frozen_optimizer_config": hashlib.sha256(
            (ROOT / "configs/baseline_yolov8n.yaml").read_bytes()
        ).hexdigest()
        == run.get("config_sha256"),
        "proper_training_exit_and_epochs": a.get("process_exit_code") == 0
        and run.get("status") == "completed"
        and run.get("epochs_completed") == 30,
        "proper_checkpoint": a.get("checkpoints", {}).get("weights", {}).get("readable")
        is True
        and m.get("weights_sha256") == run.get("weights", {}).get("sha256"),
        "standalone_evaluation": v.get("status") == "passed"
        and v.get("standalone_evaluation_exit_code") == 0,
        "proper_model_timing": t.get("device") == "mps"
        and t.get("measured_images") == 100
        and "Explicit torch.mps.synchronize" in t.get("stage_timing", ""),
        "qualitative_review": v.get("manual_prediction_pairs_reviewed") == 6,
        "faculty_artifacts_and_tests": v.get("pdf_layout") == "passed"
        and v.get("dashboard") == "passed"
        and v.get("portable_zip") == "passed"
        and "58 passed" in (ROOT / "reports/audit/closeout_pytest.txt").read_text(),
        "git_checkpoint": committed,
        "baseline_pushed": d.get("push_status") == "success" and ancestor,
    }
    passed = all(checks.values())
    result = dict(
        status="passed" if passed else "blocked",
        phase1_complete=passed,
        controlled_phase2_may_begin=passed,
        phase2_training_started=False,
        scope="Frozen UVH-26 MV 8000/2000 subset; unselected acquisition is not required",
        checks=checks,
        blockers=[k for k, v in checks.items() if not v],
        baseline_commit=checkpoint,
        branch=git("branch", "--show-current").stdout.strip(),
        note="This gate records eligibility only; it never launches training. Portable copies rely on saved evidence and do not include original data or weights.",
    )
    target = ROOT / "reports/audit/phase2_subset_gate.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
