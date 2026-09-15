"""Run only registered E1 stages; stop on any error, preserve logs and exit evidence.

Launch only after E1 preflight verification includes manual visual acceptance.
The final paired visual review/documentation/commit remain human-agent review steps.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NAME = "E1_yolov8s_uvh26_mv_640_seed42"
DATA = "data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2/dataset.yaml"
STATUS = ROOT / "reports/audit/E1_pipeline_status.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    pre = json.loads(
        (ROOT / "reports/audit/E1_preflight_verification.json").read_text()
    )
    assert (
        pre["status"] == "passed"
        and pre["visual_review"] == "passed_with_documented_preflight_model_limitations"
    )
    assert not STATUS.exists(), (
        "Pipeline status exists; do not overwrite or automatically resume"
    )
    for name in [NAME, NAME + "_validation"]:
        assert not (ROOT / "runs" / name).exists(), f"Existing run: {name}"
    state = dict(
        experiment=NAME,
        status="running",
        started_at=now(),
        supervisor_pid=os.getpid(),
        stages=[],
    )

    def save():
        STATUS.write_text(json.dumps(state, indent=2) + "\n")

    def stage(name, arguments, logfile):
        item = dict(stage=name, status="running", started_at=now(), log=logfile)
        state["stages"].append(item)
        save()
        try:
            with (ROOT / logfile).open("x") as log:
                p = subprocess.Popen(
                    [sys.executable, *arguments],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                item["pid"] = p.pid
                save()
                code = p.wait()
            item.update(
                status="completed" if code == 0 else "failed",
                exit_code=code,
                completed_at=now(),
            )
            save()
            if code:
                raise RuntimeError(f"{name} exited {code}; see {logfile}")
        except BaseException:
            # Do not kill or overwrite surviving work; report interruption for explicit recovery.
            state.update(
                status="failed_or_interrupted", failed_stage=name, ended_at=now()
            )
            save()
            raise

    best = f"runs/{NAME}/weights/best.pt"
    e0 = "runs/yolov8n_uvh26_mv_baseline_seed42_v1/weights/best.pt"
    stage(
        "training",
        [
            "-m",
            "src.training.train_e1",
            "--config",
            f"configs/{NAME}.yaml",
            "--data",
            DATA,
            "--name",
            NAME,
        ],
        "data/interim/E1_training.log",
    )
    stage(
        "checkpoint_integrity",
        ["scripts/verify_e1_completed.py", "--process-exit-code", "0"],
        "data/interim/E1_completed_integrity.log",
    )
    stage(
        "standalone_validation",
        [
            "-m",
            "src.evaluation.evaluate_baseline",
            "--weights",
            best,
            "--data",
            DATA,
            "--name",
            NAME + "_validation",
            "--device",
            "mps",
            "--batch",
            "8",
        ],
        "data/interim/E1_standalone_validation.log",
    )
    stage(
        "common_E0_benchmark",
        [
            "-m",
            "src.evaluation.benchmark_inference",
            "--weights",
            e0,
            "--data",
            DATA,
            "--name",
            "E0_E1_common_protocol_v1",
            "--device",
            "mps",
            "--warmup",
            "10",
            "--count",
            "100",
        ],
        "data/interim/E1_common_E0_latency.log",
    )
    stage(
        "common_E1_benchmark",
        [
            "-m",
            "src.evaluation.benchmark_inference",
            "--weights",
            best,
            "--data",
            DATA,
            "--name",
            "E1_common_protocol_v1",
            "--device",
            "mps",
            "--warmup",
            "10",
            "--count",
            "100",
        ],
        "data/interim/E1_common_E1_latency.log",
    )
    stage(
        "paired_predictions",
        [
            "-m",
            "src.evaluation.paired_e1_predictions",
            "--e0-weights",
            e0,
            "--e1-weights",
            best,
        ],
        "data/interim/E1_paired_predictions.log",
    )
    stage(
        "comparison_tables",
        [
            "-m",
            "src.evaluation.compare_e1",
            "--e0-timing",
            "reports/tables/E0_E1_common_protocol_v1_latency.json",
            "--e1-timing",
            "reports/tables/E1_common_protocol_v1_latency.json",
        ],
        "data/interim/E1_comparison.log",
    )
    state.update(status="awaiting_manual_review_and_closeout", completed_at=now())
    save()
    print(
        "E1 measured stages complete; manual paired review, report and Git closeout required"
    )


if __name__ == "__main__":
    main()
