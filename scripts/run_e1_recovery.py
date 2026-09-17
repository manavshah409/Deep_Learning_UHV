"""Post-evaluation E1 recovery stages only. Never trains or overwrites outputs."""

import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = "data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2/dataset.yaml"
E0 = "runs/yolov8n_uvh26_mv_baseline_seed42_v1/weights/best.pt"
E1 = "runs/E1_yolov8s_uvh26_mv_640_seed42/weights/best.pt"
status = ROOT / "reports/audit/E1_recovery_pipeline.json"


def main():
    assert not status.exists(), "Preserve existing recovery status"
    for name in [
        "yolov8n_uvh26_mv_e0_validation_seed42_v2",
        "yolov8s_uvh26_mv_e1_validation_seed42_v2",
    ]:
        assert (ROOT / "reports/evaluations" / name / "COMPLETE.json").exists()
    state = {
        "status": "running",
        "standalone_evaluation_exit_codes": {"E0": 0, "E1": 0},
        "stages": [],
    }
    stages = []
    for label, weights in [("E0", E0), ("E1", E1)]:
        stages.append(
            (
                label + "_benchmark",
                [
                    "-m",
                    "src.evaluation.benchmark_inference",
                    "--weights",
                    weights,
                    "--data",
                    DATA,
                    "--name",
                    label + "_common_protocol_v2",
                    "--device",
                    "mps",
                    "--count",
                    "100",
                    "--warmup",
                    "10",
                ],
            )
        )
    stages.append(
        (
            "paired_predictions",
            [
                "-m",
                "src.evaluation.paired_e1_predictions",
                "--e0-weights",
                E0,
                "--e1-weights",
                E1,
            ],
        )
    )
    stages.append(
        (
            "comparison",
            [
                "-m",
                "src.evaluation.compare_e1",
                "--e0-evaluation",
                "reports/evaluations/yolov8n_uvh26_mv_e0_validation_seed42_v2",
                "--e1-evaluation",
                "reports/evaluations/yolov8s_uvh26_mv_e1_validation_seed42_v2",
                "--e0-timing",
                "reports/tables/E0_common_protocol_v2_latency.json",
                "--e1-timing",
                "reports/tables/E1_common_protocol_v2_latency.json",
            ],
        )
    )

    def save():
        status.write_text(json.dumps(state, indent=2) + "\n")

    for name, args in stages:
        row = {
            "stage": name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        }
        state["stages"].append(row)
        save()
        with (ROOT / f"data/interim/E1_recovery_{name}.log").open("x") as log:
            p = subprocess.run(
                [sys.executable, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
        row.update(
            exit_code=p.returncode,
            status="completed" if p.returncode == 0 else "failed",
            ended_at=datetime.now(timezone.utc).isoformat(),
        )
        save()
        if p.returncode:
            state["status"] = "failed"
            save()
            raise RuntimeError(f"{name} exited {p.returncode}; preserve outputs")
    state["status"] = "awaiting_manual_review_and_closeout"
    save()


if __name__ == "__main__":
    main()
