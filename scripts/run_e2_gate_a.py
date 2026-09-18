"""Gate A corrected immutable evaluation/benchmark sequence, never trains."""

import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def main():
    status = ROOT / "reports/audit/E2_gateA_execution.json"
    assert not status.exists()
    state = {"status": "running", "stages": [], "training_started": False}
    common = [
        "--weights",
        "runs/E1_yolov8s_uvh26_mv_640_seed42/weights/best.pt",
        "--data",
        "data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2/dataset.yaml",
        "--device",
        "mps",
    ]
    stages = []
    for size in [640, 960]:
        stages.append(
            (
                f"evaluate_{size}",
                [
                    "-m",
                    "src.evaluation.evaluate_e2",
                    *common,
                    "--imgsz",
                    str(size),
                    "--name",
                    f"E2_gateA_{size}_seed42_v2",
                    "--batch",
                    "8",
                ],
            )
        )
    for size in [640, 960]:
        stages.append(
            (
                f"benchmark_{size}",
                [
                    "-m",
                    "src.evaluation.benchmark_e2",
                    *common,
                    "--imgsz",
                    str(size),
                    "--name",
                    f"E2_gateA_{size}_benchmark_v1",
                    "--count",
                    "100",
                    "--warmup",
                    "10",
                ],
            )
        )

    def save():
        status.write_text(json.dumps(state, indent=2) + "\n")

    for name, args in stages:
        row = {
            "stage": name,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        state["stages"].append(row)
        save()
        with (ROOT / f"data/interim/E2_{name}_corrected.log").open("x") as log:
            r = subprocess.run(
                [sys.executable, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
        row.update(
            exit_code=r.returncode,
            status="completed" if r.returncode == 0 else "failed",
        )
        save()
        if r.returncode:
            state["status"] = "failed"
            save()
            raise RuntimeError(f"{name} failed; preserve evidence")
    state["status"] = "awaiting_gate_decision"
    save()


if __name__ == "__main__":
    main()
