"""Offline faculty demonstration; reads saved artifacts and starts no jobs."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "yolov8n_uvh26_mv_baseline_seed42_v1"


def read(relative):
    return json.loads((ROOT / relative).read_text())


def main():
    print("UVH-26 PHASE 1 - SUBSET VALIDATION BASELINE")
    run = read(f"reports/tables/{NAME}_provenance.json")
    print(
        f"Training: {run['status']}; {run['epochs_completed']} epochs; best epoch {run['best_epoch']}; early stop {run['early_stopped']}"
    )
    print("Catalogue audit: 26,646 image records / 316,220 boxes.")
    print("Local integrity audit: 8,000 train + 2,000 validation images; 14 classes.")
    print("Acquisition/integrity of unselected images remains incomplete.")
    metrics = read(f"reports/tables/{NAME}_validation_metrics.json")
    for key in ["precision", "recall", "f1", "macro_f1", "map50", "map50_95"]:
        print(f"{key}: {metrics[key]:.6f}")
    print(metrics["f1_note"])
    timing = read(f"reports/tables/{NAME}_latency.json")
    for stage, values in timing["summary"].items():
        print(
            f"{stage}: median {values['median_ms']:.3f} ms; p95 {values['p95_ms']:.3f} ms; FPS {values['fps_from_total_time']:.2f}"
        )
    print(timing["end_to_end_definition"])
    print("Not a video FPS or production readiness claim.")
    print("Best checkpoint SHA-256:", run["weights"]["sha256"])
    print("Tests:", (ROOT / "reports/audit/closeout_pytest.txt").read_text().strip())
    gate = ROOT / "reports/audit/phase2_subset_gate.json"
    if gate.exists():
        print("Subset Phase 2 gate:", json.loads(gate.read_text())["status"])
    print("Phase 2 training has not started.")


if __name__ == "__main__":
    main()
