"""Validate E2 Gate A result identity, sums, timing and E0/E1 preservation."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.data.common import ROOT, sha256
from src.evaluation.compare_e2 import check_protocol


def main():
    load = lambda p: json.loads((ROOT / p).read_text())
    metrics = []
    counts = []
    for size in [640, 960]:
        root = Path(f"reports/evaluations/E2_gateA_{size}_seed42_v2")
        marker = load(root / "COMPLETE.json")
        assert all(
            sha256(ROOT / root / f) == h for f, h in marker["files_sha256"].items()
        )
        m = load(root / "metrics.json")
        metrics.append(m)
        p = load(root / "prediction_counts.json")
        counts.append(p)
        assert (
            len(p) == 2000 and sum(r["detections"] for r in p) == m["prediction_count"]
        )
        assert np.isclose(
            m["f1"], 2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"])
        )
        cm = np.array(load(root / "confusion_matrix.json")["matrix"])
        assert cm.shape == (15, 15) and cm[:, :14].sum() == 24342
        assert cm[:14, :].sum() == m["prediction_count"]
        size_ap = load(root / "size_ap.json")
        assert size_ap["prediction_class_map"] == list(range(1, 15))
        assert size_ap["groups"]["all"]["objects"] == 24342
        for group in size_ap["groups"].values():
            assert 0 <= group["ap50_95"] <= 1
            assert sum(r["objects"] for r in group["per_class"]) == group["objects"]
            for r in group["per_class"]:
                assert (r["ap50_95"] is None) == (r["objects"] == 0)
        for p in (ROOT / root).glob("*.png"):
            with Image.open(p) as im:
                im.verify()
        t = load(f"reports/tables/E2_gateA_{size}_benchmark_v1_latency.json")
        assert t["weights_sha256"] == m["weights_sha256"] and len(t["records"]) == 100
        for stage, s in t["summary"].items():
            a = [r[stage] for r in t["records"]]
            assert np.isclose(np.mean(a), s["mean_ms"]) and np.isclose(
                np.percentile(a, 95), s["p95_ms"]
            )
    check_protocol(*metrics)
    assert [r["image"] for r in counts[0]] == [r["image"] for r in counts[1]]
    original = load("reports/audit/E2_preservation.json")
    assert all(sha256(ROOT / p) == h for p, h in original["frozen_file_sha256"].items())
    assert not (ROOT / "runs/E2_yolov8s_uvh26_mv_960_seed42").exists()
    result = {
        "status": "passed",
        "resolutions": [640, 960],
        "images_each": 2000,
        "classes": 14,
        "original_E0_E1_artifacts_unchanged": True,
        "full_E2_training_started": False,
    }
    (ROOT / "reports/audit/E2_artifact_validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(result)


if __name__ == "__main__":
    main()
