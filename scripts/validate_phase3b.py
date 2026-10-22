"""Validate only explicit Phase 3B evidence and historical summary fingerprints."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.common import sha256
from src.video.phase3b import MODES, load_protocol
from src.video.pipeline import atomic_json


def collect(protocol_path, suffix):
    protocol = load_protocol(protocol_path)
    runs = []
    for mode in MODES:
        directory = ROOT / "runs/phase3b" / f"phase3b_{mode}_{suffix}"
        marker = json.loads((directory / "COMPLETE.json").read_text())
        for name, digest in marker["files"].items():
            if sha256(directory / name) != digest:
                raise ValueError("Run file changed: " + name)
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["protocol_sha256"] == protocol["sha256"]
        assert summary["actual_checkpoint_verified"] and summary["full_clip"]
        rows = list(csv.DictReader((directory / "frame_timings.csv").open()))
        assert [int(r["frame"]) for r in rows] == list(
            range(protocol["payload"]["video"]["frames"])
        )
        assert (
            summary["frames"] == len(rows)
            and summary["failed_frames"] == summary["dropped_frames"] == 0
        )
        assert (
            summary["counting_metrics"] is None and not summary["ground_truth_reviewed"]
        )
        runs.append(summary)
    result = {
        "protocol_sha256": protocol["sha256"],
        "runs": runs,
        "review_status": "human_annotation_required",
    }
    target = ROOT / "reports/phase3b/benchmark_summary.json"
    if target.exists():
        if json.loads(target.read_text()) != result:
            raise FileExistsError("Do not replace earlier benchmark evidence")
    else:
        atomic_json(target, result)
    return result


def historical():
    p = ROOT / "reports/phase3b/historical_hashes.json"
    values = json.loads(p.read_text())
    for name, digest in values.items():
        assert sha256(ROOT / name) == digest, name
    return len(values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="reports/phase3b/protocol_v2.json")
    parser.add_argument("--suffix", default="v2")
    args = parser.parse_args()
    result = collect(args.protocol, args.suffix)
    print(
        json.dumps(
            {
                "passed": True,
                "modes": len(result["runs"]),
                "historical_summaries_unchanged": historical(),
                "human_review_complete": False,
            }
        )
    )
