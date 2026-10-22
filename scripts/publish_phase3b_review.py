"""After human review only: evaluate and publish small non-personal dashboard evidence."""

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.video.phase3b import evaluate_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--annotations",
        required=True,
        help="Completed human annotation revision directory",
    )
    p.add_argument("--result-id", required=True)
    a = p.parse_args()
    if not a.result_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Use a simple result ID")
    target = ROOT / "reports/phase3b/reviewed_result.json"
    if target.exists():
        raise FileExistsError("Existing reviewed result must not be overwritten")
    result = evaluate_run(
        ROOT / "reports/phase3b/protocol_v2.json",
        "phase3b_complete_v2",
        a.annotations,
        ROOT / "runs/phase3b_evaluations" / a.result_id,
    )
    public = copy.deepcopy(result)
    for row in public["review_gallery"]:
        for key in ["predicted", "ground_truth"]:
            if row[key]:
                row[key].pop("annotator", None)
                row[key].pop("notes", None)
    with target.open("x") as f:
        json.dump(public, f, indent=2, allow_nan=False)
    print(json.dumps(public["total"]))


if __name__ == "__main__":
    main()
