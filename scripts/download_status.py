"""Compact live acquisition status, without exposing transfer URLs or credentials."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = ROOT / "data/raw/UVH-26"
    files = list(raw.rglob("*.png"))
    stats = [p.stat() for p in files]
    plan = ROOT / "data/interim/subset_plans/uvh26_mv_baseline_subset_v1/selection.json"
    selected = json.loads(plan.read_text()) if plan.exists() else []
    status = {
        "utc": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "full_images": len(files),
        "expected_images": 26646,
        "image_gib": round(sum(s.st_size for s in stats) / 2**30, 2),
        "subset_images": sum((raw / r["source"]).is_file() for r in selected),
        "subset_expected": len(selected),
        "subset_prepared": (
            ROOT / "data/processed/uvh26_mv_baseline_subset_v1/manifest.json"
        ).exists(),
        "seconds_since_last_image": round(time.time() - max(s.st_mtime for s in stats))
        if stats
        else None,
    }
    print(json.dumps(status))
    (ROOT / "data/interim/live_status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
