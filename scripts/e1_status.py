"""Lightweight saved-artifact status; does not start or modify model jobs."""

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
E1 = "E1_yolov8s_uvh26_mv_640_seed42"


def main():
    recovery = ROOT / "reports/audit/E1_recovery_pipeline.json"
    if recovery.exists():
        print("Evaluation recovery:", json.loads(recovery.read_text())["status"])
        print(
            "Original pipeline below is preserved failure history; recovery supersedes evaluation status."
        )
    pipeline = ROOT / "reports/audit/E1_pipeline_status.json"
    if pipeline.exists():
        state = json.loads(pipeline.read_text())
        print("Pipeline:", state["status"])
        for row in state["stages"]:
            print(row["stage"], row["status"], row.get("exit_code", "not exited"))
    for name, logname in [
        (E1 + "_preflight_v1", "E1_preflight_v1.log"),
        (E1 + "_preflight_v2", "E1_preflight_v2.log"),
        (E1, "E1_training.log"),
    ]:
        p = ROOT / f"reports/tables/{name}_provenance.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        print(
            name,
            ":",
            r["status"],
            "last saved epoch:",
            r.get("last_saved_epoch", "none"),
        )
        log = ROOT / "data/interim" / logname
        if log.exists():
            with log.open("rb") as f:
                f.seek(max(0, log.stat().st_size - 4096))
                tail = f.read().decode(errors="replace")
            tail = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", tail)
            lines = [s.strip() for s in tail.splitlines() if s.strip()]
            if r["status"] == "running" and lines:
                print(lines[-1])
    print("Preflight is a pipeline check, not E1 accuracy. E0 stays frozen.")


if __name__ == "__main__":
    main()
