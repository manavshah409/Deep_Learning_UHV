"""Explicit new-data ZIP evaluation; never scan project datasets."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.new_data.ingest import ingest, unpack
from dashboard.new_data.runner import publish, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("runs/new_data"))
    args = parser.parse_args()
    if not 0 <= args.bootstrap <= 10000:
        parser.error("Bootstrap count must be 0–10000.")
    if any(
        "reserved" in str(p).lower() for p in [args.zip, args.metadata, args.output]
    ):
        parser.error("Reserved inputs are prohibited.")
    import yaml

    meta = yaml.safe_load(args.metadata.read_text())
    try:
        images, records, audit = ingest(
            unpack(args.zip.read_bytes()), meta["format"], meta
        )
        result = run(images, records, audit, meta, args.device, args.bootstrap)
    except Exception as exc:  # noqa: BLE001 - CLI boundary publishes failure evidence
        import uuid

        result = {
            "run_id": "failed_" + uuid.uuid4().hex,
            "status": "FAILED",
            "reason": type(exc).__name__,
        }
        path = publish(result, args.output)
        print(json.dumps({"status": "FAILED", "bundle": str(path)}))
        return 1
    print(publish(result, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
