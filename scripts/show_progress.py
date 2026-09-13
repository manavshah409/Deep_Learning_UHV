"""Offline faculty demonstration of saved evidence; starts no jobs or downloads."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    path = ROOT / relative
    return json.loads(path.read_text()) if path.exists() else None


def main():
    print("UVH-26 VEHICLE DETECTION - VERIFIED PROGRESS")
    print("Status: proper baseline completion has not yet been established.\n")
    audit = read("reports/audit/annotation_audit.json")
    for split, row in audit["splits"].items():
        print(
            f"MV {split}: {row['images']:,} images listed; {row['annotations']:,} objects; "
            f"{row['invalid_annotations']} invalid boxes"
        )
    print("\nTests:", (ROOT / "reports/audit/pytest.txt").read_text().strip())
    run = read("reports/tables/yolov8n_uvh26_mv_smoke_seed42_provenance.json")
    print(
        f"\nSmoke training: {run['status']}; {run['epochs_completed']} epoch; "
        f"{run['duration_seconds']:.2f} seconds wall time"
    )
    print("Smoke checkpoint SHA-256:", run["weights"]["sha256"])
    print("This 64/32 pilot is not the proper baseline.")
    state = read("reports/audit/acquisition_status.json")
    print(
        "\nLast saved acquisition snapshot:", state["completed_png_files"], "PNG files"
    )
    print("This is a saved snapshot, not a live downloader status.")
    print("\nOpen output/pdf/UVH26_Faculty_Progress_Report.pdf for the review.")
    print("Read docs/faculty_review/PRESENTATION_SCRIPT.md for the speaking script.")


if __name__ == "__main__":
    main()
