"""Build an immutable, allowlisted evidence-only faculty ZIP; no dataset discovery."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def package(output: Path) -> dict:
    files = list((ROOT / "dashboard").glob("*.py")) + list(
        (ROOT / "dashboard/new_data").glob("*.py")
    )
    names = [
        "README_DEMO.md",
        "scripts/evaluate_new_data.py",
        ".streamlit/config.toml",
        "src/evaluation/__init__.py",
        "src/evaluation/error_analysis.py",
        "src/evaluation/common_metrics.py",
        "docs/reproducibility/NEW_DATA_EVALUATION.md",
        "docs/phase_reports/NEW_DATA_EVALUATION_DASHBOARD.md",
        "requirements-demo.txt",
        "requirements-new-data.txt",
        "requirements.txt",
        "requirements-dashboard.txt",
        "requirements-e2-evaluation.txt",
        "reports/comparisons/E1_E3_stageE_v2/operating_thresholds.json",
        "configs/class_mapping.yaml",
        "src/__init__.py",
        "src/data/__init__.py",
        "src/data/common.py",
        "src/video/__init__.py",
        "src/video/pipeline.py",
        "docs/reproducibility/STREAMLIT_DEMO.md",
        "docs/phase_reports/FACULTY_STREAMLIT_DELIVERABLE.md",
        "reports/audit/phase3_synthetic_smoke.json",
        "reports/audit/baseline_closeout_integrity.json",
        "reports/tables/yolov8n_uvh26_mv_baseline_seed42_v1_training.csv",
        "reports/tables/E1_yolov8s_uvh26_mv_640_seed42_training.csv",
        "reports/evaluations/E3_fasterrcnn_best_calibration500_v1/epoch_timing.csv",
    ]
    for run in [
        "yolov8n_uvh26_mv_e0_validation_seed42_v2",
        "yolov8s_uvh26_mv_e1_validation_seed42_v2",
    ]:
        names.append(f"reports/evaluations/{run}/metrics.json")
    names.append(
        "reports/evaluations/yolov8s_uvh26_mv_e1_validation_seed42_v2/confusion_matrix_normalized.png"
    )
    names.extend(
        f"reports/comparisons/E1_E3_stageE_v2/{model}_metrics.json"
        for model in ["E1", "E3"]
    )
    names.extend(
        f"reports/figures/{name}.png"
        for name in [
            "class_distribution",
            "bbox_area_distribution",
            "objects_per_image",
            "object_center_heatmap",
        ]
    )
    files.extend(ROOT / name for name in names)
    missing = [str(p.relative_to(ROOT)) for p in files if not p.is_file()]
    if missing:
        raise ValueError(f"Missing allowlisted artifacts: {missing}")
    receipt = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, str(p.relative_to(ROOT)))
        archive.writestr("MANIFEST.json", json.dumps(receipt, indent=2))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
    return {
        "files": len(files),
        "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(package(parser.parse_args().output), indent=2))
