import numpy as np
import pandas as pd
import pytest
from src.evaluation.compare_e1 import (
    overall_comparison,
    per_class_comparison,
    check_timing_protocol,
)


def metrics(value=0.5):
    return dict(
        validation_manifest_sha256="frozen",
        split="val",
        imgsz=640,
        batch=8,
        device="mps",
        confidence_floor=0.001,
        nms_iou=0.7,
        max_det=300,
        precision=value,
        recall=value,
        f1=value,
        macro_f1=value,
        map50=value,
        map50_95=value,
    )


def test_delta_is_percentage_points_not_relative_percent():
    rows = overall_comparison(metrics(0.4), metrics(0.5))
    assert all(r["delta_percentage_points"] == pytest.approx(10) for r in rows)


def test_mismatched_manifest_blocks_comparison():
    a = metrics()
    b = metrics()
    b["validation_manifest_sha256"] = "changed"
    with pytest.raises(ValueError, match="validation_manifest"):
        overall_comparison(a, b)


def test_nonfinite_metric_blocks_comparison():
    b = metrics()
    b["map50"] = np.nan
    with pytest.raises(ValueError, match="Invalid metric"):
        overall_comparison(metrics(), b)


def classes():
    return pd.DataFrame(
        [
            dict(
                class_id=i,
                name=f"class{i}",
                precision=0.4,
                recall=0.3,
                f1=0.34,
                ap50=0.5,
                ap50_95=0.4,
            )
            for i in range(14)
        ]
    )


def test_perclass_aligns_by_id_and_records_regression():
    a = classes()
    b = classes().iloc[::-1].copy()
    b.loc[b.class_id == 13, "ap50_95"] = 0.1
    result = per_class_comparison(a, b)
    assert result.iloc[-1].ap50_95_delta_pp == pytest.approx(-30)


def test_mapping_mismatch_blocks_comparison():
    a = classes()
    b = classes()
    b.loc[0, "name"] = "different"
    with pytest.raises(ValueError, match="mapping"):
        per_class_comparison(a, b)


def test_timing_order_must_match():
    keys = [
        "validation_manifest_sha256",
        "device",
        "batch",
        "imgsz",
        "warmup_runs",
        "measured_images",
        "seed",
        "confidence",
        "nms_iou",
        "max_det",
        "model_precision",
        "stage_timing",
    ]
    a = {k: 1 for k in keys}
    a["records"] = [
        {"image_id": 1, "input_shape": [1, 3, 384, 640]},
        {"image_id": 2, "input_shape": [1, 3, 384, 640]},
    ]
    b = {**a, "records": list(reversed(a["records"]))}
    with pytest.raises(ValueError, match="images/order"):
        check_timing_protocol(a, b)
