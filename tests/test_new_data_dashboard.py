import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from dashboard.new_data import analysis, ingest, runner, views


def png(color="white"):
    out = io.BytesIO()
    Image.new("RGB", (100, 100), color).save(out, format="PNG")
    return out.getvalue()


def meta(fmt="yolo"):
    return {
        "name": "synthetic",
        "source": "test",
        "permission": "owned",
        "format": fmt,
        "names": ingest.NAMES,
        "independent_annotations": True,
        "used_for_training_or_tuning": False,
    }


def record():
    return {
        "image_id": 1,
        "name": "a.png",
        "width": 100,
        "height": 100,
        "gt_boxes": [[10, 10, 30, 30]],
        "gt_classes": [0],
        "boxes": [[10, 10, 30, 30]],
        "classes": [0],
        "scores": [0.9],
    }


def test_yolo_and_overlap():
    files = {"a.png": png(), "a.txt": b"0 .2 .2 .2 .2"}
    _, r, a = ingest.ingest(files, "yolo", meta())
    assert r[0]["gt_boxes"][0] == pytest.approx([10, 10, 30, 30])
    _, _, a = ingest.ingest(files, "yolo", meta(), {r[0]["sha256"]})
    assert a["independence"] == "not independent"


def test_duplicates_empty_and_missing():
    _, _, a = ingest.ingest(
        {"a.png": png(), "b.png": png()}, "unlabeled", meta("unlabeled")
    )
    assert a["duplicates"]
    with pytest.raises(ValueError):
        ingest.ingest({"a.png": png()}, "yolo", meta())
    _, r, _ = ingest.ingest({"a.png": png(), "a.txt": b""}, "yolo", meta())
    assert not r[0]["gt_boxes"]


@pytest.mark.parametrize(
    "label", [b"14 .5 .5 .2 .2", b"0 1 .5 .4 .4", b"0 nan .5 .2 .2", b"0 .5 .5 -.2 .2"]
)
def test_invalid(label):
    with pytest.raises(ValueError):
        ingest.ingest({"a.png": png(), "a.txt": label}, "yolo", meta())


def test_mapping():
    m = meta()
    m["names"] = list(reversed(m["names"]))
    with pytest.raises(ValueError):
        ingest.ingest({"a.png": png(), "a.txt": b""}, "yolo", m)


def test_coco():
    coco = {
        "images": [{"id": 1, "file_name": "a.png", "width": 100, "height": 100}],
        "categories": [{"id": 7, "name": "Hatchback"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 10, 20, 20]}
        ],
    }
    _, r, _ = ingest.ingest(
        {"a.png": png(), "annotations.json": json.dumps(coco).encode()},
        "coco",
        meta("coco"),
    )
    assert r[0]["gt_classes"] == [0]
    coco["annotations"][0]["bbox"] = [0, 0, 200, 20]
    with pytest.raises(ValueError):
        ingest.ingest(
            {"a.png": png(), "annotations.json": json.dumps(coco).encode()},
            "coco",
            meta("coco"),
        )


def test_zip_traversal():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../a.png", png())
    with pytest.raises(ValueError):
        ingest.unpack(buf.getvalue())


def test_matching_taxonomy():
    r = record()
    r["boxes"] += [
        [10, 10, 30, 30],
        [10, 10, 30, 30],
        [20, 20, 40, 40],
        [70, 70, 80, 80],
    ]
    r["classes"] += [0, 1, 0, 0]
    r["scores"] += [0.8, 0.7, 0.6, 0.5]
    tp, fp, fn, e = analysis.fixed(r)
    assert (tp.sum(), fp.sum(), fn.sum()) == (1, 4, 0)
    assert {x["kind"] for x in e} == {
        "correct detection",
        "duplicate detection",
        "classification error",
        "localization error",
        "background false positive",
    }
    r = record()
    r["scores"] = [0.2]
    assert analysis.fixed(r)[2].sum() == 1


def test_metrics_curves_strata_bootstrap():
    m = analysis.labeled([record()], 10)
    assert m["overall"]["map50"] == pytest.approx(1)
    assert m["per_class"][0]["gt_count"] == 1
    assert m["curves"][-1]["recall"] == 0
    assert m["strata"][0]["size"] == "small"
    assert m["bootstrap"]["intervals"]["recall"] == [1, 1]
    assert m["confusion"][0][0] == 1


def test_unlabeled_restrictions():
    m = analysis.unlabeled([record()])
    assert m["predictions"] == 1
    assert not any(
        k in m for k in ["precision", "recall", "f1", "map50", "confusion", "errors"]
    )


def test_latency():
    rows = [
        {
            k: v
            for k in [
                "preprocessing_ms",
                "inference_ms",
                "postprocessing_ms",
                "end_to_end_ms",
            ]
        }
        for v in [10, 20]
    ]
    m = analysis.latency(rows)
    assert m["end_to_end_ms"]["median"] == 15
    assert m["still_images_per_second"] == pytest.approx(1000 / 15)


def test_immutable(tmp_path):
    r = {"run_id": "test", "status": "COMPLETE"}
    p = runner.publish(r, tmp_path)
    assert (p / "evidence.zip").is_file()
    with pytest.raises(FileExistsError):
        runner.publish(r, tmp_path)


@pytest.mark.parametrize("page", views.NAV)
def test_pages(page):
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "dashboard/app.py"),
        default_timeout=20,
    ).run()
    app.sidebar.radio[1].set_value(page).run()
    assert not app.exception
    assert not app.error


def test_missing_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(views, "ROOT", tmp_path)
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "dashboard/app.py"),
        default_timeout=20,
    ).run()
    app.sidebar.radio[0].set_value("Live Inference Mode").run()
    app.sidebar.radio[1].set_value("New Data Evaluation").run()
    assert any("absent" in w.value for w in app.warning)


def test_empty_labeled():
    r = record()
    r.update(gt_boxes=[], gt_classes=[], boxes=[], classes=[], scores=[])
    m = analysis.labeled([r], 0)
    assert m["overall"]["map50"] is None


@pytest.mark.parametrize("page", views.NAV)
def test_completed_labeled_pages(page):
    r = record()
    metrics = analysis.labeled([r], 5)
    timing = [
        {
            "name": "a.png",
            "width": 100,
            "height": 100,
            "preprocessing_ms": 1,
            "inference_ms": 2,
            "postprocessing_ms": 1,
            "end_to_end_ms": 4,
        }
    ]
    result = {
        "status": "COMPLETE",
        "created_utc": "2026-09-29",
        "metadata": meta(),
        "records": [r],
        "metrics": metrics,
        "config": {
            "device": "cpu",
            "imgsz": 640,
            "warmup": 10,
            "batch": 1,
            "precision": "float32",
            "timing": "synthetic timing fixture",
            "class_names": ingest.NAMES,
            "confidence": 0.34,
            "nms_iou": 0.7,
            "max_det": 300,
            "ap_score_floor": 0.001,
            "metric_protocol": "common_metrics COCO v1; macro over GT-present classes",
        },
        "audit": {
            "independence": "synthetic",
            "duplicates": [],
            "overlap": [],
            "overlap_check": "synthetic",
        },
        "latency": analysis.latency(timing),
        "timing": timing,
        "model_load_ms": 1,
        "run_id": "synthetic",
    }
    r["sha256"] = "0" * 64
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "dashboard/app.py"),
        default_timeout=20,
    )
    app.session_state["new_result"] = result
    app.session_state["new_images"] = {"a.png": Image.new("RGB", (100, 100))}
    app.run()
    app.sidebar.radio[1].set_value(page).run()
    assert not app.exception
    assert not app.error
