"""Fresh guarded calibration predictions and common E1/E3 protocol calibration."""

import argparse
import csv
import importlib.metadata
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.common_metrics import measure
from src.evaluation.comparison_protocol import (
    CALIBRATION,
    CALIBRATION_SHA,
    GRID,
    MAPPING_SHA,
    NAMES,
    RESERVED_SHA,
    SCHEMA_VERSION,
    OutsideImage,
    detection,
    guard_path,
    load_manifest,
    seal,
    sha,
    unletterbox_boxes,
    validate_bundle,
)
from src.evaluation.complementarity import (
    paired_analysis,
    select_threshold,
    threshold_curve,
)
from src.experiments.accuracy_data import VehicleDataset
from src.experiments.faster_rcnn import build_model

REPORT = ROOT / "reports/comparisons/E1_E3_stageE_v2"
DATA = ROOT / "data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2"
MODELS = {
    "E1": {
        "id": "E1_yolov8s_calibration500_common_v2",
        "checkpoint": "runs/E1_yolov8s_uvh26_mv_640_seed42/weights/best.pt",
        "sha256": "9f1045381024445b50e33539791da224b732d61781c73b347013477ebb077fab",
        "epoch": 22,
        "preprocessing": {
            "kind": "Ultralytics OpenCV BGR decode, stride-aligned rectangular letterbox, RGB CHW float32 /255",
            "imgsz": 640,
            "rect": True,
            "stride": 32,
            "padding_value": 114,
            "scaleup": True,
        },
        "nms_iou": 0.7,
        "agnostic_nms": False,
    },
    "E3": {
        "id": "E3_fasterrcnn_calibration500_common_v2",
        "checkpoint": "runs/E3_fasterrcnn_unweighted_20ep_seed42_v1/best.pth",
        "sha256": "2d035a95206475b1e9939c5686a731a2427f84918389a2aa4d59148ed5ba5df9",
        "epoch": 13,
        "preprocessing": {
            "kind": "PIL RGB float32 /255; torchvision normalization and aspect resize with divisible32 batch padding",
            "short_side": 480,
            "max_side": 640,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "nms_iou": 0.5,
        "rpn_nms_iou": 0.7,
        "rpn_pre_nms_top_n_test": 1000,
        "rpn_post_nms_top_n_test": 1000,
    },
}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def table(path, rows):
    with Path(path).open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def configuration(model):
    return {
        **MODELS[model],
        "device": "mps",
        "dtype": "float32",
        "batch": 1,
        "workers": 0,
        "score_floor": 0.001,
        "max_detections": 300,
        "augmentation": False,
    }


def preregister():
    files = [
        "src/evaluation/common_metrics.py",
        "src/evaluation/comparison_protocol.py",
        "src/evaluation/complementarity.py",
        "src/evaluation/error_analysis.py",
        "src/experiments/accuracy_data.py",
        "src/experiments/faster_rcnn.py",
        "scripts/calibrate_e1_e3_stage_e.py",
    ]
    plan = {
        "version": 1,
        "scope": "calibration500 only",
        "image_order": "ascending integer image_id",
        "calibration_sha256": CALIBRATION_SHA,
        "reserved_sha256_recorded_only": RESERVED_SHA,
        "mapping_sha256": MAPPING_SHA,
        "schema_version": SCHEMA_VERSION,
        "class_names": NAMES,
        "models": {m: configuration(m) for m in MODELS},
        "threshold_rule": {
            "grid": GRID,
            "objective": "macro class F1 over GT-present classes; absent classes excluded",
            "tie": "round objective to 12 decimal places, select higher confidence",
            "matching_iou": 0.5,
            "matching": "per-image class-aware score-descending greedy maximum-IoU unused GT; stable prediction-index score tie, lowest GT index IoU tie",
        },
        "AP": {
            "score_floor": 0.001,
            "iou_thresholds": [0.5 + i * 0.05 for i in range(10)],
            "maxDets": [1, 10, 300],
            "area": "original-pixel box area; COCO all/small[0,32^2]/medium[32^2,96^2]/large[96^2,1e10]",
            "absent_classes": "COCO precision=-1 excluded from macro AP; zero-prediction supported class AP=0",
            "category_ids": "COCO1..14; common0..13; mapping order fixed",
        },
        "coordinates": "Fail on non-finite/non-positive raw boxes or invalid classes/scores. Map to original coordinates before clipping; positive-area boxes wholly outside image are explicitly rejected and counted as out_of_frame, not exported. Clip intersecting boxes to image bounds identically for both detectors. Explicitly count/exclude Faster R-CNN background0. YOLO adapter retains pre-clip mapped coordinates so padding-only boxes are distinguishable from invalid raw boxes.",
        "support_rule": "<50 GT objects marked low support; no strong single rare-class claims",
        "source_sha256": {f: sha(ROOT / f) for f in files},
        "packages": {
            n: importlib.metadata.version(n)
            for n in [
                "torch",
                "torchvision",
                "ultralytics",
                "numpy",
                "pycocotools",
                "Pillow",
            ]
        },
        "reserved_accessed": False,
        "fusion_started": False,
    }
    save(REPORT / "calibration_plan.json", seal(plan))
    return plan


def generate(model_id, rows):
    cfg = configuration(model_id)
    checkpoint = ROOT / cfg["checkpoint"]
    if sha(checkpoint) != cfg["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    bundle = ROOT / "runs" / cfg["id"]
    bundle.mkdir(exist_ok=False)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if model_id == "E1":
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect.predict import DetectionPredictor

        class ShapePredictor(DetectionPredictor):
            def preprocess(self, images):
                tensor = super().preprocess(images)
                self.last_shape = list(tensor.shape)
                return tensor

            def construct_result(self, pred, img, orig_img, img_path):
                from ultralytics.engine.results import Results

                pred[:, :4] = unletterbox_boxes(
                    pred[:, :4], img.shape[2:], orig_img.shape[:2]
                )
                return Results(
                    orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6]
                )

        model = YOLO(checkpoint)
        if [model.names[i] for i in range(14)] != NAMES:
            raise ValueError("YOLO class mapping mismatch")
    else:
        model = build_model()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["completed_epoch"] != 13:
            raise ValueError("Wrong E3 epoch")
        model.load_state_dict(state["model"])
        del state
        model.to("mps").float().eval()
        if (
            model.roi_heads.nms_thresh != 0.5
            or model.rpn.nms_thresh != 0.7
            or model.rpn.pre_nms_top_n() != 1000
            or model.rpn.post_nms_top_n() != 1000
        ):
            raise ValueError("Unexpected E3 inference configuration")
        shapes = []
        hook = model.transform.register_forward_hook(
            lambda _, args, output: shapes.append(list(output[0].tensors.shape))
        )
    dataset = VehicleDataset(DATA, rows)
    records = []
    background = clipped = out_of_frame = 0
    start = time.perf_counter()
    path = bundle / "predictions.jsonl"
    with path.open("x") as stream, torch.no_grad():
        for i, row in enumerate(rows):
            guard_path(DATA / row["image"], DATA, rows)
            guard_path(DATA / row["label"], DATA, rows)
            image, target = dataset[i]
            h, w = image.shape[1:]
            if model_id == "E1":
                result = model.predict(
                    str(DATA / row["image"]),
                    predictor=ShapePredictor,
                    device="mps",
                    imgsz=640,
                    rect=True,
                    batch=1,
                    conf=0.001,
                    iou=0.7,
                    max_det=300,
                    agnostic_nms=False,
                    augment=False,
                    verbose=False,
                    save=False,
                )[0]
                if next(model.model.parameters()).dtype != torch.float32:
                    raise ValueError("YOLO dtype is not float32")
                boxes = result.boxes.xyxy.cpu().tolist()
                scores = result.boxes.conf.cpu().tolist()
                classes = result.boxes.cls.cpu().tolist()
                shape = model.predictor.last_shape
            else:
                result = {
                    k: v.detach().cpu() for k, v in model([image.to("mps")])[0].items()
                }
                boxes = result["boxes"].tolist()
                scores = result["scores"].tolist()
                classes = result["labels"].tolist()
                shape = shapes[-1]
            predictions = []
            for box, score, cls in zip(boxes, scores, classes):
                try:
                    d = detection(box, score, cls, model_id, w, h)
                except OutsideImage:
                    out_of_frame += 1
                    continue
                if d is None:
                    background += 1
                    continue
                if score < 0.001:
                    raise ValueError("Detector returned sub-floor score")
                clipped += int(d["clipped"])
                predictions.append(d)
            common = {
                "schema_version": SCHEMA_VERSION,
                "image_id": row["image_id"],
                "image_key": row["image"],
                "model_id": model_id,
                "checkpoint_sha256": cfg["sha256"],
                "width": w,
                "height": h,
                "preprocessing": cfg["preprocessing"],
                "inference_configuration": cfg,
                "actual_tensor_shape": shape,
                "detections": predictions,
            }
            stream.write(
                json.dumps(common, separators=(",", ":"), allow_nan=False) + "\n"
            )
            records.append(
                {
                    "image_id": row["image_id"],
                    "width": w,
                    "height": h,
                    "gt_boxes": target["boxes"].tolist(),
                    "gt_classes": (target["labels"] - 1).tolist(),
                    "boxes": [p["xyxy"] for p in predictions],
                    "classes": [p["class_id"] for p in predictions],
                    "scores": [p["confidence"] for p in predictions],
                }
            )
    torch.mps.synchronize()
    receipt = {
        "id": cfg["id"],
        "model_id": model_id,
        "checkpoint_sha256": cfg["sha256"],
        "manifest_sha256": CALIBRATION_SHA,
        "mapping_sha256": MAPPING_SHA,
        "prediction_sha256": sha(path),
        "prediction_bytes": path.stat().st_size,
        "images": 500,
        "predictions": sum(len(r["scores"]) for r in records),
        "background_excluded": background,
        "out_of_frame_rejected": out_of_frame,
        "boxes_clipped": clipped,
        "seconds": time.perf_counter() - start,
        "inference_configuration": cfg,
        "schema_version": SCHEMA_VERSION,
    }
    validate_bundle(path, receipt, [r["image_id"] for r in rows])
    save(bundle / "receipt.json", receipt)
    save(
        bundle / "COMPLETE.json",
        {"passed": True, "receipt_sha256": sha(bundle / "receipt.json")},
    )
    save(REPORT / f"{model_id}_prediction_receipt.json", receipt)
    if model_id == "E3":
        hook.remove()
    del model
    torch.mps.empty_cache()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CALIBRATION)
    args = parser.parse_args()
    # This development command intentionally exposes no reserved authorization flag.
    rows = load_manifest(args.manifest)
    if len(rows) != 500:
        raise ValueError("Expected calibration500")
    if sha(ROOT / "configs/class_mapping.yaml") != MAPPING_SHA:
        raise ValueError("Class map changed")
    mapping = yaml.safe_load((ROOT / "configs/class_mapping.yaml").read_text())
    if [r["name"] for r in mapping] != NAMES or [r["yolo_id"] for r in mapping] != list(
        range(14)
    ):
        raise ValueError("Class map invalid")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    REPORT.mkdir(parents=True, exist_ok=False)
    plan = preregister()
    all_records = {}
    thresholds = {}
    for model in MODELS:
        records = generate(model, rows)
        all_records[model] = records
        curve = threshold_curve(records)
        table(REPORT / f"{model}_threshold_curve.csv", curve)
        threshold = select_threshold(curve)
        thresholds[model] = threshold
        metrics, classes, matrix, _, errors = measure(
            records, NAMES, confidence=threshold
        )
        selected = next(r for r in curve if r["threshold"] == threshold)
        if abs(selected["macro_class_f1"] - metrics["macro_class_f1"]) > 1e-12:
            raise ValueError("Threshold/evaluator mismatch")
        save(REPORT / f"{model}_metrics.json", metrics)
        table(REPORT / f"{model}_per_class.csv", classes)
        save(
            REPORT / f"{model}_confusion.json",
            {"labels": NAMES + ["background"], "matrix": matrix.tolist()},
        )
        save(
            REPORT / f"{model}_errors.json",
            {
                "tp": sum(e["correct"] for e in errors),
                "fp": sum(e["class_aware_fp"] for e in errors),
                "fn": sum(e["class_aware_fn"] for e in errors),
                **{
                    key: sum(len(e[key]) for e in errors)
                    for key in [
                        "duplicate_predictions",
                        "localization_candidates",
                        "background_candidates",
                        "missed_original_small",
                        "missed_resized_small",
                    ]
                },
                "limitations": "Geometric error candidates; not exhaustive mutually exclusive FP causes. Small sizes use original32px and E3-reference-resized16px area; no actual YOLO resize-size comparison.",
            },
        )
        save(ROOT / "runs" / MODELS[model]["id"] / "evaluation_records.json", records)
    result, images = paired_analysis(all_records["E1"], all_records["E3"], thresholds)
    save(REPORT / "complementarity.json", result)
    table(REPORT / "paired_image_recoveries.csv", images)
    save(
        REPORT / "operating_thresholds.json",
        seal(
            {
                "thresholds": thresholds,
                **plan["threshold_rule"],
                "calibration_manifest_sha256": CALIBRATION_SHA,
            }
        ),
    )
    save(
        REPORT / "COMPLETE.json",
        {
            "passed": True,
            "calibration_images": 500,
            "reserved_accessed": False,
            "fusion_started": False,
        },
    )
    print(
        json.dumps(
            {
                "completed": True,
                "thresholds": thresholds,
                "complementarity": result["counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
