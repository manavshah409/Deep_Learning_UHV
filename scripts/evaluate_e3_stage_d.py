"""Audit and independently evaluate E3 on the calibration allowlist only."""

import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.common import sha256
from src.evaluation.e3_metrics import measure
from src.experiments.accuracy_data import VehicleDataset, collate
from src.experiments.faster_rcnn import build_model
from src.experiments.stage_b import DATA, A, verify_protocol
from src.training.epoch_resume import (
    canonical_hash,
    finite_tree,
    text_row,
)

RUN = ROOT / "runs/E3_fasterrcnn_unweighted_20ep_seed42_v1"
ID = "E3_fasterrcnn_best_calibration500_v1"
BUNDLE = ROOT / "runs" / ID
REPORT = ROOT / "reports/evaluations" / ID
TOLERANCE = 1e-5  # Frozen before standalone results, absolute units on 0..1 scale.


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def table(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sync():
    torch.mps.synchronize()


def audit():
    verify_protocol()
    cfg = json.loads((RUN / "config.json").read_text())
    provenance = json.loads((RUN / "provenance.json").read_text())
    complete = json.loads((RUN / "COMPLETE.json").read_text())
    assert complete["completed_epochs"] == 20 and complete["exit_status"] == 0
    assert canonical_hash(cfg) == provenance["config_hash"] == complete["config_hash"]
    assert cfg["train_images"] == 8000 and cfg["calibration_images"] == 500
    assert cfg["calibration_manifest"] == str(
        (A / "protocol_v2/calibration_500.json").relative_to(ROOT)
    )
    for path, key in [
        (cfg["train_manifest"], "train_sha256"),
        (cfg["calibration_manifest"], "calibration_sha256"),
    ]:
        assert sha256(ROOT / path) == cfg[key]
    assert sha256(ROOT / "configs/class_mapping.yaml") == cfg["class_mapping_sha256"]
    assert sha256(ROOT / cfg["initializer"]["path"]) == cfg["initializer_sha256"]
    with (RUN / "epoch_timing.csv").open() as f:
        timing = list(csv.DictReader(f))
    assert [int(r["epoch"]) for r in timing] == list(range(1, 21))
    checkpoints = []
    for epoch, row in enumerate(timing, 1):
        assert float(row["learning_rate"]) == cfg["lr_by_epoch"][epoch - 1]
        loss = json.loads((RUN / f"losses_epoch_{epoch:03d}.json").read_text())
        assert len(loss["steps"]) == 8000 and finite_tree(loss)
        assert (
            abs(sum(loss["mean"].values()) - float(row["total_training_loss"])) < 1e-8
        )
        receipt = json.loads((RUN / f"epoch_{epoch:03d}.sha256.json").read_text())
        path = RUN / f"epoch_{epoch:03d}.pth"
        assert (
            sha256(path) == receipt["sha256"]
            and path.stat().st_size == receipt["bytes"]
        )
        checkpoints.append(dict(epoch=epoch, **receipt))
    best_epoch = max(
        range(1, 21), key=lambda e: float(timing[e - 1]["validation_map50_95"])
    )
    assert best_epoch == 13
    for alias, epoch in [("best.pth", 13), ("last.pth", 20)]:
        assert sha256(RUN / alias) == checkpoints[epoch - 1]["sha256"]
        payload = torch.load(RUN / alias, map_location="cpu", weights_only=True)
        assert payload["completed_epoch"] == epoch and payload["best_epoch"] == 13
        assert payload["run_uuid"] == provenance["run_uuid"] and payload[
            "config_hash"
        ] == canonical_hash(cfg)
        assert finite_tree(payload)
        assert [text_row(r) for r in payload["timing_history"]] == timing[:epoch]
        del payload
    launches = ROOT / "runs/launches" / RUN.name
    attempts = [launches, *sorted(launches.glob("resume_*"))]
    history = []
    for attempt in attempts:
        request = json.loads((attempt / "request.json").read_text())
        exit_info = json.loads((attempt / "exit.json").read_text())
        history.append(
            {
                "commit": request["commit"],
                "exit_code": exit_info["exit_code"],
                "finished_utc": exit_info["finished_utc"],
            }
        )
    assert history[-1]["exit_code"] == 0
    for path, expected in cfg["source_sha256"].items():
        blob = subprocess.check_output(
            ["git", "show", f"{history[0]['commit']}:{path}"], cwd=ROOT
        )
        assert hashlib.sha256(blob).hexdigest() == expected
    sessions = [
        json.loads(p.read_text()) for p in sorted((RUN / "sessions").glob("*.json"))
    ]
    for session in sessions:
        if session.get("approved_code_repair"):
            for path, expected in session["execution_source_sha256"].items():
                blob = subprocess.check_output(
                    ["git", "show", f"{history[-1]['commit']}:{path}"], cwd=ROOT
                )
                assert hashlib.sha256(blob).hexdigest() == expected
    start = min(datetime.fromisoformat(s["started_utc"]) for s in sessions)
    end = datetime.fromisoformat(complete["completed_utc"])
    result = {
        "passed": True,
        "selected_epoch": 13,
        "run_uuid": provenance["run_uuid"],
        "complete": complete,
        "config_hash": canonical_hash(cfg),
        "config_file_sha256": sha256(RUN / "config.json"),
        "configuration": cfg,
        "checkpoints": checkpoints,
        "process_history": history,
        "sessions": sessions,
        "failure_history_preserved": (RUN / "FAILURE.json").exists(),
        "current_status": "completed",
        "active_completed_epoch_seconds": float(timing[-1]["cumulative_seconds"]),
        "calendar_seconds": (end - start).total_seconds(),
        "checks": [
            "20 contiguous timing rows",
            "frozen per-epoch LR",
            "160000 finite component-loss steps",
            "20 checkpoint hashes",
            "best/last readable and all tensors finite",
            "original and repaired source commits",
            "manifest/class-map/initializer hashes",
            "completion plus final process exit zero",
        ],
        "reserved_split_accessed": False,
    }
    save(REPORT / "audit.json", result)
    table(REPORT / "epoch_timing.csv", timing)
    return cfg


def partition(model, path):
    sync()
    start = time.perf_counter()
    with Image.open(path) as image:
        tensor = to_tensor(image.convert("RGB")).to("mps")
    original = [tensor.shape[-2:]]
    images, _ = model.transform([tensor], None)
    sync()
    prep = time.perf_counter()
    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    proposals, _ = model.rpn(images, features, None)
    detections, _ = model.roi_heads(features, proposals, images.image_sizes, None)
    sync()
    inferred = time.perf_counter()
    detections = model.transform.postprocess(detections, images.image_sizes, original)
    raw = {k: v.cpu() for k, v in detections[0].items()}
    output = {k: v[raw["scores"] >= 0.25] for k, v in raw.items()}
    sync()
    end = time.perf_counter()
    return (
        {
            "preprocessing_ms": (prep - start) * 1000,
            "inference_ms": (inferred - prep) * 1000,
            "postprocessing_ms": (end - inferred) * 1000,
            "end_to_end_ms": (end - start) * 1000,
        },
        output,
        raw,
    )


def benchmark(model, rows, load_seconds):
    selected = sorted(rows, key=lambda r: (r["image_id"], r["image"]))
    random.Random(42).shuffle(selected)
    selected = selected[:100]
    save(REPORT / "benchmark_sample.json", selected)
    with torch.no_grad():
        path = DATA / selected[0]["image"]
        with Image.open(path) as im:
            tensor = to_tensor(im.convert("RGB")).to("mps")
        full = {k: v.cpu() for k, v in model([tensor])[0].items()}
        _, _, segmented = partition(model, path)
        assert all(
            torch.allclose(full[k], segmented[k], atol=1e-4, rtol=1e-5) for k in full
        )
        for i in range(10):
            partition(model, DATA / selected[i]["image"])
        records = []
        for row in selected:
            measured, _, _ = partition(model, DATA / row["image"])
            measured["image_id"] = row["image_id"]
            measured["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
            measured["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
            records.append(measured)
    table(REPORT / "benchmark_samples.csv", records)
    stats = {}
    for key in [
        "preprocessing_ms",
        "inference_ms",
        "postprocessing_ms",
        "end_to_end_ms",
    ]:
        values = [r[key] for r in records]
        stats[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
        }
    result = {
        "device": "mps",
        "dtype": "float32",
        "batch": 1,
        "warmups": 10,
        "timed_images": 100,
        "timed_passes": 1,
        "resize": "short480/max640 aspect preserving",
        "confidence": 0.25,
        "model_score_floor": 0.001,
        "nms": 0.5,
        "max_detections": 300,
        "model_loading_seconds_excluded": load_seconds,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_bytes": (RUN / "best.pth").stat().st_size,
        "timings": stats,
        "inference_only_images_per_second": 1000 / stats["inference_ms"]["mean"],
        "end_to_end_still_images_per_second": 1000 / stats["end_to_end_ms"]["mean"],
        "allocated_bytes_sampled_max": max(r["mps_allocated_bytes"] for r in records),
        "driver_bytes_sampled_max": max(r["mps_driver_bytes"] for r in records),
        "definitions": {
            "preprocessing": "File decode, tensor conversion/transfer, model normalization, resizing and batch padding",
            "inference": "Backbone, RPN and RoI heads INCLUDING their internal proposal/box decoding and NMS",
            "postprocessing": "Map boxes to original resolution, CPU transfer and confidence .25 filtering",
            "end_to_end": "Sum of synchronized stages; excludes model load, prior integrity hashes, display/capture, metrics and logging",
        },
        "limitations": "Warm filesystem/model; one pass; memory sampled after each image, not peak memory; no video FPS or matched E1 comparison",
        "split_forward_matches_standard": True,
    }
    save(REPORT / "benchmark.json", result)


def main():
    BUNDLE.mkdir(exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=False)
    save(
        REPORT / "protocol.json",
        {
            "id": ID,
            "scope": "calibration500 only",
            "selected_epoch": 13,
            "numerical_tolerance_absolute": TOLERANCE,
            "confidence": 0.25,
            "matching_iou": 0.5,
            "ap_score_floor": 0.001,
            "nms": 0.5,
            "max_detections": 300,
            "benchmark_seed": 42,
            "benchmark_count": 100,
            "warmups": 10,
            "reserved_split_accessed": False,
            "evaluator_source_sha256": {
                "scripts/evaluate_e3_stage_d.py": sha256(Path(__file__)),
                "src/evaluation/e3_metrics.py": sha256(
                    ROOT / "src/evaluation/e3_metrics.py"
                ),
            },
        },
    )
    cfg = audit()
    assert torch.backends.mps.is_available()
    names = json.loads((A / "protocol_v2/weighting.json").read_text())["class_names"]
    rows = json.loads((A / "protocol_v2/calibration_500.json").read_text())
    loader = DataLoader(
        VehicleDataset(DATA, rows),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    started = time.perf_counter()
    model = build_model(min_size=480, max_size=640)
    payload = torch.load(RUN / "best.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    del payload
    model.to("mps").eval()
    sync()
    loading_seconds = time.perf_counter() - started
    start = time.perf_counter()
    records = []
    with torch.no_grad():
        for images, targets in loader:
            prediction = model([images[0].to("mps")])[0]
            prediction = {k: v.detach().cpu() for k, v in prediction.items()}
            assert finite_tree(prediction)
            target = targets[0]
            records.append(
                {
                    "image_id": int(target["image_id"]),
                    "width": images[0].shape[2],
                    "height": images[0].shape[1],
                    "gt_boxes": target["boxes"].tolist(),
                    "gt_classes": (target["labels"] - 1).tolist(),
                    "boxes": prediction["boxes"].tolist(),
                    "classes": (prediction["labels"] - 1).tolist(),
                    "scores": prediction["scores"].tolist(),
                }
            )
    sync()
    prediction_seconds = time.perf_counter() - start
    save(BUNDLE / "predictions.json", records)
    summary, classes, confusion, pr, errors = measure(records, names)
    summary.update(
        evaluation_seconds=time.perf_counter() - start,
        prediction_seconds=prediction_seconds,
        checkpoint_sha256=sha256(RUN / "best.pth"),
        configuration_sha256=canonical_hash(cfg),
        manifest_sha256=sha256(A / "protocol_v2/calibration_500.json"),
        selected_epoch=13,
        scope="Independent calibration500 evaluation only; no E1 comparison or reserved evaluation",
    )
    save(REPORT / "metrics.json", summary)
    table(REPORT / "per_class.csv", classes)
    save(
        REPORT / "confusion_matrix.json",
        {
            "labels": names + ["background"],
            "matrix": confusion.tolist(),
            "method": summary["confusion_method"],
        },
    )
    save(REPORT / "image_errors.json", errors)
    np.savez_compressed(
        REPORT / "pr_curves.npz",
        precision=pr,
        recall=np.linspace(0, 1, 101),
        iou=np.linspace(0.5, 0.95, 10),
    )
    reference = json.loads((RUN / "metrics_epoch_013.json").read_text())
    differences = {
        k: summary[k] - reference[k]
        for k in ["precision", "recall", "map50", "map50_95"]
    }
    agreement = {
        "tolerance_absolute": TOLERANCE,
        "differences": differences,
        "passed": all(abs(v) <= TOLERANCE for v in differences.values()),
    }
    save(REPORT / "agreement.json", agreement)
    if not agreement["passed"]:
        raise ValueError(
            "Standalone metrics disagree with epoch13; preserve results and investigate"
        )
    hardware = {}
    for key in ["machdep.cpu.brand_string", "hw.memsize"]:
        result = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, check=False
        )
        hardware[key] = (
            result.stdout.strip() if result.returncode == 0 else "unavailable"
        )
    save(
        REPORT / "environment.json",
        {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "numpy": np.__version__,
            "system": platform.system(),
            "os_release": platform.release(),
            "hardware": hardware,
        },
    )
    benchmark(model, rows, loading_seconds)
    save(
        BUNDLE / "COMPLETE.json",
        {"evaluation_and_benchmark_complete": True, "reserved_split_accessed": False},
    )
    print(json.dumps({"metrics": summary, "agreement": agreement}))


if __name__ == "__main__":
    main()
