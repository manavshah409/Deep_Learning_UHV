"""Compact training/metric plots and deterministic calibration-only review panels."""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import findfont
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.error_analysis import iou_matrix

ID = "E3_fasterrcnn_best_calibration500_v1"
REPORT = ROOT / "reports/evaluations" / ID
BUNDLE = ROOT / "runs" / ID
DATA = ROOT / "data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    with (REPORT / "epoch_timing.csv").open() as f:
        epochs = list(csv.DictReader(f))
    x = np.arange(1, 21)
    fig, ax = plt.subplots(3, 2, figsize=(12, 11), layout="constrained")
    for key in [
        "total_training_loss",
        "classification_loss",
        "box_regression_loss",
        "rpn_objectness_loss",
        "rpn_box_loss",
    ]:
        ax[0, 0].plot(x, [float(e[key]) for e in epochs], label=key)
    ax[0, 0].legend(fontsize=7)
    ax[0, 0].set_title("Training component losses")
    for key in ["validation_map50", "validation_map50_95"]:
        ax[0, 1].plot(x, [float(e[key]) for e in epochs], label=key)
    ax[0, 1].legend()
    ax[0, 1].set_title("Calibration500 AP")
    ax[1, 0].step(x, [float(e["learning_rate"]) for e in epochs], where="mid")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_title("Frozen LR")
    for key in ["training_seconds", "validation_seconds", "total_epoch_seconds"]:
        ax[1, 1].plot(x, [float(e[key]) for e in epochs], label=key)
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("Epoch work seconds (excludes checkpoint commit)")
    ax[2, 0].plot(x, [float(e["cumulative_seconds"]) / 3600 for e in epochs])
    ax[2, 0].set_title("Cumulative completed active hours")
    with (REPORT / "per_class.csv").open() as f:
        classes = list(csv.DictReader(f))
    ax[2, 1].barh([c["name"] for c in classes], [float(c["ap50_95"]) for c in classes])
    ax[2, 1].set_title("Standalone calibration per-class AP50:95")
    for a in ax.flat:
        a.grid(alpha=0.2)
        if a is not ax[2, 1]:
            a.axvline(13, color="black", linestyle="--", alpha=0.5)
            a.set_xlabel("Epoch; dashed line = selected 13")
    fig.savefig(REPORT / "training_curves.png", dpi=140)
    plt.close(fig)
    matrix = json.loads((REPORT / "confusion_matrix.json").read_text())
    fig, ax = plt.subplots(figsize=(11, 9), layout="constrained")
    shown = ax.imshow(np.log1p(np.asarray(matrix["matrix"])), cmap="Blues")
    fig.colorbar(shown, ax=ax, label="log(1 + count)")
    ax.set_xticks(range(15), matrix["labels"], rotation=70, ha="right")
    ax.set_yticks(range(15), matrix["labels"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title("Calibration500 confusion; confidence .25 / IoU .5")
    fig.savefig(REPORT / "confusion_matrix.png", dpi=130)
    plt.close(fig)
    pr = np.load(REPORT / "pr_curves.npz")
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    for i, c in enumerate(classes):
        y = pr["precision"][0, :, i]
        ax.plot(
            pr["recall"], np.maximum(y, 0), label=f"{c['name']} (GT {c['gt_count']})"
        )
    ax.set(
        xlabel="Recall",
        ylabel="Interpolated precision",
        title="COCO PR curves at IoU .50; score floor .001",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)
    fig.savefig(REPORT / "pr_curves.png", dpi=140)
    plt.close(fig)
    records = json.loads((BUNDLE / "predictions.json").read_text())
    errors = {
        r["image_id"]: r for r in json.loads((REPORT / "image_errors.json").read_text())
    }
    lookup = {
        r["image_id"]: r
        for r in json.loads(
            (
                ROOT / "reports/accuracy_stage_a/protocol_v2/calibration_500.json"
            ).read_text()
        )
    }
    stats = []
    for r in records:
        boxes = np.asarray(r["gt_boxes"]).reshape(-1, 4)
        counts = np.bincount(r["gt_classes"], minlength=14)
        overlap = iou_matrix(boxes, boxes)
        np.fill_diagonal(overlap, 0)
        stats.append(
            {
                "image_id": r["image_id"],
                "objects": len(boxes),
                "small": int(
                    np.sum(
                        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                        < 32**2
                    )
                ),
                "two_wheelers": int(counts[7]),
                "three_wheelers": int(counts[6]),
                "heavy": int(counts[4] + counts[5]),
                "mini_bus": int(counts[9]),
                "tempo": int(counts[10]),
                "van": int(counts[12]),
                "others": int(counts[13]),
                "overlap_pairs": int((overlap >= 0.1).sum() // 2),
                "misses": errors[r["image_id"]]["class_aware_fn"],
                "fp": errors[r["image_id"]]["class_aware_fp"],
            }
        )
    criteria = [
        "objects",
        "small",
        "two_wheelers",
        "three_wheelers",
        "heavy",
        "mini_bus",
        "tempo",
        "van",
        "others",
        "overlap_pairs",
        "misses",
        "fp",
    ]
    selected = {}
    for key in criteria:
        row = min(stats, key=lambda r: (-r[key], r["image_id"]))
        selected.setdefault(row["image_id"], []).append(key)
    row = min(
        (s for s in stats if s["objects"] > 0),
        key=lambda r: (r["objects"], r["image_id"]),
    )
    selected.setdefault(row["image_id"], []).append("sparse")
    output = BUNDLE / "qualitative"
    output.mkdir(exist_ok=False)
    font = ImageFont.truetype(findfont("DejaVu Sans"), 13)
    selection = []
    for iid, reasons in selected.items():
        r = next(r for r in records if r["image_id"] == iid)
        error = errors[iid]
        with Image.open(DATA / lookup[iid]["image"]) as image:
            source = image.convert("RGB")
        width = 1000
        scale = width / source.width
        height = round(source.height * scale)
        canvas = Image.new("RGB", (width * 2, height + 70), "white")
        canvas.paste(source.resize((width, height)), (0, 40))
        canvas.paste(source.resize((width, height)), (width, 40))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 5),
            f"GT: {iid}; selection: {', '.join(reasons)}",
            fill="black",
            font=font,
        )
        draw.text(
            (width + 8, 5),
            "Predictions >= .25: green correct / orange confusion / red unmatched",
            fill="black",
            font=font,
        )
        for gi, (box, cls) in enumerate(zip(r["gt_boxes"], r["gt_classes"])):
            coordinates = [
                box[0] * scale,
                box[1] * scale + 40,
                box[2] * scale,
                box[3] * scale + 40,
            ]
            color = "red" if gi in error["missed_gt"] else "lime"
            draw.rectangle(coordinates, outline=color, width=2)
            draw.text(
                coordinates[:2],
                str(cls),
                font=font,
                fill=color,
                stroke_width=1,
                stroke_fill="black",
            )
        keep = [i for i, s in enumerate(r["scores"]) if s >= 0.25]
        matches = {m["prediction"]: m for m in error["confusion_matches"]}
        for pi, original in enumerate(keep):
            box = r["boxes"][original]
            cls = r["classes"][original]
            score = r["scores"][original]
            color = (
                "lime"
                if pi in matches and matches[pi]["correct_class"]
                else "orange"
                if pi in matches
                else "red"
            )
            coordinates = [
                box[0] * scale + width,
                box[1] * scale + 40,
                box[2] * scale + width,
                box[3] * scale + 40,
            ]
            draw.rectangle(coordinates, outline=color, width=2)
            draw.text(
                coordinates[:2],
                f"{cls}:{score:.2f}",
                font=font,
                fill=color,
                stroke_width=1,
                stroke_fill="black",
            )
        draw.text(
            (8, height + 45),
            "GT red = unmatched by same-class IoU .5. Overlap is an occlusion proxy, not an occlusion annotation.",
            fill="black",
            font=font,
        )
        path = output / f"{iid}.jpg"
        canvas.save(path, quality=90)
        selection.append(
            {
                "image_id": iid,
                "reasons": reasons,
                "source_image": lookup[iid]["image"],
                "render": str(path.relative_to(ROOT)),
                "errors": error,
            }
        )
    save(
        REPORT / "qualitative_selection.json",
        {
            "method": "Highest GT/statistic per criterion, ties smallest image_id; smallest positive GT count for sparse; overlap is only a review proxy",
            "samples": selection,
        },
    )
    print(
        f"Created three compact figures and {len(selection)} calibration-only review panels"
    )


if __name__ == "__main__":
    main()
