"""One filter contract for the gallery, overlays and CSV downloads."""

from PIL import ImageDraw

from dashboard.new_data.ingest import NAMES


def size_of(box):
    area = (box[2] - box[0]) * (box[3] - box[1])
    return "small" if area < 1024 else "medium" if area < 9216 else "large"


def select_rows(
    record,
    errors=None,
    cls="All",
    confidence=(0.34, 1.0),
    kind="All",
    iou=(0.0, 1.0),
    size="All",
):
    if errors is None:
        candidates = [
            {
                "prediction": i,
                "gt": None,
                "class_id": c,
                "confidence": s,
                "iou": None,
                "kind": "prediction",
            }
            for i, (c, s) in enumerate(zip(record["classes"], record["scores"]))
        ]
    else:
        candidates = [e for e in errors if e["image_id"] == record["image_id"]]
    rows = []
    for e in candidates:
        if cls != "All" and NAMES[e["class_id"]] != cls:
            continue
        if kind != "All" and e["kind"] != kind:
            continue
        if (
            e["confidence"] is not None
            and not confidence[0] <= e["confidence"] <= confidence[1]
        ):
            continue
        if e["iou"] is not None and not iou[0] <= e["iou"] <= iou[1]:
            continue
        b = (
            record["gt_boxes"][e["gt"]]
            if e["prediction"] is None
            else record["boxes"][e["prediction"]]
        )
        if size != "All" and size_of(b) != size:
            continue
        rows.append(
            {**e, "class_name": NAMES[e["class_id"]], "box": b, "size": size_of(b)}
        )
    return rows


def overlay(image, record, rows, mode, labeled):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    if labeled and mode != "Predictions only":
        indices = {r["gt"] for r in rows if r["gt"] is not None}
        missed = {r["gt"] for r in rows if r["kind"] == "missed object"}
        for gi in sorted(indices):
            box = record["gt_boxes"][gi]
            color = "#a62d48" if gi in missed else "#2463ab"
            draw.rectangle(box, outline=color, width=3)
            draw.text(
                (box[0], box[1]),
                ("Missed: " if gi in missed else "GT: ")
                + NAMES[record["gt_classes"][gi]],
                fill=color,
                stroke_width=1,
                stroke_fill="white",
            )
    if mode != "Ground truth only":
        for row in rows:
            if row["prediction"] is None:
                continue
            color = (
                "#13816c"
                if row["kind"] == "correct detection"
                else "#c57414"
                if labeled
                else "#2463ab"
            )
            b = row["box"]
            draw.rectangle(b, outline=color, width=3)
            label = f"{row['class_name']} {row['confidence']:.2f}"
            if labeled and row["iou"] is not None:
                label += f" / IoU {row['iou']:.2f}"
            draw.text(
                (b[0], b[1]), label, fill=color, stroke_width=1, stroke_fill="white"
            )
    return image
