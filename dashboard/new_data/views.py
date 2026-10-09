"""Faculty presentation, session-local uploads and Plotly diagnostics."""

import io
import json
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import ImageDraw

from dashboard import pages
from dashboard.new_data import ingest, runner
from dashboard.services import CHECKPOINT, ROOT

NAV = [
    "Executive Overview",
    "New Data Evaluation",
    "Deep Error Analysis",
    "Prediction Explorer",
    "Performance and Latency",
    "Historical Model Comparison",
    "Dataset and Methodology",
    "Reproducibility and Limitations",
]
WARNING = "Accuracy metrics require independent ground-truth annotations. Unlabeled-data results describe model predictions, not correctness."


def chart(rows, x, y, color=None):
    if rows:
        st.plotly_chart(
            px.bar(pd.DataFrame(rows), x=x, y=y, color=color), width="stretch"
        )


def table(rows, name):
    frame = pd.DataFrame(rows)
    st.dataframe(frame, width="stretch", hide_index=True)
    st.download_button(
        "Download " + name, frame.to_csv(index=False), name + ".csv", key=name
    )


def overview(result):
    st.title("New Data Model Evaluation and Error Analysis")
    st.caption("Verified selection · YOLOv8s E1 epoch 22 · 640 · 14 classes")
    if not result:
        st.info(
            "Unavailable · No new dataset evaluated in this session. Saved historical evidence follows; it is not a new-data result."
        )
        pages.overview()
        return
    if result.get("import_note"):
        st.warning(result["import_note"])
    st.success(
        f"{result['status']} · {result['metadata']['name']} · {result['metadata']['format']}"
    )
    st.write(
        {
            "images": len(result["records"]),
            "date": result["created_utc"],
            "device": result["config"]["device"],
            "image_size": 640,
            "checkpoint": "SHA-256 verified before inference",
            "independence": result["audit"]["independence"],
        }
    )
    if result["metadata"]["format"] == "unlabeled":
        st.warning(WARNING)
        metrics = result["metrics"]
    else:
        metrics = result["metrics"]["overall"]
    wanted = {
        "precision",
        "recall",
        "harmonic_aggregate_f1",
        "macro_class_f1",
        "map50",
        "map50_95",
        "images",
        "objects",
        "predictions_fixed",
        "predictions",
        "average_detections",
        "mean_confidence",
        "class_coverage",
    }
    scalar = {
        k: v
        for k, v in metrics.items()
        if k in wanted and (isinstance(v, (int, float)) or v is None)
    }
    for start in range(0, len(scalar), 4):
        for col, (k, v) in zip(st.columns(4), list(scalar.items())[start : start + 4]):
            col.metric(
                k,
                "Unavailable"
                if v is None
                else f"{v:.4f}"
                if isinstance(v, float)
                else v,
            )
    represented = sorted(
        {
            ingest.NAMES[c]
            for r in result["records"]
            for c, s in zip(r["classes"], r["scores"])
            if s >= 0.34
        }
    )
    st.write("Predicted classes represented", represented)
    if result["metadata"]["format"] == "unlabeled":
        chart(
            [{"class": k, "count": v} for k, v in metrics["class_counts"].items()],
            "class",
            "count",
        )
        table(
            [
                {"name": r["name"], "detections": sum(s >= 0.34 for s in r["scores"])}
                for r in result["records"]
            ],
            "per_image_predictions",
        )
    st.metric(
        "Mean pipeline latency ms", f"{result['latency']['end_to_end_ms']['mean']:.2f}"
    )
    st.metric(
        "Still images per second", f"{result['latency']['still_images_per_second']:.2f}"
    )
    st.write(
        f"Processed {len(result['records'])} images. Found {len(result['audit']['duplicates'])} duplicate pairs and {len(result['audit']['overlap'])} overlaps against the supplied index. {result['audit']['overlap_check']}."
    )


def upload(mode):
    st.title("New Data Evaluation")
    saved = st.file_uploader(
        "Load completed evidence result.json without weights", type=["json"]
    )
    if saved and st.button("Load saved evidence"):
        try:
            st.session_state["new_result"] = runner.load_saved(saved.getvalue())
            st.session_state.pop("new_images", None)
            st.success("Saved metrics reconstructed; original images are unavailable.")
        except (ValueError, TypeError, KeyError, OverflowError):
            st.error("Invalid saved evidence. No imported result was accepted.")
    if mode != "Live Inference Mode":
        st.info(
            "Saved evidence only. Switch to Live Inference Mode to supply your own permitted data."
        )
        return
    if not (ROOT / CHECKPOINT).is_file():
        st.warning("Checkpoint absent; saved evidence remains available.")
    fmt = st.selectbox("Annotation format", ["unlabeled", "yolo", "coco"])
    st.caption(
        "ZIP: images + matching TXT labels; COCO: images + annotations.json. Empty YOLO files explicitly mean no objects."
    )
    uploads = st.file_uploader(
        "ZIP or individual JPG/PNG and annotations.json",
        type=["zip", "jpg", "jpeg", "png", "json", "txt"],
        accept_multiple_files=True,
    )
    metadata_file = st.file_uploader(
        "Optional metadata JSON or YAML", type=["json", "yaml", "yml"]
    )
    safe_index = st.file_uploader(
        "Optional safe known-image SHA-256 index JSON",
        type=["json"],
        help="List of hashes from a previously approved non-reserved index. No local dataset scanning occurs.",
    )
    name = st.text_input("Dataset name")
    source = st.text_input("Dataset source")
    permission = st.text_input("Licence or permission")
    independent = st.selectbox(
        "Annotations created independently?", ["Choose", "Yes", "No"]
    )
    used = st.selectbox("Used in training or tuning?", ["Choose", "Yes", "No"])
    confirmed = st.checkbox(
        "I confirm the source, permission, format and exact class mapping shown below"
    )
    st.caption(", ".join(f"{i}: {n}" for i, n in enumerate(ingest.NAMES)))
    device = st.selectbox("Device", ["cpu", "mps"])
    if st.button("Validate and evaluate", disabled=not uploads):
        try:
            if not confirmed or independent == "Choose" or used == "Choose" or not name:
                raise ValueError("Complete all declarations.")
            files = {}
            for f in uploads:
                chunk = (
                    ingest.unpack(f.getvalue())
                    if f.name.lower().endswith(".zip")
                    else {f.name: f.getvalue()}
                )
                if set(files) & set(chunk):
                    raise ValueError("Duplicate filename across uploads.")
                files.update(chunk)
            meta = {}
            if metadata_file:
                import yaml

                if metadata_file.size > 1024**2:
                    raise ValueError("Metadata exceeds 1 MB.")
                meta = yaml.safe_load(metadata_file.getvalue())
                if not isinstance(meta, dict):
                    raise ValueError("Metadata must be a mapping.")
            meta.update(
                name=name,
                source=source,
                permission=permission,
                format=fmt,
                independent_annotations=independent == "Yes",
                used_for_training_or_tuning=used == "Yes",
            )
            if "names" not in meta:
                meta["names"] = ingest.NAMES
            known = None
            if safe_index:
                if safe_index.size > 10 * 1024**2:
                    raise ValueError("Index exceeds 10 MB.")
                known = json.loads(safe_index.getvalue())
                if not isinstance(known, list) or not all(
                    isinstance(v, str) and len(v) == 64 for v in known
                ):
                    raise ValueError("Index must be a list of SHA-256 strings.")
                known = set(known)
            images, records, audit = ingest.ingest(files, fmt, meta, known)
            st.write(audit)
            with st.spinner("Evaluating session-local data"):
                result = runner.run(images, records, audit, meta, device)
            st.session_state["new_result"] = result
            st.session_state["new_images"] = images
            st.success(
                "Complete. Open Executive Overview, Deep Error Analysis or Prediction Explorer."
            )
        except Exception as exc:  # noqa: BLE001 - UI boundary records failure without traceback
            st.session_state.pop("new_result", None)
            st.session_state.pop("new_images", None)
            failure = {
                "status": "FAILED",
                "run_id": "failed_upload",
                "reason": type(exc).__name__,
            }
            st.error(
                str(exc)
                if isinstance(exc, ValueError)
                else "Evaluation failed safely. Check formats, checkpoint and device; no completed result was published."
            )
            st.download_button(
                "Download failure record", json.dumps(failure), "FAILED.json"
            )
    result = st.session_state.get("new_result")
    if result:
        st.download_button(
            "Download immutable evidence bundle",
            runner.bundle(result),
            result["run_id"] + ".zip",
        )
    if st.button("Clear session uploads and results"):
        st.session_state.pop("new_result", None)
        st.session_state.pop("new_images", None)
        st.rerun()


def errors(result):
    st.title("Deep Error Analysis")
    if not result:
        st.info("Unavailable · Evaluate labeled new data first.")
        return
    if result["metadata"]["format"] == "unlabeled":
        st.warning(WARNING)
        return
    m = result["metrics"]
    tabs = st.tabs(["Overall", "Class-wise", "Errors", "Stratification", "Reliability"])
    with tabs[0]:
        st.plotly_chart(
            px.line(
                pd.DataFrame(m["pr"]),
                x="recall",
                y="precision",
                title="COCO interpolated PR at IoU 0.50",
            ),
            width="stretch",
        )
        st.plotly_chart(
            px.line(
                pd.DataFrame(m["curves"]),
                x="confidence",
                y=["precision", "recall", "f1"],
            ),
            width="stretch",
        )
        chart(m["ap_by_iou"], "iou", "ap")
        import numpy as np

        a = np.asarray(m["confusion"])
        names = ingest.NAMES + ["background"]
        st.plotly_chart(
            px.imshow(
                a,
                x=names,
                y=names,
                labels={"x": "Predicted", "y": "Ground truth"},
                title="Confusion counts",
            ),
            width="stretch",
        )
        norm = np.divide(
            a,
            a.sum(axis=1, keepdims=True),
            out=np.zeros_like(a, dtype=float),
            where=a.sum(axis=1, keepdims=True) > 0,
        )
        st.plotly_chart(
            px.imshow(norm, x=names, y=names, title="Row-normalized confusion"),
            width="stretch",
        )
    with tabs[1]:
        table(m["per_class"], "per_class")
        chart(m["per_class"], "name", "ap50_95")
        chart(m["per_class"], "name", "ap50")
        stable = [c for c in m["per_class"] if c["gt_count"] >= 20]
        st.caption(
            "Experimental support rule: fewer than 20 GT objects is insufficient for confident ranking."
        )
        for key in ["ap50_95", "recall", "precision"]:
            st.write(
                "Lowest " + key,
                [
                    {"name": c["name"], "support": c["gt_count"], "value": c[key]}
                    for c in sorted(
                        stable, key=lambda c: c[key] if c[key] is not None else 1
                    )[:3]
                ],
            )
    with tabs[2]:
        table(m["errors"], "error_records")
        total = max(1, sum(m["error_counts"].values()))
        table(
            [
                {"error": k, "count": v, "percent": 100 * v / total}
                for k, v in m["error_counts"].items()
            ],
            "error_counts",
        )
        if m["errors"]:
            st.dataframe(
                pd.crosstab(
                    pd.DataFrame(m["errors"]).class_id, pd.DataFrame(m["errors"]).kind
                )
            )
        matrix = m["confusion"]
        pairs = sorted(
            [
                {
                    "ground_truth": ingest.NAMES[i],
                    "prediction": ingest.NAMES[j],
                    "count": matrix[i][j],
                }
                for i in range(14)
                for j in range(14)
                if i != j and matrix[i][j]
            ],
            key=lambda x: -x["count"],
        )
        st.write("Most common class confusions", pairs[:10])
        st.caption(
            "Taxonomy events can overlap across prediction/GT records. Review candidates are not confirmed annotation errors. Fixed matching and confusion matching differ; see methodology."
        )
    with tabs[3]:
        frame = pd.DataFrame(m["strata"])
        for key in [
            "size",
            "density",
            "aspect_ratio",
            "position",
            "class_id",
            "class_frequency",
        ]:
            if not frame.empty:
                st.dataframe(
                    frame.groupby(key)
                    .matched.agg(["count", "mean"])
                    .rename(columns={"count": "GT support", "mean": "matched recall"})
                )
        st.caption(
            "Strata report GT recall, not AP. Size is original-pixel area. Occlusion, weather and truncation unavailable without explicit metadata."
        )
        bands = pd.DataFrame(
            [
                {"confidence": e["confidence"], "kind": e["kind"]}
                for e in m["errors"]
                if e["confidence"] is not None
            ]
        )
        if not bands.empty:
            st.plotly_chart(
                px.histogram(bands, x="confidence", color="kind", nbins=10),
                width="stretch",
            )
    with tabs[4]:
        st.json(m["bootstrap"])
        st.warning(
            "Rare classes and correlated images make intervals unstable. No significance test is claimed."
        )


def explorer(result):
    st.title("Prediction Explorer")
    if not result:
        st.info("Unavailable · No session predictions.")
        return
    images = st.session_state.get("new_images", {})
    labeled = result["metadata"]["format"] != "unlabeled"
    cls = st.selectbox("Class", ["All"] + ingest.NAMES)
    confidence = st.slider("Confidence range", 0.0, 1.0, (0.34, 1.0))
    kind = (
        st.selectbox(
            "Error filter", ["All"] + sorted(result["metrics"]["error_counts"])
        )
        if labeled
        else "All"
    )
    iou = (
        st.slider("Matched IoU range", 0.0, 1.0, (0.0, 1.0)) if labeled else (0.0, 1.0)
    )
    size = st.selectbox("Object size", ["All", "small", "medium", "large"])
    density = st.selectbox("Scene density", ["All", "sparse", "moderate", "dense"])
    sort = (
        st.selectbox("Image ordering", ["Filename", "Best fixed F1", "Worst fixed F1"])
        if labeled
        else "Filename"
    )
    records = result["records"]
    errors = result["metrics"].get("errors", []) if labeled else []
    if labeled and sort != "Filename":
        scores = {r["image_id"]: r["f1"] for r in result["metrics"]["per_image"]}
        records = sorted(
            records,
            key=lambda r: scores[r["image_id"]],
            reverse=sort.startswith("Best"),
        )
    choices = []
    for r in records:
        n = len(r["gt_boxes"]) if labeled else sum(s >= 0.34 for s in r["scores"])
        d = "sparse" if n <= 5 else "moderate" if n <= 15 else "dense"
        if density != "All" and d != density:
            continue
        if labeled:
            matching = [
                e
                for e in errors
                if e["image_id"] == r["image_id"]
                and (kind == "All" or e["kind"] == kind)
                and (cls == "All" or e["class_id"] == ingest.NAMES.index(cls))
                and (
                    e["confidence"] is None
                    or confidence[0] <= e["confidence"] <= confidence[1]
                )
                and (e["iou"] is None or iou[0] <= e["iou"] <= iou[1])
            ]
            boxes = [
                r["gt_boxes"][e["gt"]]
                if e["prediction"] is None
                else r["boxes"][e["prediction"]]
                for e in matching
            ]
        else:
            boxes = [
                b
                for b, c, s in zip(r["boxes"], r["classes"], r["scores"])
                if confidence[0] <= s <= confidence[1]
                and (cls == "All" or c == ingest.NAMES.index(cls))
            ]
        if size != "All":
            boxes = [
                b
                for b in boxes
                if (
                    "small"
                    if (b[2] - b[0]) * (b[3] - b[1]) < 1024
                    else "medium"
                    if (b[2] - b[0]) * (b[3] - b[1]) < 9216
                    else "large"
                )
                == size
            ]
        if boxes or (
            kind == "All" and cls == "All" and size == "All" and not r["boxes"]
        ):
            choices.append(r)
    if not choices:
        st.info("No images match these filters.")
        return
    selected = st.selectbox("Image", [r["name"] for r in choices])
    r = next(r for r in choices if r["name"] == selected)
    mode = st.radio(
        "Visual mode",
        ["Ground truth only", "Predictions only", "Overlay comparison"]
        if labeled
        else ["Predictions only"],
        horizontal=True,
    )
    if selected in images:
        image = images[selected].copy()
        draw = ImageDraw.Draw(image)
        ee = [e for e in errors if e["image_id"] == r["image_id"]]
        by_pred = {e["prediction"]: e for e in ee if e["prediction"] is not None}
        if mode != "Predictions only":
            missed = {e["gt"] for e in ee if e["kind"] == "missed object"}
            for gi, (b, c) in enumerate(zip(r["gt_boxes"], r["gt_classes"])):
                draw.rectangle(b, outline="red" if gi in missed else "blue", width=3)
                draw.text((b[0], b[1]), "GT " + ingest.NAMES[c], fill="blue")
        if mode != "Ground truth only":
            for pi, (b, c, s) in enumerate(zip(r["boxes"], r["classes"], r["scores"])):
                if not confidence[0] <= s <= confidence[1] or (
                    cls != "All" and c != ingest.NAMES.index(cls)
                ):
                    continue
                e = by_pred.get(pi, {})
                color = "green" if e.get("kind") == "correct detection" else "orange"
                draw.rectangle(b, outline=color, width=3)
                draw.text((b[0], b[1]), f"{ingest.NAMES[c]} {s:.2f}", fill=color)
        st.image(image, width="stretch")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        st.download_button("Download annotated image", buf.getvalue(), "annotated.png")
    if labeled:
        table(
            [e for e in errors if e["image_id"] == r["image_id"]],
            "selected_image_errors",
        )
        st.write(
            next(
                p
                for p in result["metrics"]["per_image"]
                if p["image_id"] == r["image_id"]
            )
        )
    else:
        rows = [
            {
                "class_name": ingest.NAMES[c],
                "confidence": s,
                "area": (b[2] - b[0]) * (b[3] - b[1]),
            }
            for b, c, s in zip(r["boxes"], r["classes"], r["scores"])
            if s >= 0.34
        ]
        table(rows, "predictions")
        if rows:
            st.plotly_chart(
                px.histogram(pd.DataFrame(rows), x="confidence"), width="stretch"
            )
            st.plotly_chart(px.histogram(pd.DataFrame(rows), x="area"), width="stretch")
    table(
        [
            {"name": r["name"], "width": r["width"], "height": r["height"]}
            for r in choices
        ],
        "filtered_images",
    )


def performance(result):
    st.title("Performance and Latency")
    if not result:
        st.info("Unavailable · Evaluate new data to measure latency.")
        return
    st.json(dict(model_load_ms=result["model_load_ms"], **result["latency"]))
    st.caption(result["config"]["timing"])
    st.write(
        {
            k: result["config"][k]
            for k in ["device", "warmup", "batch", "imgsz", "precision"]
        }
    )
    table(result["timing"], "latency")
    st.plotly_chart(
        px.histogram(pd.DataFrame(result["timing"]), x="end_to_end_ms"), width="stretch"
    )
    st.plotly_chart(
        px.scatter(
            pd.DataFrame(result["timing"]), x="width", y="end_to_end_ms", color="height"
        ),
        width="stretch",
    )
    st.warning(
        "Still-image pipeline throughput includes decode but excludes upload transfer and display; it is not live-video FPS."
    )


def methodology(result):
    st.title("Dataset and Methodology")
    if result:
        st.json(result["audit"])
        st.json(result["metadata"])
        st.json(result["config"])
        records = result["records"]
        table(
            [{k: r[k] for k in ["name", "width", "height", "sha256"]} for r in records],
            "manifest",
        )
        if result["metadata"]["format"] != "unlabeled":
            chart(
                [
                    {"name": ingest.NAMES[k], "objects": v}
                    for k, v in Counter(
                        c for r in records for c in r["gt_classes"]
                    ).items()
                ],
                "name",
                "objects",
            )
            chart(
                [{"name": r["name"], "objects": len(r["gt_boxes"])} for r in records],
                "name",
                "objects",
            )
    st.write(
        "Missing labels and invalid boxes are rejected before inference. Duplicate findings are retained. Annotation incompleteness can inflate apparent false positives. Few examples make rare-class AP unstable."
    )
    with st.expander("Historical dataset and training evidence"):
        pages.dataset()
        pages.training()


def render(selected, mode):
    result = st.session_state.get("new_result")
    st.caption(
        "DATASET STATUS: "
        + (
            result["metadata"]["name"] + " / COMPLETE"
            if result
            else "No new evaluation · saved historical fallback"
        )
    )
    if mode == "Faculty Presentation Mode":
        st.info(
            "Guided sequence: Overview → Deep Error Analysis → Prediction Explorer. Only completed saved session results and verified historical evidence are shown; no model loading."
        )
    if selected == NAV[0]:
        overview(result)
    elif selected == NAV[1]:
        upload(mode)
    elif selected == NAV[2]:
        errors(result)
    elif selected == NAV[3]:
        explorer(result)
    elif selected == NAV[4]:
        performance(result)
    elif selected == NAV[5]:
        pages.comparison()
        st.warning(
            "Historical validation2000 and matched calibration500 are distinct pools. No automatic new-vs-historical delta: historical YOLO uses a different evaluator; composition also affects scores."
        )
        if result and result["metadata"]["format"] != "unlabeled":
            st.subheader("New evaluation dataset — separate pool")
            st.json(result["metrics"]["overall"])
            config = result["config"]
            eligible = (
                config["class_names"] == ingest.NAMES
                and config["confidence"] == 0.34
                and config["nms_iou"] == 0.7
                and config["max_det"] == 300
                and config["ap_score_floor"] == 0.001
                and config["metric_protocol"]
                == "common_metrics COCO v1; macro over GT-present classes"
                and all(c["gt_count"] > 0 for c in result["metrics"]["per_class"])
            )
            reference = pages.load(
                "reports/comparisons/E1_E3_stageE_v2/E1_metrics.json"
            )
            if eligible and reference:
                rows = []
                for k in [
                    "precision",
                    "recall",
                    "harmonic_aggregate_f1",
                    "macro_class_f1",
                    "map50",
                    "map50_95",
                ]:
                    a = result["metrics"]["overall"][k]
                    b = reference[k]
                    rows.append(
                        {
                            "metric": k,
                            "new dataset": a,
                            "matched calibration500 historical": b,
                            "absolute difference": a - b,
                            "percentage points": 100 * (a - b),
                        }
                    )
                table(rows, "compatible_calibration_comparison")
                st.warning(
                    "Settings and metric definitions match calibration500. These are different image pools; composition and difficulty can explain differences. No significance or independent-test claim."
                )

    elif selected == NAV[6]:
        methodology(result)
    else:
        pages.documentation()
        st.markdown(
            "See `docs/reproducibility/NEW_DATA_EVALUATION.md`. No training, reserved evaluation or live-video claims. Uploaded data is session-local; evidence downloads contain predictions and manifests but no source images. No cross-user upload caching."
        )
