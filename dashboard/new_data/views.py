"""Faculty presentation, session-local uploads and Plotly diagnostics."""

import io
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import pages
from dashboard import presentation as ui
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
    display = frame.rename(
        columns=lambda name: str(name).replace("_", " ").capitalize()
    )
    st.dataframe(display, width="stretch", hide_index=True)
    st.download_button(
        "Download " + name.replace("_", " "),
        frame.to_csv(index=False),
        name + ".csv",
        key=name,
    )


def overview(result):
    ui.heading(
        "Vehicle detection, backed by evidence",
        "Explore the selected detector, evaluate new road images and explain where predictions succeed or fail.",
    )
    if not result:
        cols = st.columns([2, 1])
        with cols[0], st.container(border=True):
            st.subheader("Ready for your next dataset")
            st.write(
                "YOLOv8s is the selected project detector. Start with images alone to inspect predictions, or include annotations to measure detection quality."
            )
            c1, c2 = st.columns(2)
            with c1:
                ui.action(
                    "Evaluate new images",
                    NAV[1],
                    "Live Inference Mode",
                    primary=True,
                )
            with c2:
                ui.action("Explore model evidence", NAV[5])
        with cols[1], st.container(border=True):
            st.subheader("Faculty walkthrough")
            st.write(
                "Project summary → historical results → methodology. For new data: metrics → errors → examples."
            )
            ui.action("Start presentation", NAV[5], "Faculty Presentation Mode")
        st.subheader("Selected model · historical validation")
        st.caption(
            "Saved results on 2,000 validation images. These are not results on newly uploaded data."
        )
        m = pages.load(pages.EVAL + pages.HIST[1][1] + "/metrics.json")
        if m:
            ui.cards(
                [
                    (
                        "Precision",
                        ui.percent(m["precision"]),
                        "Macro precision at the historical operating point.",
                    ),
                    ("Recall", ui.percent(m["recall"])),
                    ("AP at IoU 0.50", ui.percent(m["map50"])),
                    ("AP at IoU 0.50–0.95", ui.percent(m["map50_95"])),
                ]
            )
        else:
            st.info(
                "Historical metrics are unavailable in this installation. You can still prepare or load a new evaluation."
            )
        st.subheader("Project at a glance")
        ui.cards(
            [
                ("Training images", "8,000"),
                ("Historical validation", "2,000"),
                ("Vehicle classes", "14"),
                ("Selected checkpoint", "Epoch 22"),
            ]
        )
        with st.expander("What is complete, and what remains?"):
            st.write(
                "Detector training and saved model comparisons are complete. New-data evaluation is available. Video tracking and counting accuracy are outside this dashboard. The selected subset has been verified; unselected full-dataset image acquisition is incomplete."
            )
            st.write(
                "Model confidence measures prediction strength, not correctness. Accuracy metrics require labeled data."
            )
        return
    if result.get("import_note"):
        st.info(result["import_note"])
    labeled = result["metadata"]["format"] != "unlabeled"
    predictions = sum(
        sum(score >= 0.34 for score in r["scores"]) for r in result["records"]
    )
    ui.cards(
        [
            ("Images processed", f"{len(result['records']):,}"),
            (
                "Ground-truth objects",
                f"{sum(len(r['gt_boxes']) for r in result['records']):,}"
                if labeled
                else "Not supplied",
            ),
            ("Detections ≥ 0.34", f"{predictions:,}"),
            ("Evaluation type", "Labeled" if labeled else "Predictions only"),
        ]
    )
    st.subheader("Detection quality" if labeled else "Prediction summary")
    if labeled:
        m = result["metrics"]["overall"]
        ui.cards(
            [
                ("Precision", ui.percent(m["precision"])),
                ("Recall", ui.percent(m["recall"])),
                ("Harmonic F1", ui.percent(m["harmonic_aggregate_f1"])),
                ("Macro class F1", ui.percent(m["macro_class_f1"])),
            ]
        )
        ui.cards(
            [
                ("AP at IoU 0.50", ui.percent(m["map50"])),
                ("AP at IoU 0.50–0.95", ui.percent(m["map50_95"])),
                (
                    "Mean latency",
                    f"{result['latency']['end_to_end_ms']['mean']:.1f} ms",
                ),
                (
                    "Still-image throughput",
                    f"{result['latency']['still_images_per_second']:.2f} img/s",
                ),
            ]
        )
        st.caption(
            "Precision and recall are macro averages across classes with ground truth. F1 uses confidence 0.34 and matching IoU 0.50; AP integrates across confidence levels."
        )
    else:
        st.info(WARNING)
        m = result["metrics"]
        ui.cards(
            [
                ("Detections per image", f"{m['average_detections']:.2f}"),
                (
                    "Mean confidence",
                    ui.percent(m["mean_confidence"]),
                    "Prediction confidence is not accuracy.",
                ),
                ("Classes detected", f"{m['class_coverage']} / 14"),
                (
                    "Mean latency",
                    f"{result['latency']['end_to_end_ms']['mean']:.1f} ms",
                ),
            ]
        )
        chart(
            [
                {"Vehicle class": k, "Detections": v}
                for k, v in m["class_counts"].items()
            ],
            "Vehicle class",
            "Detections",
        )
    audit = result["audit"]
    if audit["overlap"]:
        st.warning(
            f"{len(audit['overlap'])} images overlap the supplied index. Do not describe this as independent evaluation."
        )
    if audit["duplicates"]:
        st.warning(
            f"{len(audit['duplicates'])} duplicate image pairs found; repeated scenes can bias the summary."
        )
    st.caption(
        f"{audit['overlap_check']}. Independence status: {audit['independence']}."
    )
    st.subheader("Continue the review")
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.action("Inspect predictions", NAV[3], key="overview_explore", primary=True)
    with c2:
        ui.action(
            "Explain model errors" if labeled else "Review timing",
            NAV[2] if labeled else NAV[4],
            key="overview_analyze",
        )
    with c3:
        st.download_button(
            "Download evaluation bundle",
            runner.bundle(result),
            result["run_id"] + ".zip",
            width="stretch",
        )
    with st.expander("Evaluation record"):
        st.write(
            {
                "Dataset": result["metadata"]["name"],
                "Date UTC": result["created_utc"],
                "Device": result["config"]["device"],
                "Input size": 640,
                "Run": result["run_id"],
            }
        )


def upload(mode):
    ui.heading(
        "Evaluate new road images",
        "A guided workflow: supply data, review validation findings, then run the selected model.",
    )
    with st.expander(
        "Reopen a completed evaluation", expanded=mode != "Live Inference Mode"
    ):
        st.write(
            "Upload result.json from a previously exported bundle. Metrics are recomputed without loading weights; source images are not included."
        )
        saved = st.file_uploader(
            "Saved result.json",
            type=["json"],
            key=f"saved_result_upload_{st.session_state.get('upload_generation', 0)}",
        )
        if st.button("Load saved evidence", disabled=not saved):
            try:
                imported = runner.load_saved(saved.getvalue())
                st.session_state["new_result"] = imported
                st.session_state.pop("new_images", None)
                st.success("Saved result loaded. Open the overview to present it.")
            except ImportError:
                st.error(
                    "Recomputing a saved labeled evaluation needs the evaluation dependencies. Install requirements-new-data.txt in your project environment; historical results remain available without them."
                )
            except (ValueError, TypeError, KeyError, OverflowError, IndexError):
                st.error(
                    "This file is not a valid completed evaluation. Your previous result is unchanged."
                )
        ui.action("Open evaluation overview", NAV[0], key="import_overview")
    if mode != "Live Inference Mode":
        st.info(
            "You are browsing saved evidence. Switch workspaces to evaluate a new dataset."
        )
        ui.action(
            "Switch to new-data evaluation",
            NAV[1],
            "Live Inference Mode",
            key="switch_evaluate",
            primary=True,
        )
        return
    ready = (ROOT / CHECKPOINT).is_file()
    if not ready:
        st.warning(
            "Checkpoint absent. You can validate uploads, but model execution needs the verified E1 weights. Saved evidence remains available."
        )
    generation = st.session_state.get("upload_generation", 0)
    st.subheader("1 · Choose data and tell us where it came from")
    c1, c2 = st.columns([2, 1])
    with c1:
        fmt = st.selectbox(
            "Annotation format",
            ["unlabeled", "yolo", "coco"],
            format_func=lambda f: {
                "unlabeled": "Images only — inspect predictions",
                "yolo": "YOLO labels — measure detection quality",
                "coco": "COCO JSON — measure detection quality",
            }[f],
        )
        instructions = {
            "unlabeled": "Upload JPG/PNG images, individually or as a ZIP. This workflow cannot measure accuracy.",
            "yolo": "ZIP images with one matching .txt label file per image. An empty .txt explicitly means no vehicles.",
            "coco": "Upload images plus annotations.json, individually or together in a ZIP. Category names must match UVH-26.",
        }
        st.info(instructions[fmt])
        uploads = st.file_uploader(
            "Images and annotations",
            type=["zip", "jpg", "jpeg", "png", "json", "txt"],
            accept_multiple_files=True,
            key=f"dataset_upload_{generation}",
        )
        st.caption(
            "Up to 100 MB total · 500 images · 100 megapixels decoded · JPG/PNG · unique base filenames"
        )
    with c2:
        st.markdown("**Required details**")
        name = st.text_input(
            "Dataset name", placeholder="For example: Campus road collection"
        )
        source = st.text_input(
            "Dataset source", placeholder="Describe who collected or supplied it"
        )
        permission = st.text_input(
            "Licence or permission",
            placeholder="For example: Photographs taken by our group",
        )
    independent = (
        st.selectbox("Annotations created independently?", ["Choose", "Yes", "No"])
        if fmt != "unlabeled"
        else "No"
    )
    used = st.selectbox("Used in training or tuning?", ["Choose", "Yes", "No"])
    if fmt == "unlabeled":
        st.caption(
            "Annotation independence does not apply because no annotations are supplied."
        )
    with st.expander("Class mapping and optional metadata"):
        st.dataframe(
            pd.DataFrame({"YOLO ID": range(14), "Vehicle class": ingest.NAMES}),
            hide_index=True,
            width="stretch",
        )
        metadata_file = st.file_uploader(
            "Optional metadata JSON or YAML",
            type=["json", "yaml", "yml"],
            key=f"metadata_{generation}",
        )
        safe_index = st.file_uploader(
            "Optional known-image SHA-256 index JSON",
            type=["json"],
            key=f"index_{generation}",
            help="Use an approved non-reserved index only. Without this, overlap verification is unavailable.",
        )
    confirmed = st.checkbox(
        "I confirm the source, permission, annotation format and class mapping"
    )
    device = st.selectbox(
        "Compute device",
        ["cpu", "mps"],
        format_func=lambda d: (
            "CPU — works on all supported machines"
            if d == "cpu"
            else "Apple MPS — CPU fallback if unavailable"
        ),
    )
    import hashlib

    fingerprint = hashlib.sha256(
        json.dumps(
            [fmt, name, source, permission, independent, used, confirmed, device],
            sort_keys=True,
        ).encode()
    )
    for f in list(uploads or []) + [v for v in [metadata_file, safe_index] if v]:
        fingerprint.update(f.name.encode())
        fingerprint.update(f.getvalue())
    signature = fingerprint.hexdigest()
    prepared = st.session_state.get("prepared_data")
    if prepared and prepared["signature"] != signature:
        st.session_state.pop("prepared_data", None)
        prepared = None
    can_validate = bool(
        uploads
        and name.strip()
        and source.strip()
        and permission.strip()
        and confirmed
        and independent != "Choose"
        and used != "Choose"
    )
    if not can_validate:
        st.caption(
            "Complete the required details and confirmation to enable validation."
        )
    if st.button("Validate dataset", disabled=not can_validate, type="primary"):
        try:
            files = {}
            for f in uploads:
                chunk = (
                    ingest.unpack(f.getvalue())
                    if f.name.lower().endswith(".zip")
                    else {f.name: f.getvalue()}
                )
                if set(files) & set(chunk):
                    raise ValueError(
                        "Duplicate filenames across uploads. Rename the files before retrying."
                    )
                files.update(chunk)
            meta = {}
            if metadata_file:
                import yaml

                if metadata_file.size > 1024**2:
                    raise ValueError("Metadata exceeds 1 MB.")
                meta = yaml.safe_load(metadata_file.getvalue())
                if not isinstance(meta, dict):
                    raise ValueError("Metadata must be an object with named fields.")
            meta.update(
                name=name.strip(),
                source=source.strip(),
                permission=permission.strip(),
                format=fmt,
                independent_annotations=independent == "Yes",
                used_for_training_or_tuning=used == "Yes",
            )
            meta.setdefault("names", ingest.NAMES)
            known = None
            if safe_index:
                import re

                if safe_index.size > 10 * 1024**2:
                    raise ValueError("Hash index exceeds 10 MB.")
                known = json.loads(safe_index.getvalue())
                if not isinstance(known, list) or not all(
                    isinstance(h, str) and re.fullmatch("[a-fA-F0-9]{64}", h)
                    for h in known
                ):
                    raise ValueError(
                        "Hash index must be a JSON list of SHA-256 strings."
                    )
                known = {h.lower() for h in known}
            with st.spinner("Checking images, labels, dimensions and hashes…"):
                images, records, audit = ingest.ingest(files, fmt, meta, known)
            prepared = {
                "signature": signature,
                "images": images,
                "records": records,
                "audit": audit,
                "metadata": meta,
            }
            st.session_state["prepared_data"] = prepared
        except Exception as exc:  # noqa: BLE001 - safe upload boundary
            st.session_state.pop("prepared_data", None)
            prepared = None
            st.error(
                f"Validation failed: {exc}"
                if isinstance(exc, (ValueError, TypeError))
                else "Could not read this dataset. Check the ZIP and annotation format. The previous completed evaluation is unchanged."
            )
    st.subheader("2 · Review validation findings")
    if prepared:
        audit = prepared["audit"]
        records = prepared["records"]
        ui.cards(
            [
                ("Images validated", str(len(records))),
                (
                    "Annotated objects",
                    str(sum(len(r["gt_boxes"]) for r in records))
                    if fmt != "unlabeled"
                    else "Not supplied",
                ),
                ("Duplicate pairs", str(len(audit["duplicates"]))),
                ("Known overlaps", str(len(audit["overlap"]))),
            ]
        )
        st.caption(audit["overlap_check"])
        if audit["duplicates"] or audit["overlap"]:
            st.warning(
                "Duplicates or known overlaps are present. Scores may not describe independent generalization."
            )
        st.write("Independence status: " + audit["independence"])
        with st.expander("Inspect validation details"):
            st.json(audit)
        st.subheader("3 · Run the frozen model")
        st.caption(
            "E1 YOLOv8s · 640 px · confidence 0.34 · batch 1. No training or tuning occurs. Processing time depends on the number and size of images."
        )
        if st.button(
            "Run evaluation" if fmt != "unlabeled" else "Run predictions",
            disabled=not ready,
            type="primary",
        ):
            progress = st.progress(0, text="Verifying model and preparing inference…")
            try:
                result = runner.run(
                    prepared["images"],
                    prepared["records"],
                    audit,
                    prepared["metadata"],
                    device,
                    progress=lambda value, text: progress.progress(value, text=text),
                )
                st.session_state["new_result"] = result
                st.session_state["new_images"] = prepared["images"]
                st.success("Completed. Your new result is ready to review or download.")
            except Exception as exc:  # noqa: BLE001 - preserve last successful result
                import uuid

                failure = {
                    "status": "FAILED",
                    "run_id": "failed_" + uuid.uuid4().hex,
                    "reason": type(exc).__name__,
                }
                st.error(
                    str(exc)
                    if isinstance(exc, ValueError)
                    else "Evaluation could not complete. Try CPU, check model dependencies, or use saved evidence. Your previous completed result is unchanged."
                )
                st.download_button(
                    "Download failure record", json.dumps(failure), "FAILED.json"
                )
            finally:
                progress.empty()
    else:
        st.info(
            "Validate your dataset first. Model inference will not start until you review the findings."
        )
    if st.session_state.get("new_result"):
        st.subheader("Completed result")
        st.caption(
            "Current result: " + st.session_state["new_result"]["metadata"]["name"]
        )
        ui.action("View results", NAV[0], key="upload_view_result", primary=True)
        r = st.session_state["new_result"]
        st.download_button(
            "Download evidence bundle", runner.bundle(r), r["run_id"] + ".zip"
        )
    if st.button("Clear uploaded data and session results"):
        for key in ["new_result", "new_images", "prepared_data", "saved_result_upload"]:
            st.session_state.pop(key, None)
        st.session_state["upload_generation"] = generation + 1
        st.rerun()


def errors(result):
    if not result or result["metadata"]["format"] == "unlabeled":
        ui.empty(
            "Understand detection errors",
            "Correctness cannot be determined without ground-truth annotations. Images-only results remain available in Prediction Explorer.",
            labeled=True,
        )
        return
    ui.heading(
        "Understand detection errors",
        "Inspect missed vehicles, class confusion and localization failures. Every class score is shown with its ground-truth support.",
    )
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
        bootstrap = m["bootstrap"]
        st.subheader("Uncertainty of the fixed-threshold metrics")
        table(
            [
                {
                    "Metric": key.replace("_", " ").title(),
                    "Lower 95% bound": ui.percent(bounds[0]),
                    "Upper 95% bound": ui.percent(bounds[1]),
                }
                for key, bounds in bootstrap["intervals"].items()
            ],
            "confidence_intervals",
        )
        st.caption(
            f"Image-level bootstrap · {bootstrap['replicates']} samples · seed {bootstrap['seed']} · AP confidence intervals are not calculated."
        )
        st.warning(
            "Rare classes and correlated images make intervals unstable. No significance test is claimed."
        )


def explorer(result):
    if not result:
        ui.empty(
            "Prediction explorer",
            "See annotated images and trace every detection back to its prediction record.",
        )
        return
    ui.heading(
        "Prediction explorer",
        "Filter images and inspect the exact detections behind each result. Downloads use the same filters as the gallery.",
    )
    from dashboard.new_data.explorer import overlay, select_rows

    labeled = result["metadata"]["format"] != "unlabeled"
    images = st.session_state.get("new_images", {})
    errors = result["metrics"].get("errors", []) if labeled else None
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            cls = st.selectbox("Vehicle class", ["All"] + ingest.NAMES)
        with c2:
            kind = (
                st.selectbox(
                    "Detection outcome",
                    ["All"] + sorted(result["metrics"]["error_counts"]),
                )
                if labeled
                else "All"
            )
        with c3:
            size = st.selectbox("Object size", ["All", "small", "medium", "large"])
        c1, c2, c3 = st.columns(3)
        with c1:
            confidence = st.slider(
                "Prediction confidence",
                0.0,
                1.0,
                (0.0 if kind == "low-confidence correct candidate" else 0.34, 1.0),
            )
        with c2:
            iou = (
                st.slider("Matched IoU", 0.0, 1.0, (0.0, 1.0))
                if labeled
                else (0.0, 1.0)
            )
        with c3:
            density = st.selectbox(
                "Scene density", ["All", "sparse", "moderate", "dense"]
            )
    st.caption(
        "Filters affect boxes, records and downloads. Missed objects have no prediction confidence or IoU, so those sliders do not exclude them. Ground truth associated with selected predictions is shown for context."
        if labeled
        else "Predictions only: confidence is not accuracy. Object size uses the original image dimensions."
    )
    records = sorted(result["records"], key=lambda r: r["name"])
    sort = (
        st.selectbox("Image ordering", ["Filename", "Best fixed F1", "Worst fixed F1"])
        if labeled
        else "Filename"
    )
    if sort != "Filename":
        scores = {r["image_id"]: r["f1"] for r in result["metrics"]["per_image"]}
        records = sorted(
            records,
            key=lambda r: scores[r["image_id"]],
            reverse=sort == "Best fixed F1",
        )
    choices = []
    allrows = []
    for record in records:
        n = (
            len(record["gt_boxes"])
            if labeled
            else sum(s >= 0.34 for s in record["scores"])
        )
        d = "sparse" if n <= 5 else "moderate" if n <= 15 else "dense"
        if density != "All" and density != d:
            continue
        rows = select_rows(record, errors, cls, confidence, kind, iou, size)
        empty_default = (
            not record["boxes"]
            and not record["gt_boxes"]
            and cls == kind == size == "All"
            and iou == (0.0, 1.0)
        )
        if rows or empty_default:
            choices.append((record, rows))
            allrows.extend({**row, "image": record["name"]} for row in rows)
    if not choices:
        st.info("No matching images. Broaden the class, outcome or confidence filters.")
        return
    st.caption(f"{len(choices)} matching images · {len(allrows)} matching records")
    name = st.selectbox("Image", [r["name"] for r, rows in choices])
    record, rows = next(v for v in choices if v[0]["name"] == name)
    visual = st.radio(
        "Visual mode",
        ["Overlay comparison", "Ground truth only", "Predictions only"]
        if labeled
        else ["Predictions only"],
        horizontal=True,
    )
    if name in images:
        image = overlay(images[name], record, rows, visual, labeled)
        st.image(image, width="stretch")
        st.caption(
            "Green: correct detection · Amber: unmatched or confused prediction · Red: missed ground truth · Blue: ground truth context"
            if labeled
            else "Blue: model prediction; no correctness assessment"
        )
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        st.download_button(
            "Download this annotated image", buf.getvalue(), "annotated.png"
        )
    else:
        st.info(
            "Source image unavailable. Imported bundles contain records, not photographs. The filtered records below remain available."
        )
    if labeled:
        m = next(
            r
            for r in result["metrics"]["per_image"]
            if r["image_id"] == record["image_id"]
        )
        ui.cards(
            [
                ("Image true positives", m["tp"]),
                ("Image false positives", m["fp"]),
                ("Image missed objects", m["fn"]),
                ("Image harmonic F1", ui.percent(m["f1"])),
            ]
        )
        st.caption(
            "Image totals above use the frozen evaluation threshold and do not change with display filters."
        )
    table(rows, "selected_image_records")
    with st.expander("Download all filtered records"):
        table(allrows, "filtered_records")


def performance(result):
    if not result:
        ui.empty(
            "Performance and latency",
            "Timing is measured when you evaluate a dataset. No device speed is assumed.",
        )
        return
    ui.heading(
        "Performance and latency",
        "Understand how long the complete still-image pipeline takes, separately from model loading.",
    )
    m = result["latency"]
    e = m["end_to_end_ms"]
    ui.cards(
        [
            ("Mean end-to-end", f"{e['mean']:.2f} ms"),
            ("Median", f"{e['median']:.2f} ms"),
            ("95th percentile", f"{e['p95']:.2f} ms"),
            ("Still-image throughput", f"{m['still_images_per_second']:.2f} img/s"),
        ]
    )
    st.caption(
        f"Device: {result['config']['device'].upper()} · Batch: {result['config']['batch']} · Input: {result['config']['imgsz']} px · Warm-up: {result['config']['warmup']} iterations · Model loading: {result['model_load_ms']:.1f} ms (excluded)"
    )
    timing = pd.DataFrame(result["timing"])
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.histogram(
                timing,
                x="end_to_end_ms",
                labels={"end_to_end_ms": "End-to-end latency (ms)"},
                title="Latency distribution",
            ),
            width="stretch",
        )
    with right:
        rows = [
            {"Stage": label, "Mean (ms)": m[key]["mean"]}
            for key, label in [
                ("preprocessing_ms", "Decode and preprocess"),
                ("inference_ms", "Inference"),
                ("postprocessing_ms", "Postprocess"),
            ]
        ]
        chart(rows, "Stage", "Mean (ms)")
    st.info(
        "These measurements describe still images, not live-video FPS. Upload transfer, browser display and export are excluded. Timing stages may not sum to the outer pipeline interval."
    )
    with st.expander("Timing protocol and detailed measurements"):
        st.write(result["config"]["timing"])
        table(result["timing"], "latency")
        table(
            [
                {"Stage": key, **value}
                for key, value in m.items()
                if isinstance(value, dict)
            ],
            "timing_summary",
        )
    st.plotly_chart(
        px.scatter(
            timing,
            x="width",
            y="end_to_end_ms",
            color="height",
            labels={
                "width": "Source width (px)",
                "height": "Source height (px)",
                "end_to_end_ms": "Latency (ms)",
            },
            title="Latency by source dimensions",
        ),
        width="stretch",
    )


def methodology(result):
    ui.heading(
        "Dataset and methodology",
        "Understand the evidence behind the results: class definitions, annotation quality, training history and evaluation rules.",
    )
    tabs = st.tabs(
        [
            "Current dataset",
            "Project dataset",
            "Training evidence",
            "Metric definitions",
        ]
    )
    with tabs[0]:
        if result:
            audit = result["audit"]
            records = result["records"]
            labeled = result["metadata"]["format"] != "unlabeled"
            ui.cards(
                [
                    ("Images", len(records)),
                    (
                        "Labeled objects",
                        sum(len(r["gt_boxes"]) for r in records)
                        if labeled
                        else "Not supplied",
                    ),
                    ("Duplicate pairs", len(audit["duplicates"])),
                    ("Known overlaps", len(audit["overlap"])),
                ]
            )
            st.write("Source: " + result["metadata"]["source"])
            st.write("Permission: " + result["metadata"]["permission"])
            st.caption(audit["overlap_check"])
            if labeled:
                chart(
                    [
                        {"Class": ingest.NAMES[k], "Objects": v}
                        for k, v in Counter(
                            c for r in records for c in r["gt_classes"]
                        ).items()
                    ],
                    "Class",
                    "Objects",
                )
                counts = [
                    {"Image": r["name"], "Objects": len(r["gt_boxes"])} for r in records
                ]
                st.plotly_chart(
                    px.histogram(
                        pd.DataFrame(counts), x="Objects", title="Objects per image"
                    ),
                    width="stretch",
                )
            with st.expander("Manifest, validation details and configuration"):
                table(
                    [
                        {k: r[k] for k in ["name", "width", "height", "sha256"]}
                        for r in records
                    ],
                    "manifest",
                )
                st.json(audit)
                st.json(result["config"])
        else:
            st.info(
                "No new dataset has been evaluated. The project dataset and training evidence remain available in the other tabs."
            )
        st.write(
            "Incomplete annotations can make correct predictions look like false positives. Duplicate images overrepresent particular scenes. Very few examples make rare-class scores unstable."
        )
    with tabs[1]:
        pages.dataset()
    with tabs[2]:
        pages.training()
    with tabs[3]:
        table(
            [
                {
                    "Metric": "Precision",
                    "Meaning": "Of the detections made, how many match ground truth? Macro-averaged over represented classes.",
                },
                {
                    "Metric": "Recall",
                    "Meaning": "Of the annotated vehicles, how many were detected? Macro-averaged over represented classes.",
                },
                {
                    "Metric": "Harmonic F1",
                    "Meaning": "Harmonic mean of aggregate precision and recall; not an average of class F1.",
                },
                {
                    "Metric": "Macro class F1",
                    "Meaning": "Average F1 across classes with ground truth.",
                },
                {
                    "Metric": "AP50",
                    "Meaning": "Area under the interpolated precision–recall curve at IoU 0.50.",
                },
                {
                    "Metric": "AP50:95",
                    "Meaning": "AP averaged across IoU 0.50, 0.55, …, 0.95; penalizes imprecise boxes more strongly.",
                },
                {
                    "Metric": "IoU",
                    "Meaning": "Intersection area divided by union area of predicted and annotated boxes.",
                },
                {
                    "Metric": "Confidence",
                    "Meaning": "Model prediction score; it is not a measurement of accuracy.",
                },
            ],
            "metric_glossary",
        )
        st.caption(
            "New-data evaluation: operating confidence0.34, matching IoU0.50, AP score floor0.001. Historical validation2000 uses a different protocol and stays separate."
        )


def documentation():
    ui.heading(
        "Reproducibility and limitations",
        "A clear record of what the system measures, what you can export and what remains outside scope.",
    )
    with st.container(border=True):
        st.subheader("Prepare → validate → evaluate → explain")
        st.write(
            "1. Choose images-only for predictions, or YOLO/COCO labels for accuracy metrics.\n\n2. Supply source and permission details, then validate the dataset.\n\n3. Review duplicates and overlap findings before running inference.\n\n4. Review overview, errors and predictions. Download the evidence bundle to preserve the result."
        )
    left, right = st.columns(2)
    with left:
        st.subheader("What works")
        st.write(
            "Image inference, labeled evaluation, per-class metrics, filtered overlays, latency analysis and immutable result downloads. Saved project evidence works without model weights."
        )
    with right:
        st.subheader("What is not claimed")
        st.write(
            "Production readiness, real-time video performance, counting accuracy or an independent test result without verified provenance. No training runs from this dashboard."
        )
    st.info(
        "Saved bundles contain predictions and annotations but no source photographs. Reopened results recompute metrics; their supplied provenance and timing are not independently authenticated."
    )
    with st.expander("Commands and technical report"):
        st.code(".venv/bin/python -m streamlit run dashboard/app.py", language="bash")
        st.code(
            ".venv/bin/python scripts/evaluate_new_data.py --zip new_images.zip --metadata metadata.json --device cpu --output runs/new_data",
            language="bash",
        )
        for path, label in [
            (
                "docs/reproducibility/NEW_DATA_EVALUATION.md",
                "Dataset preparation guide",
            ),
            (
                "docs/phase_reports/NEW_DATA_EVALUATION_DASHBOARD.md",
                "Evaluation protocol",
            ),
        ]:
            text = pages.load(path)
            if text:
                st.download_button(label, text, Path(path).name, mime="text/markdown")
    st.caption(
        "Uncertainty intervals cover fixed-threshold precision, recall and F1; not AP. Stratified views report matched recall. Occlusion, weather and lighting are not inferred from images."
    )


def render(selected, mode):
    result = st.session_state.get("new_result")
    ui.context(result)
    if mode == "Faculty Presentation Mode":
        sequence = (
            [NAV[0], NAV[2], NAV[3], NAV[4]]
            if result
            else [NAV[0], NAV[5], NAV[6], NAV[7]]
        )
        st.caption("PRESENTATION WALKTHROUGH · " + " → ".join(sequence))
        if selected in sequence:
            next_page = sequence[(sequence.index(selected) + 1) % len(sequence)]
            ui.action("Next: " + next_page, next_page, key="presentation_next")
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
        ui.historical()
        st.warning(
            "Historical validation2000 and matched calibration500 are distinct pools. Original validation scores are not directly comparable to new data. A calibration comparison appears below only when metric definitions and settings match; composition still affects scores."
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
        documentation()
