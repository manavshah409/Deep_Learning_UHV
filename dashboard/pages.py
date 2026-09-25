"""Faculty pages backed by existing, immutable evidence."""

import time

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.services import (
    CHECKPOINT,
    EXPECTED_SHA,
    ROOT,
    artifact,
    counts,
    decode_image,
    density,
    exports,
    infer,
    make_detector,
    verify_checkpoint,
)

load = st.cache_data(show_spinner=False)(artifact)
model = st.cache_resource(show_spinner=False)(make_detector)
EVAL = "reports/evaluations/"
HIST = [
    ("YOLOv8n", "yolov8n_uvh26_mv_e0_validation_seed42_v2"),
    ("YOLOv8s", "yolov8s_uvh26_mv_e1_validation_seed42_v2"),
]


def unavailable():
    st.info("Unavailable · This artifact is not available in this installation.")


def overview():
    st.caption("VERIFIED RESEARCH · FACULTY DEMONSTRATION")
    st.title("Indian Urban Traffic Analytics System")
    st.write(
        "Vehicle detection for heterogeneous Indian urban roads: cars, two-wheelers, "
        "three-wheelers and commercial vehicles share dense, frequently occluded scenes."
    )
    values = [
        ("Annotated images", "26,646"),
        ("Vehicle boxes", "316,220"),
        ("Classes", "14"),
        ("Training subset", "8,000"),
        ("Historical validation", "2,000"),
        ("Selected model", "YOLOv8s"),
    ]
    for cols, group in [(st.columns(3), values[:3]), (st.columns(3), values[3:])]:
        for col, (label, value) in zip(cols, group):
            col.metric(label, value)
    m = load(EVAL + HIST[1][1] + "/metrics.json")
    if m:
        st.metric("Historical validation2000 AP50:95", f"{100 * m['map50_95']:.3f}%")
    st.subheader("From road imagery to interpretable evidence")
    st.write(
        "Upload → validate → verified E1 / 640 detector → class-labelled boxes → composition and density analytics"
    )
    st.write(
        "Objectives: recognise 14 vehicle classes, inspect detection failures, compare models fairly, "
        "and provide reproducible, interpretable traffic summaries."
    )
    st.dataframe(
        pd.DataFrame(
            [
                [
                    "Verified",
                    "E1 selected; image inference and saved research evidence",
                ],
                ["Experimental", "Faster R-CNN completed; Stage F fusion rejected"],
                [
                    "Planned",
                    "Recorded-video UI, validated tracking and line-crossing counts",
                ],
            ],
            columns=["Status", "Capability"],
        ),
        hide_index=True,
        width="stretch",
    )
    st.warning(
        "Research prototype, not production-ready. The 70–80% precision and recall target has not been achieved."
    )
    st.caption(
        "Full annotation-catalogue audit: 26,646 images. Local integrity audit: selected 10,000 images. "
        "Acquisition and integrity of unselected images remain incomplete. Reserved1500 remains unused."
    )


def image_page(mode):
    st.title("Image Detection")
    st.caption(
        "E1 YOLOv8s · image size 640 · checkpoint verified before every inference request"
    )
    if mode == "Saved Evidence Mode":
        st.info(
            "Presentation-safe mode: no weights or inference required. Switch modes for your own image."
        )
        st.write(
            "Saved annotated demonstration media: Not available in the portable package. "
            "No dataset photographs are bundled without an explicit media selection."
        )
        image_artifact(
            EVAL + HIST[1][1] + "/confusion_matrix_normalized.png",
            "Saved E1 validation2000 confusion matrix",
        )
        return
    if not (ROOT / CHECKPOINT).is_file():
        st.warning(
            "Unavailable · E1 checkpoint is absent. Use Saved Evidence Mode; see launch instructions for placement."
        )
        return
    upload = st.file_uploader(
        "JPG, JPEG or PNG · maximum 10 MB / 20 MP", type=["jpg", "jpeg", "png"]
    )
    a, b, c = st.columns(3)
    conf = a.slider("Confidence", 0.05, 0.95, 0.34, 0.01)
    iou = b.slider("NMS IoU", 0.1, 0.95, 0.7, 0.05)
    maximum = c.number_input("Maximum detections", 1, 300, 300)
    device = st.selectbox(
        "Device", ["mps", "cpu"], help="Unavailable MPS falls back to CPU."
    )
    labels = st.checkbox("Show labels", True)
    scores = st.checkbox("Show confidence", True)
    if upload and st.button("Detect vehicles", type="primary"):
        st.session_state.pop("detection", None)
        try:
            # Checkpoint loading/verification deliberately excluded from media timing.
            verify_checkpoint(ROOT / CHECKPOINT)
            with st.spinner("Preparing verified detector…"):
                resource = model(device, conf, iou, int(maximum))
            begin = time.perf_counter()
            original = decode_image(upload.getvalue(), upload.name)
            result = infer(original, resource, labels, scores)
            result["end_to_end_ms"] = 1000 * (time.perf_counter() - begin)
            result["original"] = original
            st.session_state["detection"] = result
        except (ValueError, RuntimeError, OSError, ImportError):
            st.error(
                "Inference unavailable: check image format, verified checkpoint and device compatibility. "
                "Try CPU or Saved Evidence Mode. No output was saved."
            )
    result = st.session_state.get("detection")
    if result:
        a, b = st.columns(2)
        a.image(result["original"], caption="Original", width="stretch")
        b.image(
            result["png"],
            caption="E1 predictions · last completed request",
            width="stretch",
        )
        st.metric("Total detections", len(result["rows"]))
        st.write(counts(result["rows"]))
        st.dataframe(pd.DataFrame(result["rows"]), hide_index=True)
        st.write(
            f"Device: {result['device']} · detector: {result['inference_ms']:.2f} ms · "
            f"end-to-end: {result['end_to_end_ms']:.2f} ms"
        )
        st.caption(
            "Synchronized single-image request. Detector includes preprocessing/NMS and CPU box transfer. "
            "End-to-end includes decode, detector, annotation and PNG encoding; excludes model load, "
            "hash verification, network transfer and browser rendering. Not a benchmark or live-video FPS."
        )
        csv_data, json_data = exports(result["rows"])
        st.download_button(
            "Annotated PNG", result["png"], "e1_prediction.png", "image/png"
        )
        st.download_button(
            "Predictions CSV", csv_data, "e1_predictions.csv", "text/csv"
        )
        st.download_button(
            "Predictions JSON", json_data, "e1_predictions.json", "application/json"
        )


def video_page():
    st.title("Recorded Video Detection")
    st.info(
        "Planned · Upload integration is coming next. The existing detector-only CLI is preserved."
    )
    st.write(
        "The current pipeline lacks streamed UI progress and persisted per-class frame predictions. "
        "It also uses an MP4 codec that is not reliably browser-playable. A safe adapter and codec checks "
        "are required before enabling uploads. No video processing is started here."
    )
    st.warning(
        "Detection totals across frames are not unique-vehicle counts. Tracking and validated line-crossing counts are future work."
    )
    evidence = load("reports/audit/phase3_synthetic_smoke.json")
    if evidence:
        st.subheader("Verified saved synthetic smoke evidence")
        st.caption(
            "Synthetic 20-frame smoke; not road-video accuracy or real-time performance."
        )
        st.json(
            {
                k: evidence.get(k)
                for k in [
                    "scope",
                    "processed_frame_count",
                    "processing_seconds",
                    "latency_ms",
                    "failed_reads",
                ]
            }
        )
    else:
        unavailable()


def analytics():
    st.title("Traffic Analytics")
    result = st.session_state.get("detection")
    if not result:
        st.info(
            "No session predictions yet. Run Image Detection to populate analytics. "
            "Saved Evidence Mode does not invent traffic observations."
        )
        return
    rows = result["rows"]
    totals = counts(rows)
    a, b = st.columns(2)
    low = a.number_input("Low: maximum visible vehicles", 0, 100, 5)
    medium = b.number_input(
        "Moderate: maximum visible vehicles", int(low) + 1, 1000, max(15, int(low) + 1)
    )
    st.metric("Prototype density", density(len(rows), low, medium))
    st.caption(
        "Configurable visible-vehicle rule, not an official congestion standard. One image is one observation; no unique-vehicle count."
    )
    if totals:
        df = pd.DataFrame(totals.items(), columns=["Class", "Detections"])
        a, b = st.columns(2)
        a.plotly_chart(
            px.bar(df, x="Class", y="Detections", color_discrete_sequence=["#194a80"]),
            width="stretch",
        )
        b.plotly_chart(
            px.pie(df, names="Class", values="Detections", hole=0.6), width="stretch"
        )
        st.write(f"Most frequently detected class: {max(totals, key=totals.get)}")
    st.write(
        f"Average detections per observation: {len(rows)} · Peak-density observation: uploaded image ({len(rows)})"
    )
    st.caption(
        "Detections across video frames: Not available until the video adapter is implemented."
    )


def comparison():
    st.title("Model Comparison")
    for title, entries in [
        (
            "Historical validation2000",
            [(name, EVAL + key + "/metrics.json") for name, key in HIST],
        ),
        (
            "Matched calibration500",
            [
                (name, "reports/comparisons/E1_E3_stageE_v2/" + key + "_metrics.json")
                for name, key in [("YOLOv8s", "E1"), ("Faster R-CNN", "E3")]
            ],
        ),
    ]:
        st.subheader(title)
        rows = []
        for name, path in entries:
            m = load(path)
            if m:
                rows.append(
                    {
                        "Model": name,
                        "Precision": m["precision"],
                        "Recall": m["recall"],
                        "F1": m.get("f1", m.get("harmonic_aggregate_f1")),
                        "Macro_F1": m.get("macro_f1", m.get("macro_class_f1")),
                        "AP50": m["map50"],
                        "AP50_95": m["map50_95"],
                        "Threshold": m.get(
                            "confidence", m.get("f1_operating_confidence")
                        ),
                    }
                )
        if rows:
            st.dataframe(
                pd.DataFrame(rows).set_index("Model").style.format("{:.6f}"),
                width="stretch",
            )
        else:
            unavailable()
    st.warning(
        "These evaluation pools and operating protocols differ. Do not compare scores across tables."
    )
    st.write(
        "YOLOv8s is selected for deployment demonstrations. Faster R-CNN remains experimental. "
        "F1 is the harmonic mean of aggregate precision and recall; macro F1 averages class F1. "
        "AP integrates a score sweep, separately from fixed-threshold P/R/F1."
    )
    st.write(
        "Stage F fusion feasibility was completed and rejected: gains did not meet the preregistered gate. "
        "No fusion runs in this application. Neither model achieved 70–80% precision and recall."
    )


def training():
    st.title("Training Analysis")
    paths = [
        "reports/tables/yolov8n_uvh26_mv_baseline_seed42_v1_training.csv",
        "reports/tables/E1_yolov8s_uvh26_mv_640_seed42_training.csv",
        EVAL + "E3_fasterrcnn_best_calibration500_v1/epoch_timing.csv",
    ]
    for tab, path, best in zip(
        st.tabs(["YOLOv8n", "YOLOv8s", "Faster R-CNN"]), paths, [30, 22, 13]
    ):
        with tab:
            df = load(path)
            st.caption(f"Selected checkpoint: best · epoch {best}. Source: {path}")
            if df is None or df.empty:
                unavailable()
                continue
            df.columns = df.columns.str.strip()
            for label, tokens in [
                ("Losses", ["loss"]),
                ("Accuracy", ["precision", "recall", "mAP", "map50"]),
                ("Learning rate", ["lr/", "learning_rate"]),
                ("Timing (seconds)", ["seconds", "time"]),
            ]:
                cols = [
                    c
                    for c in df
                    if any(t in c for t in tokens)
                    and pd.api.types.is_numeric_dtype(df[c])
                ]
                if cols:
                    st.subheader(label)
                    st.line_chart(df.set_index("epoch")[cols])
                else:
                    st.caption(f"{label}: Not available in saved history.")
            st.caption(
                "YOLO time is recorded cumulative training time; E3 has explicit train/validation/total active timing. "
                "Missing validation losses or timing components are not synthesized."
            )
            st.download_button(
                "Download recorded history",
                df.to_csv(index=False),
                path.split("/")[-1],
                key=path,
            )


def image_artifact(path, caption):
    p = ROOT / path
    if p.is_file():
        st.image(str(p), caption=caption, width="stretch")
    else:
        unavailable()


def dataset():
    st.title("Dataset Insights")
    st.write(
        "Saved annotation-catalogue EDA; no startup dataset scan. Full catalogue: 26,646 images / "
        "316,220 boxes. Local selected subset integrity: 10,000 images."
    )
    for title, filename in [
        ("Class imbalance", "class_distribution.png"),
        ("Object size", "bbox_area_distribution.png"),
        ("Objects per image / scene density", "objects_per_image.png"),
        ("Spatial distribution", "object_center_heatmap.png"),
    ]:
        with st.expander(title, expanded=title == "Class imbalance"):
            image_artifact("reports/figures/" + filename, title)
    st.write(
        "Known limitations: small and occluded vehicles, visually ambiguous class boundaries, "
        "annotation omissions and low support for rare classes. Unselected-image integrity remains incomplete."
    )
    st.info(
        "Representative annotation photographs: Not available in this portable demo. Frozen annotations are unchanged."
    )
    with st.expander("Recorded subset integrity"):
        value = load("reports/audit/baseline_closeout_integrity.json")
        st.json(value) if value else unavailable()


def documentation():
    st.title("Documentation and Reproducibility")
    st.write(
        "E0 baseline → E1 YOLOv8s selection → E2 resolution study → E3 Faster R-CNN → "
        "Stage E matched calibration protocol → Stage F fusion rejection → faculty app."
    )
    st.code("streamlit run dashboard/app.py")
    st.caption(
        "Use the dedicated demo environment described in README_DEMO.md. No model loads on import."
    )
    st.write("Expected E1 SHA-256")
    st.code(EXPECTED_SHA)
    st.write("Local checkpoint location")
    st.code(CHECKPOINT)
    for path in [
        "docs/ALL_MODEL_PERFORMANCE.md",
        "docs/phase_reports/E4_FUSION_FEASIBILITY.md",
        "docs/reproducibility/STREAMLIT_DEMO.md",
    ]:
        with st.expander(path):
            content = load(path)
            if content:
                st.markdown(content)
                st.download_button(
                    "Download Markdown", content, path.split("/")[-1], key=path
                )
            else:
                unavailable()
    with st.expander("Recorded E1 environment and provenance"):
        m = load(EVAL + HIST[1][1] + "/metrics.json")
        if m:
            st.json(
                {
                    k: v
                    for k, v in m.items()
                    if "sha256" in k
                    or k in ["python", "torch", "ultralytics", "platform"]
                }
            )
        else:
            unavailable()
    st.caption(
        "Recorded environment belongs to evaluation, not necessarily this running demo. "
        "Training, reserved-split evaluation, fusion inference and public deployment are outside this app."
    )
