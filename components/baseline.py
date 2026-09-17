"""Current subset baseline evidence, read from saved artifacts only."""

import streamlit as st
from utils.artifact_loader import load_json, load_csv

NAME = "yolov8n_uvh26_mv_baseline_seed42_v1"
EVAL = "yolov8n_uvh26_mv_baseline_seed42_v1_validation"


def render_baseline_section():
    st.subheader("UVH-26 MV 8,000/2,000 subset baseline")
    st.caption(
        "Subset validation results. The one-epoch smoke and preflight runs are separate engineering checks."
    )
    provenance = load_json(f"reports/tables/{NAME}_provenance.json")
    metrics = load_json(f"reports/tables/{EVAL}_metrics.json")
    frozen = load_json("reports/audit/baseline_subset_frozen_provenance.json")
    timing = load_json(f"reports/tables/{NAME}_latency.json")
    if provenance:
        st.write("Training status:", provenance.get("status", "unknown"))
        st.write(
            "Last saved epoch:", provenance.get("last_saved_epoch", "not yet saved")
        )
    else:
        st.info(
            "Proper baseline has not started. Final subset preflight is a separate run."
        )
    if metrics:
        cols = st.columns(5)
        for col, label, key in zip(
            cols,
            ["Precision", "Recall", "F1", "mAP@0.5", "mAP@0.5:0.95"],
            ["precision", "recall", "f1", "map50", "map50_95"],
        ):
            col.metric(label, f"{metrics[key]:.4f}")
        st.caption(metrics["f1_note"])
        st.write("Macro per-class F1:", metrics["macro_f1"])
        per_class = load_csv(f"reports/tables/{EVAL}_per_class.csv")
        if per_class is not None:
            st.dataframe(per_class, hide_index=True)
        from utils.artifact_loader import find_file

        with st.expander("Standalone validation curves and confusion matrix"):
            st.caption(
                "Confusion matrix: confidence .001, matching IoU .45; distinct from F1 threshold."
            )
            for suffix in ["BoxPR_curve", "BoxF1_curve", "confusion_matrix_normalized"]:
                path = find_file(f"reports/figures/{EVAL}_{suffix}.png")
                if path:
                    st.image(str(path), width="stretch")
    if timing:
        st.write(
            "Batch-one still-image timing, including file read/decode; excludes video capture and display."
        )
        st.json(timing["summary"])
    if frozen:
        with st.expander("Frozen subset provenance"):
            st.json(frozen)


def render_e1_section():
    st.subheader("Phase 2 E1: YOLOv8s versus frozen YOLOv8n")
    registry = load_json("configs/experiment_registry.json")
    if not registry:
        st.info("E1 is not registered.")
        return
    e1 = next(
        (
            x
            for x in registry["experiments"]
            if x["id"] == "E1_yolov8s_uvh26_mv_640_seed42"
        ),
        None,
    )
    if e1:
        st.write("E1 status:", e1["status"])
        st.caption(
            "Same frozen 8000/2000 subset. Model capacity is the intended independent variable; preflight is not an accuracy result."
        )
    run = load_json("reports/tables/E1_yolov8s_uvh26_mv_640_seed42_provenance.json")
    if run:
        st.write("Proper E1 training:", run["status"])
        st.write("Last saved E1 epoch:", run.get("last_saved_epoch", "none yet"))
    review = load_json("reports/comparisons/E1_paired_summary.json")
    if review and review.get("status") == "manually_reviewed":
        st.write("Preferred research detector:", review["preferred_model"])
        st.caption(
            "Training completed; initial export failure recovered. Matched still-image timing does not establish live-video performance."
        )
    table = load_csv("reports/comparisons/E1_vs_E0/overall.csv")
    if table is not None:
        st.dataframe(table, hide_index=True)
        st.caption(
            "Deltas are absolute percentage points. Harmonic aggregate F1 is not micro-F1. No independent test or live-video claim."
        )
        st.dataframe(
            load_csv("reports/comparisons/E1_vs_E0/per_class.csv"), hide_index=True
        )
        st.dataframe(
            load_csv("reports/comparisons/E1_vs_E0/latency.csv"), hide_index=True
        )
