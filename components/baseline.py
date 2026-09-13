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
