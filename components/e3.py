"""Artifact-only E3 calibration closeout; no model or dataset access."""

import streamlit as st

from utils.artifact_loader import find_file, load_json

BASE = "reports/evaluations/E3_fasterrcnn_best_calibration500_v1/"


def render_e3_section():
    metrics = load_json(BASE + "metrics.json")
    index = load_json(BASE + "artifact_index.json")
    if not metrics or not index:
        return
    st.header("E3 Faster R-CNN · calibration-only closeout")
    st.caption(
        "20/20 epochs completed · selected epoch 13 · calibration500 only. No matched E1 comparison; reserved1500 and fusion remain unstarted."
    )
    columns = st.columns(3)
    for column, label, key in zip(
        columns,
        ["E3 calibration AP50", "E3 calibration AP50:95", "E3 harmonic aggregate F1"],
        ["map50", "map50_95", "harmonic_aggregate_f1"],
    ):
        column.metric(label, f"{metrics[key]:.4f}")
    bench = load_json(BASE + "benchmark.json")
    if bench:
        st.write(
            f"MPS batch-one: {bench['timings']['end_to_end_ms']['median']:.2f} ms median / {bench['timings']['end_to_end_ms']['p95']:.2f} ms p95; {bench['end_to_end_still_images_per_second']:.2f} still images/s. Includes file preprocessing and postprocessing; excludes video capture/display. Historical YOLO timings use a different protocol."
        )
    with st.expander("E3 measured artifacts and faculty summary"):
        for key, label in [
            ("technical_report", "Technical report"),
            ("faculty_summary", "Faculty summary"),
            ("reproduction", "Reproduction commands"),
        ]:
            path = find_file(index[key])
            if path:
                st.download_button(
                    label,
                    path.read_text(),
                    file_name=path.name,
                    mime="text/markdown",
                    key="e3_" + key,
                )
        figure = find_file(BASE + "training_curves.png")
        if figure:
            st.image(str(figure), caption="Measured E3 training and calibration curves")
        st.json(
            {
                "scope": "calibration500 only",
                "selected_epoch": 13,
                "metrics": metrics,
                "artifact_index": index,
            },
            expanded=False,
        )
