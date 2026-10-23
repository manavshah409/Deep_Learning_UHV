"""Read-only Stage F fusion feasibility summary from measured artifacts."""

import streamlit as st

from utils.artifact_loader import find_file, load_json

BASE = "reports/accuracy_stage_f/"


def render_fusion_section():
    result = load_json(BASE + "cross_validation_results.json")
    latency = load_json(BASE + "latency.json")
    if not result:
        return
    st.header("Stage F · fusion feasibility")
    st.info(
        "E4 fusion rejected under the frozen improvement gate; retain E1 YOLOv8s. Calibration-only five-fold evidence. Reserved1500 remains unopened."
    )
    st.table(
        [
            {
                "Family": name,
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "Macro F1": round(m["macro_class_f1"], 4),
                "AP50:95": round(m["map50_95"], 4),
            }
            for name, m in result["pooled_oof"].items()
        ]
    )
    if latency:
        st.caption(
            f"Strongest rejected candidate, actual sequential E1+E3+fusion: {latency['stage_ms']['end_to_end_ms']['mean']:.2f} ms mean / {latency['stage_ms']['end_to_end_ms']['p95']:.2f} ms p95; {latency['still_image_fps']:.2f} still images/s. Includes both detector pipelines and fusion; no live-video FPS claim."
        )
    st.write(
        "Neither precision nor recall reached70%. Next proposal: versioned training-data/label improvements followed by a separately authorized controlled YOLOv8s fine-tuning pilot. No training started."
    )
    report = find_file("docs/phase_reports/E4_FUSION_FEASIBILITY.md")
    if report:
        with st.expander("Fusion feasibility report"):
            st.download_button(
                "Download Stage F report",
                report.read_text(),
                file_name=report.name,
                mime="text/markdown",
                key="stage_f_report",
            )
