"""Current Status, Roadmap, and Phase 2/3 Preview component."""

from pathlib import Path
import streamlit as st
from utils.artifact_loader import find_file


def render_status_section():
    """Render sections 10, Roadmap, and Future Phase 2/3 preview."""
    # --------------------------------------------------------------------------
    # 10 — Current Status / What Remains
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div style="margin-top: 3.5rem; margin-bottom: 1.5rem;" id="what-remains">
            <div class="section-kicker">10 / MILESTONE SUMMARY</div>
            <div class="section-title">Current Phase 1 Status &amp; What Remains</div>
            <div class="section-subtitle">
                Clear boundary between verified completed engineering infrastructure and pending experimental benchmarks.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    from utils.artifact_loader import load_json
    from utils.data_loader import get_test_suite_status
    audit = load_json("reports/audit/baseline_subset_frozen_provenance.json")
    run = load_json("reports/tables/yolov8n_uvh26_mv_baseline_seed42_v1_provenance.json")
    tests = get_test_suite_status()
    st.write("Selected-image audit:", audit.get("image_audit_status") if audit else "pending")
    st.write("Proper subset training:", run.get("status") if run else "not started")
    st.write("Saved test result:", tests["text"])
    st.info("Full 26,646-image acquisition remains a separate future task. It does not block the explicitly defined subset baseline. Phase 2 remains unstarted.")

    # --------------------------------------------------------------------------
    # Next — Phase 2 & Phase 3 Preview
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div style="margin-top: 3.5rem; margin-bottom: 1.5rem;" id="future-phases">
            <div class="section-kicker">FUTURE RESEARCH</div>
            <div class="section-title">Next Horizons: Phase 2 &amp; Phase 3</div>
            <div class="section-subtitle">
                Planned architectural enhancements, comparative model evaluations, and real-time traffic analytics.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown(
            """
            <div class="glass-card" style="height: 100%;">
                <div style="font-family: var(--font-mono); font-size: 0.8rem; color: #00e5ff; font-weight: 700; margin-bottom: 0.4rem;">PHASE 2</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-bottom: 0.6rem;">
                    Model Comparison &amp; Imbalance Tuning
                </div>
                <div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.55;">
                    <ul style="padding-left: 1.1rem;">
                        <li><b>YOLOv8n vs YOLOv8s:</b> Evaluate accuracy vs latency trade-offs.</li>
                        <li><b>Class Imbalance Mitigation:</b> Compare Focal Loss vs Inverse Class Frequency weighting.</li>
                        <li><b>Consensus Variants:</b> ST vs MV detection robustness comparison.</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p2:
        st.markdown(
            """
            <div class="glass-card" style="height: 100%;">
                <div style="font-family: var(--font-mono); font-size: 0.8rem; color: #38bdf8; font-weight: 700; margin-bottom: 0.4rem;">PHASE 2+</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-bottom: 0.6rem;">
                    Small Object &amp; Dense Queue Recall
                </div>
                <div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.55;">
                    <ul style="padding-left: 1.1rem;">
                        <li><b>Multi-Scale Feature Anchoring:</b> Improve detection of distant two-wheelers and bicycles.</li>
                        <li><b>Augmentation Strategy:</b> Mosaic, MixUp, and Albumentations tailored for congested traffic scenes.</li>
                        <li><b>Diagnostic Error Isolation:</b> Confusion matrix decomposition.</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p3:
        st.markdown(
            """
            <div class="glass-card" style="height: 100%;">
                <div style="font-family: var(--font-mono); font-size: 0.8rem; color: #a78bfa; font-weight: 700; margin-bottom: 0.4rem;">PHASE 3</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-bottom: 0.6rem;">
                    Traffic Analytics &amp; Edge Deployment
                </div>
                <div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.55;">
                    <ul style="padding-left: 1.1rem;">
                        <li><b>Multi-Object Tracking:</b> ByteTrack / BoT-SORT integration.</li>
                        <li><b>Bi-Directional Counting:</b> Virtual tripwire crossing analytics.</li>
                        <li><b>Spatial Density &amp; Congestion:</b> Real-time heatmaps &amp; live dashboard UI.</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Formal Progress Report Download
    pdf_path = find_file("output/pdf/UVH26_Faculty_Progress_Report.pdf")
    if pdf_path and pdf_path.is_file():
        st.markdown(
            """
            <div style="margin-top: 2rem; background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 12px; padding: 1.2rem 1.6rem; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 1rem;">
                <div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff;">📄 Faculty Progress Report Deliverable</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">5-page formal ReportLab PDF containing audited metrics, EDA plots, and technical limitations.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download UVH-26 Faculty Progress Report (PDF)",
                data=f.read(),
                file_name="UVH26_Faculty_Progress_Report.pdf",
                mime="application/pdf",
                type="secondary",
            )
