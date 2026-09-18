"""Real-Time Vehicle Detection for Indian Urban Roads Using YOLOv8 and UVH-26
Phase 1 Research Showcase & Faculty Presentation Application.

Author: Manav Shah
Date: 13 September 2026
"""

import streamlit as st

# Configure wide layout and page metadata
st.set_page_config(
    page_title="Real-Time Vehicle Detection · Phase 1 Review",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Stylesheet Injection
from utils.artifact_loader import ROOT, find_file

css_path = ROOT / "assets/styles.css"
if css_path.is_file():
    st.html(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
    )

# Import modular components
from components.hero import render_hero
from components.metrics import render_metrics
from components.pipeline import render_pipeline
from components.dataset import render_dataset_section
from components.eda import render_eda_section
from components.training import render_training_section
from components.testing import render_testing_section
from components.status import render_status_section
from components.baseline import render_baseline_section, render_e1_section, render_e2_section


def main():
    # --------------------------------------------------------------------------
    # Sidebar Navigation & Presentation Mode
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.html(
            """
            <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 1rem;">
                <div style="font-family: var(--font-mono); font-size: 0.72rem; color: #00e5ff; letter-spacing: 0.1em; font-weight: 700;">
                    RESEARCH DASHBOARD
                </div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #ffffff; margin-top: 0.2rem;">
                    UVH-26 · YOLOv8n
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8;">
                    Phase 1 Faculty Progress Review
                </div>
            </div>
            """,
        )

        pres_mode = st.toggle(
            "📽️ Presentation Mode",
            value=False,
            help="Increases font scaling and optimizes layout for high-resolution projectors.",
        )

        if pres_mode:
            st.html(
                """
                <style>
                    .section-title { font-size: 2.8rem !important; }
                    .kpi-value { font-size: 2.8rem !important; }
                    .hero-title { font-size: 3.8rem !important; }
                    .glass-card { padding: 2.2rem !important; }
                    body { font-size: 1.15rem !important; }
                </style>
                """,
            )

        st.html(
            "<p style='font-size: 0.8rem; font-family: var(--font-mono); color: #64748b; margin-bottom: 0.4rem;'>JUMP TO SECTION</p>",
        )

        st.html(
            """
            <div style="display: flex; flex-direction: column; gap: 0.4rem; font-size: 0.88rem;">
                <a href="#dataset-audit" style="color: #cbd5e1; text-decoration: none; padding: 4px 8px; border-radius: 6px; background: rgba(255,255,255,0.02);">📊 01. Dataset Audit</a>
                <a href="#data-engineering" style="color: #cbd5e1; text-decoration: none; padding: 4px 8px; border-radius: 6px; background: rgba(255,255,255,0.02);">📐 02. Data Engineering</a>
                <a href="#data-quality" style="color: #cbd5e1; text-decoration: none; padding: 4px 8px; border-radius: 6px; background: rgba(255,255,255,0.02);">🛡️ 03. Data Quality</a>
                <a href="#eda" style="color: #cbd5e1; text-decoration: none; padding: 4px 8px; border-radius: 6px; background: rgba(255,255,255,0.02);">📈 04. Exploratory Analysis</a>
                <a href="#annotation-review" style="color: #cbd5e1; text-decoration: none; padding: 4px 8px; border-radius: 6px; background: rgba(255,255,255,0.02);">🔍 05. Annotation Review</a>
                <a href="#model-architecture" style="color: #cbd5e1; text-decoration: none; padding: 4px 8px; border-radius: 6px; background: rgba(255,255,255,0.02);">🧠 06. YOLOv8n Architecture</a>
                <a href="#smoke-test" style="color: #cbd5e1; text-decoration: none; padding: 4px 8px; border-radius: 6px; background: rgba(255,255,255,0.02);">🖥️ 07. Smoke Run Pilot</a>
                <a href="#testing" style="color: #cbd5e1; text-decoration: none; padding: 4px 8px; border-radius: 6px; background: rgba(255,255,255,0.02);">🧪 08. Synthetic Testing</a>
                <a href="#reproducibility" style="color: #cbd5e1; text-decoration: none; padding: 4px 8px; border-radius: 6px; background: rgba(255,255,255,0.02);">🔒 09. Reproducibility</a>
                <a href="#what-remains" style="color: #cbd5e1; text-decoration: none; padding: 4px 8px; border-radius: 6px; background: rgba(255,255,255,0.02);">⏳ 10. Status &amp; Roadmap</a>
                <a href="#future-phases" style="color: #cbd5e1; text-decoration: none; padding: 4px 8px; border-radius: 6px; background: rgba(255,255,255,0.02);">🔮 Future Phases</a>
            </div>
            """,
        )

        st.html(
            "<div style='margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.08);'></div>",
        )

        pdf_file = find_file("output/pdf/UVH26_Faculty_Progress_Report.pdf")
        if pdf_file and pdf_file.is_file():
            with open(pdf_file, "rb") as f:
                st.download_button(
                    label="📄 Faculty Report (PDF)",
                    data=f.read(),
                    file_name="UVH26_Faculty_Progress_Report.pdf",
                    mime="application/pdf",
                    width="stretch",
                )

        st.html(
            """
            <div style="margin-top: 1rem; font-size: 0.75rem; color: #64748b; text-align: center; font-family: var(--font-mono);">
                Apple Silicon MPS · PyTorch 2.14<br/>
                Ultralytics 8.4 · Python 3.12
            </div>
            """,
        )

    # --------------------------------------------------------------------------
    # Main Dashboard Body
    # --------------------------------------------------------------------------
    render_hero()
    render_baseline_section()
    render_e1_section()
    render_e2_section()
    render_metrics()
    render_pipeline()
    render_dataset_section()
    render_eda_section()
    render_training_section()
    render_testing_section()
    render_status_section()

    # --------------------------------------------------------------------------
    # Professional Footer
    # --------------------------------------------------------------------------
    st.html(
        """
        <div class="app-footer">
            <div style="font-weight: 700; color: #f1f5f9; font-size: 1.05rem; margin-bottom: 0.3rem;">
                Real-Time Vehicle Detection and Traffic Analytics for Indian Urban Roads
            </div>
            <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.6rem;">
                Manav Shah · Fourth-Year Undergraduate Research · Phase 1 Progress Review · 13 September 2026
            </div>
            <div style="font-family: var(--font-mono); font-size: 0.78rem; color: #64748b;">
                Built with Python · Streamlit · YOLOv8 · UVH-26 (IISc AIM) · CC BY 4.0
            </div>
        </div>
        """,
    )


if __name__ == "__main__":
    main()
