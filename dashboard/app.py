"""Run with streamlit run dashboard/app.py; preserve the legacy root showcase."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from dashboard import pages

st.set_page_config(
    page_title="Indian Urban Traffic Analytics System", page_icon="🚗", layout="wide"
)
st.html("""<style>
.stApp {background:#f6f8fc;color:#142b49}
[data-testid="stSidebar"] {background:#142b49;color:white}
[data-testid="stSidebar"] * {color:inherit}
[data-testid="stMetric"] {background:white;border:1px solid #dce5f0;border-top:3px solid #ed9741;padding:18px;border-radius:10px}
[data-testid="stMetric"] * {color:#142b49 !important}
h1,h2,h3 {color:#173e68} .stMainBlockContainer {max-width:1400px}
</style>""")
with st.sidebar:
    st.header("URBAN TRAFFIC")
    st.caption("UVH-26 · Faculty demonstration")
    mode = st.radio("Application mode", ["Saved Evidence Mode", "Live Inference Mode"])
    selected = st.radio(
        "Navigate",
        [
            "Project Overview",
            "Image Detection",
            "Recorded Video Detection",
            "Traffic Analytics",
            "Model Comparison",
            "Training Analysis",
            "Dataset Insights",
            "Documentation and Reproducibility",
        ],
    )
    st.caption(
        "Verified · E1 YOLOv8s / 640\n\nResearch prototype · no live-video claim"
    )
st.caption(f"ACTIVE MODE: {mode}")
handlers = {
    "Project Overview": pages.overview,
    "Image Detection": lambda: pages.image_page(mode),
    "Recorded Video Detection": pages.video_page,
    "Traffic Analytics": pages.analytics,
    "Model Comparison": pages.comparison,
    "Training Analysis": pages.training,
    "Dataset Insights": pages.dataset,
    "Documentation and Reproducibility": pages.documentation,
}
try:
    handlers[selected]()
except (OSError, ValueError, KeyError, TypeError, ImportError):
    st.error(
        "This page could not load an artifact. Other pages remain available. See README_DEMO.md for troubleshooting."
    )
st.divider()
st.caption(
    "Indian Urban Traffic Analytics System · measured evidence, explicit limitations · no datasets or uploads stored by this app"
)
