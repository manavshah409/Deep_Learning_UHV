"""Run with streamlit run dashboard/app.py; preserve the legacy root showcase."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from dashboard import presentation as ui
from dashboard.new_data import views

st.set_page_config(
    page_title="UVH-26 | Vehicle Detection Research", page_icon="🚗", layout="wide"
)
ui.theme()
with st.sidebar:
    st.header("UVH-26")
    st.caption("VEHICLE DETECTION RESEARCH")
    mode = st.radio(
        "Workspace",
        ui.MODES,
        format_func=lambda m: ui.MODE_LABELS[m],
        key="application_mode",
    )
    st.divider()
    selected = st.radio("Navigate", views.NAV, key="navigation")
    st.divider()
    st.caption("Selected detector")
    st.markdown("**YOLOv8s · Epoch 22**")
    st.caption(
        "14 vehicle classes · 640 px — Research evaluation, not a live-video system"
    )
try:
    views.render(selected, mode)
except (OSError, ValueError, KeyError, TypeError, ImportError):
    st.error(
        "This page could not load an artifact. Other pages remain available. See README_DEMO.md for troubleshooting."
    )
st.divider()
st.caption(
    "UVH-26 research project · Historical evidence stays separate from new-data results · Uploads stay in this browser session unless you download them"
)
