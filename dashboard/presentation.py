"""Consistent project presentation and explicit next-step navigation."""

from html import escape

import plotly.express as px
import streamlit as st

MODES = ["Saved Evidence Mode", "Live Inference Mode", "Faculty Presentation Mode"]
MODE_LABELS = dict(
    zip(MODES, ["Browse saved results", "Evaluate new images", "Present to faculty"])
)


def theme():
    px.defaults.template = "plotly_white"
    px.defaults.color_discrete_sequence = [
        "#245b87",
        "#eaa14b",
        "#348679",
        "#9273ae",
        "#be5c5c",
    ]
    st.html("""<style>
    .stApp {background:#f5f7fb;color:#162c43}
    [data-testid="stHeader"] {background:#f5f7fb}
    [data-testid="stSidebar"] {background:#14283f;color:#e5edf5}
    [data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] h2 {color:#e5edf5}
    .stMainBlockContainer {max-width:1440px;padding-top:4.5rem;padding-bottom:3rem}
    h1 {font-size:2.35rem !important;letter-spacing:-.045em;line-height:1.13 !important;color:#14283f}
    h2,h3 {color:#14283f;letter-spacing:-.025em}
    [data-testid="stMetric"] {background:white;border:1px solid #dfe6ee;border-radius:12px;padding:18px;min-height:106px}
    [data-testid="stMetricValue"] {font-size:1.8rem;color:#173d60;white-space:normal;overflow:visible}
    [data-testid="stMetricLabel"] p {color:#53667b;white-space:normal}
    [data-testid="stVerticalBlockBorderWrapper"] {border-radius:12px}
    .eyebrow {font-size:.73rem;letter-spacing:.16em;text-transform:uppercase;font-weight:700;color:#687c90;margin-bottom:12px}
    .context-bar {padding:12px 18px;background:#e9eff6;border:1px solid #d9e2ed;border-radius:9px;font-size:.85rem;color:#36516c;margin-bottom:22px}
    .context-bar strong {color:#183a59}
    .hero-copy {font-size:1.06rem;line-height:1.65;color:#53667b;max-width:830px;margin-bottom:24px}
    .stButton button {border-radius:8px;min-height:42px}
    [data-testid="stTabs"] {margin-top:10px}
    @media(max-width:760px) {h1 {font-size:1.8rem !important} .stMainBlockContainer {padding:4.5rem 1.25rem 1.25rem} [data-testid="stMetricValue"] {font-size:1.3rem}}
    </style>""")


def heading(title, description, eyebrow="UVH-26 / Vehicle detection research"):
    st.html(f'<div class="eyebrow">{escape(eyebrow)}</div>')
    st.title(title)
    st.html(f'<div class="hero-copy">{escape(description)}</div>')


def go(page, mode=None):
    st.session_state["navigation"] = page
    if mode:
        st.session_state["application_mode"] = mode


def action(label, page, mode=None, key=None, primary=False):
    st.button(
        label,
        on_click=go,
        args=(page, mode),
        key=key or label,
        type="primary" if primary else "secondary",
        width="stretch",
    )


def empty(title, reason, labeled=False):
    heading(title, reason)
    with st.container(border=True):
        st.subheader("Your next step")
        st.write(
            "Upload images with independent YOLO or COCO labels to measure correctness."
            if labeled
            else "Evaluate your own images or reopen an exported result. Historical project evidence is always available."
        )
        left, right = st.columns(2)
        with left:
            action(
                "Evaluate new images",
                "New Data Evaluation",
                "Live Inference Mode",
                key=title + "_start",
                primary=True,
            )
        with right:
            action(
                "View historical results",
                "Historical Model Comparison",
                key=title + "_history",
            )


def percent(value):
    return "Not available" if value is None else f"{value * 100:.2f}%"


def cards(items):
    for i in range(0, len(items), 4):
        for col, item in zip(st.columns(min(4, len(items) - i)), items[i : i + 4]):
            label, value, *helptext = item
            col.metric(label, value, help=helptext[0] if helptext else None)


def context(result):
    if result:
        kind = (
            "Predictions only"
            if result["metadata"]["format"] == "unlabeled"
            else "Labeled evaluation"
        )
        origin = (
            "Imported records · provenance supplied by user"
            if result.get("import_note")
            else "Completed in this session"
        )
        text = f"<strong>{escape(result['metadata']['name'])}</strong> &nbsp; / &nbsp; {kind} &nbsp; / &nbsp; {len(result['records'])} images &nbsp; / &nbsp; {origin}"
    else:
        text = "<strong>Project evidence</strong> &nbsp; / &nbsp; No new dataset evaluated &nbsp; / &nbsp; E1 YOLOv8s selected"
    st.html(f'<div class="context-bar">{text}</div>')


def historical():
    """Display metric percentages only within a clearly named evaluation pool."""
    import pandas as pd

    from dashboard import pages

    heading(
        "Historical model comparison",
        "Model decisions come from saved experiments. Compare models within one evaluation pool; do not treat scores from different pools as interchangeable.",
    )
    pools = [
        (
            "Validation · 2,000 images",
            [(name, pages.EVAL + key + "/metrics.json") for name, key in pages.HIST],
        ),
        (
            "Calibration · 500 images",
            [
                (name, "reports/comparisons/E1_E3_stageE_v2/" + key + "_metrics.json")
                for name, key in [("YOLOv8s", "E1"), ("Faster R-CNN", "E3")]
            ],
        ),
    ]
    for tab, (title, entries) in zip(st.tabs([p[0] for p in pools]), pools):
        with tab:
            st.caption(title + " · Saved evidence")
            rows = []
            for name, path in entries:
                m = pages.load(path)
                if m:
                    rows.append(
                        {
                            "Model": name,
                            "Precision": m["precision"],
                            "Recall": m["recall"],
                            "Harmonic F1": m.get("f1", m.get("harmonic_aggregate_f1")),
                            "AP50": m["map50"],
                            "AP50:95": m["map50_95"],
                        }
                    )
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(
                    df.style.format(
                        {k: "{:.2%}" for k in df if k != "Model"},
                        na_rep="Not available",
                    ),
                    hide_index=True,
                    width="stretch",
                )
                fig = px.bar(
                    df,
                    x="Model",
                    y="AP50:95",
                    color="Model",
                    title="Box localization and classification quality",
                    range_y=[0, 1],
                )
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("This saved result is unavailable in the current installation.")
    st.success("Selected detector: YOLOv8s E1, epoch22, at640 px.")
    st.write(
        "Faster R-CNN was evaluated but did not replace E1. Fusion was rejected because it failed the predefined improvement criteria. Neither outcome authorizes a production-readiness claim."
    )
    st.caption(
        "Historical validation uses the original Ultralytics protocol. Matched calibration uses the common COCO evaluator and model-specific operating thresholds. AP integrates confidence, while precision/recall/F1 use a fixed threshold."
    )
