"""Exploratory Data Analysis (EDA) and Annotation Review components."""

import streamlit as st
import plotly.graph_objects as go
from utils.data_loader import get_eda_class_distribution
from utils.artifact_loader import find_file


def render_eda_section():
    """Render sections 04 and 05: EDA Analysis and Visual Annotation Review."""
    eda_df = get_eda_class_distribution()

    # --------------------------------------------------------------------------
    # 04 — Exploratory Data Analysis
    # --------------------------------------------------------------------------
    st.html(
        """
        <div style="margin-top: 3.5rem; margin-bottom: 1.5rem;" id="eda">
            <div class="section-kicker">04 / DATASET INSIGHTS</div>
            <div class="section-title">Exploratory Data Analysis (EDA)</div>
            <div class="section-subtitle">
                Quantitative distributions of 316,220 vehicle objects across traffic density, spatial positioning, scale variations, and class imbalance.
            </div>
        </div>
        """,
    )

    # Interactive Class Distribution Plotly Chart
    if eda_df is not None and not eda_df.empty:
        # Group by category
        if "category" in eda_df.columns:
            class_col = "category"
        elif "name" in eda_df.columns:
            class_col = "name"
        else:
            class_col = eda_df.columns[1]

        totals = eda_df.groupby(class_col)["instances"].sum().reset_index()
        totals = totals.sort_values(by="instances", ascending=False)
        total_sum = totals["instances"].sum()
        totals["share_pct"] = (totals["instances"] / total_sum) * 100.0

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=totals[class_col],
                y=totals["instances"],
                text=[
                    f"{v:,}<br>({p:.1f}%)"
                    for v, p in zip(totals["instances"], totals["share_pct"])
                ],
                textposition="auto",
                marker=dict(
                    color=totals["instances"],
                    colorscale=[[0, "#3b82f6"], [0.5, "#06b6d4"], [1.0, "#00e5ff"]],
                    line=dict(color="rgba(255, 255, 255, 0.2)", width=1),
                ),
                hovertemplate="<b>%{x}</b><br>Instances: %{y:,}<br>Share: %{customdata:.2f}%<extra></extra>",
                customdata=totals["share_pct"],
            )
        )
        fig.update_layout(
            title=dict(
                text="<b>UVH-26 Majority Voting Vehicle Instances by Class</b> (316,220 Total)",
                font=dict(size=16, color="#ffffff"),
            ),
            paper_bgcolor="rgba(14, 23, 42, 0.7)",
            plot_bgcolor="rgba(6, 12, 24, 0.5)",
            font=dict(family="JetBrains Mono, sans-serif", color="#94a3b8"),
            xaxis=dict(
                title="Vehicle Class (14 Categories)",
                tickangle=-35,
                gridcolor="rgba(255, 255, 255, 0.05)",
            ),
            yaxis=dict(
                title="Instance Count (Log Scale Recommended)",
                gridcolor="rgba(255, 255, 255, 0.08)",
            ),
            height=430,
            margin=dict(l=40, r=20, t=60, b=80),
        )
        st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)

    with c1:
        st.html(
            """
            <div class="glass-card">
                <div class="glass-card-header">
                    <div class="glass-card-title">⚠️ Class Imbalance Finding</div>
                    <span class="badge-pending">EXTREME IMBALANCE</span>
                </div>
                <div style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6;">
                    Indian urban road scenes exhibit heavy-tailed distributions:
                    <ul style="margin-top: 0.4rem; padding-left: 1.2rem;">
                        <li><b style="color: #00e5ff;">Two-wheelers:</b> 149,730 instances (<b>47.35%</b> of all traffic objects)</li>
                        <li><b style="color: #38bdf8;">Three-wheelers (Auto-rickshaws):</b> 52,428 instances (<b>16.58%</b>)</li>
                        <li><b style="color: #93c5fd;">Hatchbacks:</b> 30,290 instances (<b>9.58%</b>)</li>
                        <li><b style="color: #f59e0b;">Minority classes:</b> Others (352), Mini-bus (873), Tempo-traveller (1,680)</li>
                    </ul>
                </div>
                <div class="warning-callout" style="margin-top: 0.8rem; font-size: 0.88rem;">
                    <b>Evaluation Guidance:</b> Overall mAP alone will mask severe performance drop-offs on minority classes. Phase 2 must evaluate class-weighted loss and focal loss adaptations.
                </div>
            </div>
            """,
        )

    with c2:
        st.html(
            """
            <div class="glass-card">
                <div class="glass-card-header">
                    <div class="glass-card-title">🚗 Spatial Density &amp; Scale</div>
                    <span class="badge-complete">CALCULATED</span>
                </div>
                <div style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6;">
                    <ul style="margin-top: 0.4rem; padding-left: 1.2rem;">
                        <li><b>Objects per Image:</b> Mean = <b>11.87</b>, Median = <b>10.0</b>, Max = <b>66</b> objects/image</li>
                        <li><b>Resolution:</b> Dimensions checked per selected image; source mismatch quarantined</li>
                        <li><b>Bounding Box Scale Breakdown (Original Pixels):</b></li>
                    </ul>
                </div>
                <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem; font-family: var(--font-mono); margin-top: 0.6rem;">
                    <thead>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); color: #38bdf8;">
                            <th style="padding: 4px;">Scale Category</th>
                            <th style="padding: 4px; text-align: right;">Pixel Area</th>
                            <th style="padding: 4px; text-align: right;">Count</th>
                            <th style="padding: 4px; text-align: right;">Share</th>
                        </tr>
                    </thead>
                    <tbody style="color: #e2e8f0;">
                        <tr>
                            <td style="padding: 6px 4px;">Small</td>
                            <td style="padding: 6px 4px; text-align: right; color: #94a3b8;">&lt; 32² px</td>
                            <td style="padding: 6px 4px; text-align: right;">4,041</td>
                            <td style="padding: 6px 4px; text-align: right; color: #38bdf8;">1.3%</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 4px;">Medium</td>
                            <td style="padding: 6px 4px; text-align: right; color: #94a3b8;">32² – 96² px</td>
                            <td style="padding: 6px 4px; text-align: right;">112,887</td>
                            <td style="padding: 6px 4px; text-align: right; color: #38bdf8;">35.7%</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 4px;">Large</td>
                            <td style="padding: 6px 4px; text-align: right; color: #94a3b8;">&ge; 96² px</td>
                            <td style="padding: 6px 4px; text-align: right;">199,292</td>
                            <td style="padding: 6px 4px; text-align: right; color: #38bdf8;">63.0%</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """,
        )

    # Generated Figures Gallery (Tabs)
    st.html(
        "<h4 style='color: #ffffff; margin-top: 1.5rem;'>Generated EDA Figure Suite</h4>",
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔥 Spatial Heatmap",
            "📊 Class & Split",
            "📦 Box Geometry",
            "📈 Density Distributions",
        ]
    )

    with tab1:
        img_p = find_file("reports/figures/object_center_heatmap.png")
        if img_p:
            st.image(
                str(img_p),
                caption="Normalized Object Center 2D Spatial Heatmap across Indian Road Scenes",
                width="stretch",
            )
        else:
            st.info(
                "Heatmap figure available at reports/figures/object_center_heatmap.png"
            )

    with tab2:
        c_a, c_b = st.columns(2)
        with c_a:
            p1 = find_file("reports/figures/class_distribution.png")
            if p1:
                st.image(
                    str(p1),
                    caption="Valid Majority Voting Instances by Class",
                    width="stretch",
                )
        with c_b:
            p2 = find_file("reports/figures/split_distribution.png")
            if p2:
                st.image(
                    str(p2),
                    caption="Official Split Distribution (Train vs Val)",
                    width="stretch",
                )

    with tab3:
        g1, g2 = st.columns(2)
        with g1:
            p3 = find_file("reports/figures/bbox_area_distribution.png")
            if p3:
                st.image(
                    str(p3),
                    caption="Bounding Box Area Log Distribution",
                    width="stretch",
                )
        with g2:
            p4 = find_file("reports/figures/bbox_aspect_ratio.png")
            if p4:
                st.image(
                    str(p4),
                    caption="Bounding Box Aspect Ratio Distribution",
                    width="stretch",
                )

    with tab4:
        d1, d2 = st.columns(2)
        with d1:
            p5 = find_file("reports/figures/objects_per_image.png")
            if p5:
                st.image(
                    str(p5), caption="Objects per Image Histogram", width="stretch"
                )
        with d2:
            p6 = find_file("reports/figures/resolution_distribution.png")
            if p6:
                st.image(
                    str(p6),
                    caption="Image Dimension Distribution (1920x1080)",
                    width="stretch",
                )

    # --------------------------------------------------------------------------
    # 05 — Manual Annotation Review
    # --------------------------------------------------------------------------
    st.html(
        """
        <div style="margin-top: 3.5rem; margin-bottom: 1.5rem;" id="annotation-review">
            <div class="section-kicker">05 / VISUAL QUALITY ASSURANCE</div>
            <div class="section-title">Manual Annotation Review &amp; Qualitative Analysis</div>
            <div class="section-subtitle">
                Final subset annotation review covered 42 images; proper-model closeout compared six validation GT/prediction pairs. Earlier preview images below are historical examples.
            </div>
        </div>
        """,
    )

    r1, r2 = st.columns([1.2, 0.8])

    with r1:
        st.html(
            """
            <div class="glass-card">
                <div class="glass-card-header">
                    <div class="glass-card-title">🔍 Visual Inspection Summary</div>
                    <span class="badge-complete">42 FINAL SUBSET GT IMAGES</span>
                </div>
                <div style="font-size: 0.92rem; color: #cbd5e1; line-height: 1.6;">
                    <b>Key Observations:</b>
                    <ul style="padding-left: 1.2rem; margin-top: 0.4rem;">
                        <li><b>Coordinate Alignment:</b> No systematic coordinate transformation error found in the reviewed final subset; source limitations remain documented.</li>
                        <li><b>Dense Queue Handling:</b> Heavy overlapping between adjacent vehicles in bumper-to-bumper city traffic. Two-wheeler boxes correctly encompass riders.</li>
                        <li><b>Observed Source Annotation Imperfections:</b> Occasional unlabelled foreground motorbikes and loose bounding boxes around large trucks.</li>
                        <li><b>Redaction &amp; Privacy:</b> Source face/license-plate redactions are retained; no universal claim about their effect on model accuracy.</li>
                    </ul>
                </div>
            </div>
            """,
        )

    with r2:
        st.html(
            """
            <div class="glass-card">
                <div class="glass-card-header">
                    <div class="glass-card-title">📋 QA Findings Log</div>
                    <span class="badge-complete">JSON AUDIT</span>
                </div>
                <div class="terminal-window" style="margin: 0;">
                    <div class="terminal-header">
                        <span class="terminal-btn red"></span>
                        <span class="terminal-btn yellow"></span>
                        <span class="terminal-btn green"></span>
                        <span style="font-size: 0.72rem; color: #94a3b8; margin-left: 0.5rem;">reports/audit/visual_review.json</span>
                    </div>
                    <div class="terminal-body" style="font-size: 0.78rem; max-height: 140px; overflow-y: auto;">
{<br/>
&nbsp;&nbsp;"coverage": "All 14 categories represented",<br/>
&nbsp;&nbsp;"alignment": "No systematic coordinate scaling error",<br/>
&nbsp;&nbsp;"status": "reviewed_with_annotation_quality_limitations"<br/>
}
                    </div>
                </div>
            </div>
            """,
        )

    # Gallery of local sample contact sheets
    st.html(
        "<h4 style='color: #ffffff; margin-top: 1rem;'>Visual Annotation Previews</h4>",
    )

    sheet0 = find_file("reports/annotation_samples/early/sheet_0.jpg")
    pilot_sheet = find_file("reports/annotation_samples/pilot_val_sheet.jpg")
    val_sample = find_file("reports/annotation_samples/val_10412.jpg")
    train_sample = find_file("reports/annotation_samples/train_23593.jpg")

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        if sheet0:
            st.image(
                str(sheet0),
                caption="Preview Contact Sheet 0 — Multi-Class Indian Urban Traffic Sample",
                width="stretch",
            )
        elif pilot_sheet:
            st.image(
                str(pilot_sheet),
                caption="Pilot Validation Annotation Sheet",
                width="stretch",
            )

    with col_img2:
        if train_sample:
            st.image(
                str(train_sample),
                caption="train_23593.jpg — Dense Queue with Overlapping Two-Wheelers & Auto-Rickshaws",
                width="stretch",
            )
        elif val_sample:
            st.image(
                str(val_sample),
                caption="val_10412.jpg — Bounding Box Verification Sample",
                width="stretch",
            )
        elif sheet0:
            st.info("Sample preview images available in reports/annotation_samples/")
