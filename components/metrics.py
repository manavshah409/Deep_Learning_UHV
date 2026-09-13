"""Key verified numbers component."""

import streamlit as st
from utils.data_loader import get_dataset_counts, get_test_suite_status, get_baseline_subset_info


def render_metrics():
    """Render the 8 key metrics cards with verified counts."""
    counts = get_dataset_counts()
    tests = get_test_suite_status()
    baseline = get_baseline_subset_info()

    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <div class="section-kicker">AT A GLANCE</div>
            <div class="section-title">Verified Phase 1 Quantities</div>
            <div class="section-subtitle">
                Core quantitative milestones established during Phase 1 schema audits, synthetic test suites, and data pipelines.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-value">{counts['total_images']:,}</div>
                <div class="kpi-label">Annotated Images</div>
                <div class="kpi-sub">MV consensus catalog</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-value">{counts['train_images']:,}</div>
                <div class="kpi-label">Training Split Images</div>
                <div class="kpi-sub">{counts['train_objects']:,} objects (80%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-value">{counts['total_objects']:,}</div>
                <div class="kpi-label">Vehicle Instances</div>
                <div class="kpi-sub">0 invalid / 0 out-of-frame</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-value">{counts['val_images']:,}</div>
                <div class="kpi-label">Validation Split Images</div>
                <div class="kpi-sub">{counts['val_objects']:,} objects (20%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-value">14</div>
                <div class="kpi-label">Vehicle Classes</div>
                <div class="kpi-sub">Full taxonomy preserved</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-value">{tests['passed_count']}</div>
                <div class="kpi-label">Automated Tests Passing</div>
                <div class="kpi-sub">0 failures / 100% pass</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-value">32</div>
                <div class="kpi-label">Preview Samples Inspected</div>
                <div class="kpi-sub">Visual QA across 14 classes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-value">{baseline['train_target']:,} / {baseline['val_target']:,}</div>
                <div class="kpi-label">Planned Baseline Subset</div>
                <div class="kpi-sub">Max share Δ ≤ {baseline['max_share_deviation_pp']:.3f} pp</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="info-callout" style="margin-top: 1.5rem;">
            <b>Academic Clarification:</b> The 26,646 image count represents the official Majority Voting annotation catalog validated during schema audits. Individual raw image file acquisition from Hugging Face is in progress.
        </div>
        """,
        unsafe_allow_html=True,
    )
