"""Interactive engineering pipeline component."""

import streamlit as st


def render_pipeline():
    """Render the interactive step-by-step engineering pipeline."""
    st.html(
        """
        <div style="margin-top: 3rem; margin-bottom: 1.5rem;">
            <div class="section-kicker">WORKFLOW &amp; ARCHITECTURE</div>
            <div class="section-title">Phase 1 Engineering Pipeline</div>
            <div class="section-subtitle">
                An end-to-end reproducible research workflow from raw COCO annotations to model baseline and downstream analytics.
            </div>
        </div>
        """,
    )

    steps = [
        {
            "num": "01",
            "name": "UVH-26 Annotation Ingestion",
            "status": "COMPLETED",
            "badge": "badge-complete",
            "desc": "Official IISc AIM UVH-26 dataset. Pinned revision hash <code>59f82c57</code>. Majority Voting (MV) consensus annotations loaded.",
            "artifact": "data/raw/UVH-26/ (pinned revision)",
        },
        {
            "num": "02",
            "name": "COCO Schema & Geometry Audit",
            "status": "COMPLETED",
            "badge": "badge-complete",
            "desc": "Audited annotation catalogue: 26,646 image records and 316,220 bounding boxes. Zero out-of-frame or non-finite boxes detected. Zero duplicate IDs.",
            "artifact": "reports/audit/schema.json & annotation_audit.json",
        },
        {
            "num": "03",
            "name": "COCO → YOLO Translation",
            "status": "COMPLETED",
            "badge": "badge-complete",
            "desc": "Mathematical conversion: <code>[x, y, w, h] → [class_id, (x+w/2)/W, (y+h/2)/H, w/W, h/H]</code>. Strict rejection of faulty boxes (0 implicit repairs). Empty images preserved as background.",
            "artifact": "src/data/convert_to_yolo.py & configs/class_mapping.yaml",
        },
        {
            "num": "04",
            "name": "Split Integrity & Leakage Checks",
            "status": "COMPLETED",
            "badge": "badge-complete",
            "desc": "Annotation-catalog split checks and independently audited 8,000/2,000 subset hash separation. No full-catalog pixel-integrity claim.",
            "artifact": "reports/audit/annotation_audit.json ('leakage')",
        },
        {
            "num": "05",
            "name": "Exploratory Data Analysis (EDA)",
            "status": "COMPLETED",
            "badge": "badge-complete",
            "desc": "Calculated class frequencies (Two-wheelers 47.35% vs Others 0.11%), object density per image (mean: 11.87), and bounding box scale histograms.",
            "artifact": "reports/figures/*.png & reports/tables/eda_summary.json",
        },
        {
            "num": "06",
            "name": "Automated Test Suite",
            "status": "COMPLETED",
            "badge": "badge-complete",
            "desc": "Automated unit and integration tests covering geometry normalization, class preservation, leakage blocking, and provenance logging.",
            "artifact": "tests/test_*.py (pytest: 100% pass)",
        },
        {
            "num": "07",
            "name": "YOLOv8n Smoke Training Pilot",
            "status": "COMPLETED",
            "badge": "badge-complete",
            "desc": "One-epoch training test on 64 train / 32 val images on Apple Silicon MPS. Checkpoint SHA-256 verified. Standalone evaluation confirmed.",
            "artifact": "runs/yolov8n_uvh26_mv_smoke_seed42/ & provenance.json",
        },
        {
            "num": "08",
            "name": "Proper 30-Epoch Baseline Experiment",
            "status": "COMPLETED",
            "badge": "badge-complete",
            "desc": "Frozen 8,000 train / 2,000 val subset; maximum class-share deviation 0.3063 pp. 30 epochs completed on MPS. Standalone best-checkpoint validation and batch-one timing appear above.",
            "artifact": "runs/yolov8n_uvh26_mv_baseline_seed42_v1/",
        },
        {
            "num": "09",
            "name": "Phase 2 & Phase 3: Analytics & Tracking",
            "status": "ROADMAP",
            "badge": "badge-pending",
            "desc": "Class-imbalance loss optimization, small vehicle detection refinement, ByteTrack multi-object tracking, zone vehicle counting, and live UI.",
            "artifact": "Phase 2/3 Milestones",
        },
    ]

    col_left, col_right = st.columns([1, 1])

    for i, step in enumerate(steps):
        target_col = col_left if i % 2 == 0 else col_right
        is_pending = "PENDING" in step["status"] or "ROADMAP" in step["status"]

        with target_col:
            st.html(
                f"""
                <div class="glass-card" style="border-left: 4px solid {"#f59e0b" if is_pending else "#00e5ff"}; margin-bottom: 1.2rem;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.6rem;">
                        <span style="font-family: var(--font-mono); font-size: 0.85rem; color: #94a3b8; font-weight: 700;">STEP {step["num"]}</span>
                        <span class="{step["badge"]}">{step["status"]}</span>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-bottom: 0.5rem;">
                        {step["name"]}
                    </div>
                    <div style="font-size: 0.9rem; color: #94a3b8; line-height: 1.5; margin-bottom: 0.8rem;">
                        {step["desc"]}
                    </div>
                    <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #38bdf8; background: rgba(56, 189, 248, 0.08); padding: 0.3rem 0.6rem; border-radius: 6px; display: inline-block;">
                        📁 {step["artifact"]}
                    </div>
                </div>
                """,
            )
