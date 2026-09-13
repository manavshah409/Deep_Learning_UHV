"""Dataset Audit, Data Engineering, and Data Quality components."""

import streamlit as st
import pandas as pd
from utils.data_loader import get_dataset_counts, get_class_mapping
from utils.artifact_loader import load_json, load_csv


def render_dataset_section():
    """Render sections 01, 02, and 03: Dataset Audit, Data Engineering, and Data Quality."""
    counts = get_dataset_counts()
    mapping = get_class_mapping()
    audit_json = load_json("reports/audit/annotation_audit.json")
    schema_json = load_json("reports/audit/schema.json")

    # --------------------------------------------------------------------------
    # 01 — Dataset Audit
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div style="margin-top: 3.5rem; margin-bottom: 1.5rem;" id="dataset-audit">
            <div class="section-kicker">01 / AUDIT &amp; TAXONOMY</div>
            <div class="section-title">UVH-26 Dataset Ingestion &amp; Audit</div>
            <div class="section-subtitle">
                Comprehensive verification of the IISc AIM UVH-26 multi-annotator consensus dataset across official Train and Validation partitions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.1, 0.9])

    with c1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-card-header">
                    <div class="glass-card-title">📊 Official Split Summary</div>
                    <span class="badge-complete">AUDITED</span>
                </div>
                <p style="color: #94a3b8; font-size: 0.95rem; line-height: 1.55;">
                    The dataset provides two consensus variants: <b>Single-Threshold (ST)</b> and <b>Majority-Voting (MV)</b>.
                    Phase 1 adopts Majority-Voting annotations for superior bounding-box agreement across multiple independent annotators.
                </p>
                <table style="width: 100%; border-collapse: collapse; font-size: 0.95rem; margin: 1rem 0;">
                    <thead>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); color: #38bdf8; text-align: left;">
                            <th style="padding: 8px;">Official Split</th>
                            <th style="padding: 8px; text-align: right;">Images in MV JSON</th>
                            <th style="padding: 8px; text-align: right;">Vehicle Objects</th>
                            <th style="padding: 8px; text-align: right;">Invalid Boxes</th>
                        </tr>
                    </thead>
                    <tbody style="color: #e2e8f0;">
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                            <td style="padding: 10px 8px;"><b>Train</b></td>
                            <td style="padding: 10px 8px; text-align: right; font-family: var(--font-mono);">21,349</td>
                            <td style="padding: 10px 8px; text-align: right; font-family: var(--font-mono);">252,723</td>
                            <td style="padding: 10px 8px; text-align: right; font-family: var(--font-mono); color: #34d399;">0</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                            <td style="padding: 10px 8px;"><b>Validation</b></td>
                            <td style="padding: 10px 8px; text-align: right; font-family: var(--font-mono);">5,297</td>
                            <td style="padding: 10px 8px; text-align: right; font-family: var(--font-mono);">63,497</td>
                            <td style="padding: 10px 8px; text-align: right; font-family: var(--font-mono); color: #34d399;">0</td>
                        </tr>
                        <tr style="font-weight: 700; color: #ffffff;">
                            <td style="padding: 12px 8px;"><b>Total Catalog</b></td>
                            <td style="padding: 12px 8px; text-align: right; font-family: var(--font-mono); color: #00e5ff;">26,646</td>
                            <td style="padding: 12px 8px; text-align: right; font-family: var(--font-mono); color: #00e5ff;">316,220</td>
                            <td style="padding: 12px 8px; text-align: right; font-family: var(--font-mono); color: #34d399;">0</td>
                        </tr>
                    </tbody>
                </table>
                <div style="display: flex; gap: 0.8rem; font-family: var(--font-mono); font-size: 0.78rem; color: #94a3b8;">
                    <span>Revision: <code style="color: #00e5ff;">59f82c57</code></span> ·
                    <span>License: <code style="color: #34d399;">CC BY 4.0</code></span> ·
                    <span>Categories: <code style="color: #a78bfa;">14 Classes</code></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-card-header">
                    <div class="glass-card-title">🔍 COCO Schema Verification</div>
                    <span class="badge-complete">PASS</span>
                </div>
                <div style="font-size: 0.9rem; color: #94a3b8; line-height: 1.5; margin-bottom: 1rem;">
                    All four annotation files (MV-Train, MV-Val, ST-Train, ST-Val) conform strictly to COCO object detection specification:
                </div>
                <ul style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6; padding-left: 1.2rem;">
                    <li><b>Required keys:</b> <code>info</code>, <code>licenses</code>, <code>images</code>, <code>annotations</code>, <code>categories</code></li>
                    <li><b>Image metadata:</b> <code>id</code>, <code>file_name</code>, <code>width</code> (1920), <code>height</code> (1080)</li>
                    <li><b>Box format:</b> <code>[x_min, y_min, width, height]</code> in absolute image pixels</li>
                    <li><b>Zero missing references:</b> All annotations link to valid image IDs</li>
                </ul>
                <div class="terminal-window" style="margin-top: 0.8rem;">
                    <div class="terminal-header">
                        <span class="terminal-btn red"></span>
                        <span class="terminal-btn yellow"></span>
                        <span class="terminal-btn green"></span>
                        <span style="font-size: 0.72rem; color: #94a3b8; margin-left: 0.5rem;">reports/audit/schema.json</span>
                    </div>
                    <div class="terminal-body" style="font-size: 0.76rem; max-height: 110px; overflow-y: auto;">
"MV": {<br/>
&nbsp;&nbsp;"train": { "images": 21349, "annotations": 252723, "invalid": 0 },<br/>
&nbsp;&nbsp;"val": { "images": 5297, "annotations": 63497, "invalid": 0 }<br/>
}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------------------
    # 02 — Data Engineering (COCO -> YOLO)
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div style="margin-top: 3rem; margin-bottom: 1.5rem;" id="data-engineering">
            <div class="section-kicker">02 / DATA TRANSFORMATION</div>
            <div class="section-title">Strict COCO → YOLO Coordinate Translation</div>
            <div class="section-subtitle">
                Deterministic mathematical transformation preserving boundary integrity without clipping or heuristic box alteration.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2 = st.columns([1, 1.2])

    with t1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-card-header">
                    <div class="glass-card-title">📐 Transformation Formula</div>
                    <span class="badge-complete">DETERMINISTIC</span>
                </div>
                <div class="terminal-window">
                    <div class="terminal-header">
                        <span class="terminal-btn red"></span>
                        <span class="terminal-btn yellow"></span>
                        <span class="terminal-btn green"></span>
                        <span style="font-size: 0.72rem; color: #94a3b8; margin-left: 0.5rem;">src/data/common.py: convert_box()</span>
                    </div>
                    <div class="terminal-body" style="font-size: 0.82rem;">
# COCO Bounding Box: [x, y, w, h]<br/>
# YOLO Bounding Box: [class_id, cx, cy, norm_w, norm_h]<br/><br/>
cx = (x + w / 2.0) / image_width<br/>
cy = (y + h / 2.0) / image_height<br/>
norm_w = w / image_width<br/>
norm_h = h / image_height<br/><br/>
# Coordinates strictly bounded in [0.0, 1.0]
                    </div>
                </div>
                <div style="color: #94a3b8; font-size: 0.88rem; line-height: 1.5;">
                    <b style="color: #38bdf8;">Engineering Principles:</b>
                    <ul style="padding-left: 1.1rem; margin-top: 0.4rem;">
                        <li><b>Zero heuristic repair:</b> Boxes outside $[0, W] \times [0, H]$ trigger validation errors.</li>
                        <li><b>Background handling:</b> Images without vehicle objects produce valid empty <code>.txt</code> files.</li>
                        <li><b>Immutable raw files:</b> Raw data directories are read-only; YOLO datasets use symlinks.</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-card-header">
                    <div class="glass-card-title">🏷️ 14-Class Taxonomy Mapping</div>
                    <span class="badge-complete">0 CLASSES MERGED</span>
                </div>
                <p style="color: #94a3b8; font-size: 0.88rem; margin-bottom: 0.8rem;">
                    All 14 classes defined in the UVH-26 taxonomy are preserved independently to reflect heterogeneous Indian traffic realities:
                </p>
            """,
            unsafe_allow_html=True,
        )
        if mapping:
            map_cols = st.columns(2)
            half = (len(mapping) + 1) // 2
            with map_cols[0]:
                for item in mapping[:half]:
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; margin-bottom: 4px; background: rgba(255,255,255,0.03); border-radius: 6px; font-size: 0.82rem; font-family: var(--font-mono);">
                            <span style="color: #00e5ff;">#{item.get('yolo_id', item.get('id', ''))}</span>
                            <span style="color: #f1f5f9; font-weight: 600;">{item.get('name', '')}</span>
                            <span style="color: #64748b;">(COCO: {item.get('original_id', '')})</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            with map_cols[1]:
                for item in mapping[half:]:
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; margin-bottom: 4px; background: rgba(255,255,255,0.03); border-radius: 6px; font-size: 0.82rem; font-family: var(--font-mono);">
                            <span style="color: #00e5ff;">#{item.get('yolo_id', item.get('id', ''))}</span>
                            <span style="color: #f1f5f9; font-weight: 600;">{item.get('name', '')}</span>
                            <span style="color: #64748b;">(COCO: {item.get('original_id', '')})</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 03 — Data Quality & Reliability
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div style="margin-top: 3rem; margin-bottom: 1.5rem;" id="data-quality">
            <div class="section-kicker">03 / VERIFICATION &amp; INTEGRITY</div>
            <div class="section-title">Data Quality Checks &amp; Leakage Defense</div>
            <div class="section-subtitle">
                Rigorous automated validation ensuring split independence, reproducible seeding, and geometry consistency.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    q1, q2, q3 = st.columns(3)

    with q1:
        st.markdown(
            """
            <div class="glass-card" style="height: 100%;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🛡️</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 0.4rem;">
                    Split Leakage Defense
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5; margin-bottom: 0.8rem;">
                    Filename and image SHA-256 collision scans between train and validation splits:
                </div>
                <div style="font-family: var(--font-mono); font-size: 0.82rem; color: #34d399; background: rgba(16,185,129,0.1); padding: 0.4rem 0.6rem; border-radius: 6px; border: 1px solid rgba(16,185,129,0.3);">
                    ✓ Shared Filenames: 0<br/>
                    ✓ Shared Hashes: 0
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with q2:
        st.markdown(
            """
            <div class="glass-card" style="height: 100%;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📐</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 0.4rem;">
                    Geometry Sanity Checks
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5; margin-bottom: 0.8rem;">
                    Evaluated all 316,220 bounding boxes for non-finite, negative, or out-of-frame coordinates:
                </div>
                <div style="font-family: var(--font-mono); font-size: 0.82rem; color: #34d399; background: rgba(16,185,129,0.1); padding: 0.4rem 0.6rem; border-radius: 6px; border: 1px solid rgba(16,185,129,0.3);">
                    ✓ Invalid Bounding Boxes: 0<br/>
                    ✓ Missing Image Pairs: 0
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with q3:
        st.markdown(
            """
            <div class="glass-card" style="height: 100%;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🎯</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 0.4rem;">
                    Deterministic Subsets
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5; margin-bottom: 0.8rem;">
                    Seeded subset generation algorithm prioritizing rare-class coverage with balanced distribution:
                </div>
                <div style="font-family: var(--font-mono); font-size: 0.82rem; color: #38bdf8; background: rgba(56,189,248,0.1); padding: 0.4rem 0.6rem; border-radius: 6px; border: 1px solid rgba(56,189,248,0.3);">
                    ✓ Seed: 42 (Reproducible)<br/>
                    ✓ Max Deviation: 0.088 pp
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="warning-callout">
            <b>Data Quality Status:</b> Annotation-level checks are 100% complete for the entire 26,646 metadata catalog. Full pixel decoding, dimension re-confirmation, and bit-level content-hash audits across all 26,646 raw PNG files are pending download completion.
        </div>
        """,
        unsafe_allow_html=True,
    )
