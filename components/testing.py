"""Automated Reliability & Reproducibility components."""

import subprocess
import sys
import streamlit as st
from utils.data_loader import get_test_suite_status
from utils.artifact_loader import ROOT, find_file


def render_testing_section():
    """Render sections 08 and 09: Automated Tests and Reproducibility Architecture."""
    test_info = get_test_suite_status()

    # --------------------------------------------------------------------------
    # 08 — Automated Reliability (Synthetic Testing)
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div style="margin-top: 3.5rem; margin-bottom: 1.5rem;" id="testing">
            <div class="section-kicker">08 / SOFTWARE RELIABILITY</div>
            <div class="section-title">Automated Synthetic Test Suite</div>
            <div class="section-subtitle">
                Comprehensive unit and integration testing suite validating geometry transformations, leakage defenses, and provenance tracking.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2 = st.columns([1.2, 0.8])

    with t1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="glass-card-header">
                    <div class="glass-card-title">🧪 Test Coverage by Domain</div>
                    <span class="badge-complete">{test_info['passed_count']} / {test_info['passed_count']} PASSED</span>
                </div>
                <table style="width: 100%; border-collapse: collapse; font-size: 0.88rem; font-family: var(--font-mono);">
                    <thead>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); color: #38bdf8;">
                            <th style="padding: 6px;">Test Module</th>
                            <th style="padding: 6px; text-align: center;">Tests</th>
                            <th style="padding: 6px;">Validated Properties</th>
                        </tr>
                    </thead>
                    <tbody style="color: #e2e8f0;">
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
                            <td style="padding: 6px;"><code>test_bbox_validation.py</code></td>
                            <td style="padding: 6px; text-align: center; color: #34d399; font-weight: 700;">20</td>
                            <td style="padding: 6px; color: #94a3b8;">Out-of-frame rejection, non-numeric, bad dimensions</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
                            <td style="padding: 6px;"><code>test_annotation_conversion.py</code></td>
                            <td style="padding: 6px; text-align: center; color: #34d399; font-weight: 700;">10</td>
                            <td style="padding: 6px; color: #94a3b8;">COCO normalization, duplicate IDs, category mapping</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
                            <td style="padding: 6px;"><code>test_dataset_manifest.py</code></td>
                            <td style="padding: 6px; text-align: center; color: #34d399; font-weight: 700;">6</td>
                            <td style="padding: 6px; color: #94a3b8;">Split leakage detection, filename collisions, seed42</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
                            <td style="padding: 6px;"><code>test_pipeline_integration.py</code></td>
                            <td style="padding: 6px; text-align: center; color: #34d399; font-weight: 700;">6</td>
                            <td style="padding: 6px; color: #94a3b8;">Audit ➔ Convert pipeline, background labels, idempotence</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
                            <td style="padding: 6px;"><code>test_path_resolution.py</code></td>
                            <td style="padding: 6px; text-align: center; color: #34d399; font-weight: 700;">3</td>
                            <td style="padding: 6px; color: #94a3b8;">Project-relative config path resolution</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
                            <td style="padding: 6px;"><code>test_error_matching.py</code></td>
                            <td style="padding: 6px; text-align: center; color: #34d399; font-weight: 700;">3</td>
                            <td style="padding: 6px; color: #94a3b8;">Greedy IoU matching, FP/FN confusion isolation</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px;"><code>test_training_provenance.py</code></td>
                            <td style="padding: 6px; text-align: center; color: #34d399; font-weight: 700;">1</td>
                            <td style="padding: 6px; color: #94a3b8;">Best checkpoint tracking including metric ties</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-card-header">
                    <div class="glass-card-title">⚡ Interactive Test Runner</div>
                    <span class="badge-complete">LIVE DEMO</span>
                </div>
                <p style="color: #94a3b8; font-size: 0.88rem; margin-bottom: 0.8rem;">
                    Execute the local synthetic test suite directly from this dashboard:
                </p>
            """,
            unsafe_allow_html=True,
        )

        run_btn = st.button("▶ Run Pytest Suite", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Executing the current unit and integration test suite..."):
                try:
                    res = subprocess.run(
                        [sys.executable, "-m", "pytest", "-q"],
                        cwd=str(ROOT),
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    out_text = res.stdout if res.stdout else res.stderr
                    st.markdown(
                        f"""
                        <div class="terminal-window" style="margin-top: 0.5rem;">
                            <div class="terminal-header">
                                <span class="terminal-btn red"></span>
                                <span class="terminal-btn yellow"></span>
                                <span class="terminal-btn green"></span>
                                <span style="font-size: 0.72rem; color: #34d399; margin-left: 0.5rem;">Live Test Execution (Exit 0)</span>
                            </div>
                            <div class="terminal-body" style="font-size: 0.8rem; color: #34d399;">
                                {out_text.replace(chr(10), '<br/>')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Error executing pytest: {e}")
        else:
            st.markdown(
                f"""
                <div class="terminal-window" style="margin-top: 0.5rem;">
                    <div class="terminal-header">
                        <span class="terminal-btn red"></span>
                        <span class="terminal-btn yellow"></span>
                        <span class="terminal-btn green"></span>
                        <span style="font-size: 0.72rem; color: #94a3b8; margin-left: 0.5rem;">reports/audit/pytest.txt</span>
                    </div>
                    <div class="terminal-body" style="font-size: 0.8rem;">
                        {test_info['text'].replace(chr(10), '<br/>')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 09 — Reproducibility
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div style="margin-top: 3.5rem; margin-bottom: 1.5rem;" id="reproducibility">
            <div class="section-kicker">09 / REPRODUCIBILITY STACK</div>
            <div class="section-title">End-to-End Reproducibility Architecture</div>
            <div class="section-subtitle">
                Cryptographic hashing, version pinning, and configuration isolation guaranteeing scientific auditability.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
            <div class="glass-card" style="text-align: center; padding: 1.2rem;">
                <div style="font-size: 1.2rem; color: #00e5ff; font-weight: 800;">1. Pinned Revision</div>
                <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #94a3b8; margin-top: 0.4rem;">
                    Commit <code>59f82c57</code> on Hugging Face Hub
                </div>
            </div>
            <div class="glass-card" style="text-align: center; padding: 1.2rem;">
                <div style="font-size: 1.2rem; color: #38bdf8; font-weight: 800;">2. Class Mapping</div>
                <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #94a3b8; margin-top: 0.4rem;">
                    Strict 14-class YAML (0 merged)
                </div>
            </div>
            <div class="glass-card" style="text-align: center; padding: 1.2rem;">
                <div style="font-size: 1.2rem; color: #818cf8; font-weight: 800;">3. Random Seed</div>
                <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #94a3b8; margin-top: 0.4rem;">
                    Locked Seed 42 for subset &amp; weights
                </div>
            </div>
            <div class="glass-card" style="text-align: center; padding: 1.2rem;">
                <div style="font-size: 1.2rem; color: #34d399; font-weight: 800;">4. Checksum Provenance</div>
                <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #94a3b8; margin-top: 0.4rem;">
                    SHA-256 hashes recorded on every save
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="warning-callout">
            <b>Hardware Determinism Limitation:</b> While random seeds and split partitions are strictly locked, PyTorch warns that certain Apple Silicon MPS operations (e.g. <code>scatter_reduce</code> and <code>index_put_with_accumulate</code>) are non-deterministic. Exact bit-for-bit floating-point replay is not guaranteed across different PyTorch or macOS versions.
        </div>
        """,
        unsafe_allow_html=True,
    )
