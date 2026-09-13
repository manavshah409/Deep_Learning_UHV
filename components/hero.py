"""Hero and status banner component."""

import streamlit as st


def render_hero():
    """Render the high-impact hero header and abstract computer vision simulation."""
    st.html(
        """
        <div class="hero-container">
            <div class="hero-glow-orb"></div>
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 2rem;">
                <div style="flex: 1 1 500px; min-width: 320px;">
                    <div class="hero-tag">
                        <span>⚡</span>
                        <span>COMPUTER VISION RESEARCH · PHASE 1 REVIEW</span>
                    </div>
                    <div class="hero-title">
                        Real-Time Vehicle Detection<br/>for Indian Urban Roads
                    </div>
                    <div class="hero-subtitle">
                        Phase 1 — Reproducible Data Pipeline &amp; YOLOv8n Baseline
                    </div>
                    <div class="hero-meta">
                        <div class="hero-meta-item">
                            <span>👨‍💻</span>
                            <span>Presented by <b>Manav Shah</b></span>
                        </div>
                        <div class="hero-meta-item">
                            <span>📅</span>
                            <span><b>13 September 2026</b></span>
                        </div>
                        <div class="hero-meta-item">
                            <span>🏛️</span>
                            <span>Dataset: <b>IISc AIM UVH-26</b></span>
                        </div>
                    </div>
                </div>
                <div style="flex: 1 1 340px; max-width: 420px; width: 100%;">
                    <div class="cv-radar-box">
                        <div class="cv-grid-line"></div>
                        <!-- Abstract simulated bounding boxes -->
                        <div class="cv-target-box" style="top: 25px; left: 30px; width: 120px; height: 75px;">
                            Two-wheeler
                        </div>
                        <div class="cv-target-box" style="bottom: 35px; right: 35px; width: 140px; height: 85px; border-color: #8b5cf6; color: #c4b5fd;">
                            Three-wheeler
                        </div>
                        <div class="cv-target-box" style="top: 70px; right: 40px; width: 90px; height: 60px; border-color: #3b82f6; color: #93c5fd;">
                            Hatchback
                        </div>

                        <div style="margin-bottom: 0.8rem; font-family: var(--font-mono); font-size: 0.72rem; color: #64748b; z-index: 2; text-align: center;">
                            ILLUSTRATIVE BOXES · REVISION 59f82c57
                        </div>
                        <div class="tech-badge-cloud">
                            <span class="tech-pill">UVH-26 MV</span>
                            <span class="tech-pill">YOLOv8n</span>
                            <span class="tech-pill">14 CLASSES</span>
                            <span class="tech-pill">316,220 OBJECTS</span>
                            <span class="tech-pill">APPLE MPS</span>
                            <span class="tech-pill">PYTEST</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Status Banner -->
        <div class="status-banner">
            <div class="status-headline">
                <div class="status-dot"></div>
                <div>
                    <span style="color: #34d399;">PHASE 1 STATUS:</span>
                    <span style="margin-left: 0.4rem; color: #ffffff;">Proper Baseline Evaluated</span>
                </div>
            </div>
            <div class="status-sublabel">
                UVH-26 MV 8,000/2,000 subset · Video real-time performance is not established
            </div>
        </div>
        """,
    )
