# Phase 1 Streamlit Research Showcase & Faculty Presentation App

Interactive, research-lab aesthetic Streamlit dashboard showcasing the verified Phase 1 engineering, data pipeline, exploratory data analysis, synthetic test suite, and smoke training pilot for the project:

> **Real-Time Vehicle Detection and Traffic Analytics for Indian Urban Roads Using YOLOv8 and the UVH-26 Dataset**  
> *Author: Manav Shah · Date: 11 September 2026*

---

## Quick Start

Activate the virtual environment and launch Streamlit:

```bash
source .venv/bin/activate
streamlit run app.py
```

Or run directly via the virtual environment binary:

```bash
./.venv/bin/streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## Key Features

1. **Academic Honesty & Integrity Guarantee:**
   - Clearly marks the one-epoch smoke run as pipeline verification (0 detections at conf 0.01).
   - Never fabricates or extrapolates final baseline metrics or FPS claims.
   - Shows "In Progress / Pending" for unexecuted benchmarks.

2. **Interactive EDA & Visualizations:**
   - Interactive Plotly chart of 14 vehicle classes across 316,220 objects.
   - Objects-per-image density breakdown and 2D normalized object center heatmaps.
   - Tabbed gallery of local figure artifacts from `reports/figures/`.

3. **Live Test Runner:**
   - Interactive button on the dashboard allowing real-time execution of the 49-test synthetic test suite via `pytest`.

4. **Presentation Mode:**
   - Sidebar toggle that scales fonts and optimizes card layouts for high-resolution classroom projectors and faculty reviews.

5. **Local Artifact Auto-Discovery:**
   - Safely parses `reports/audit/`, `reports/tables/`, `reports/figures/`, `configs/`, `runs/`, and `tests/` without crashing if individual files are unavailable.

---

## File Structure

```
├── app.py                      # Main Streamlit application entrypoint
├── components/                 # Modular visual components
│   ├── __init__.py
│   ├── hero.py                 # Hero header, CV radar simulation & status banner
│   ├── metrics.py              # 8 key verified quantity cards
│   ├── pipeline.py             # 9-stage interactive engineering pipeline
│   ├── dataset.py              # Sections 01 (Audit), 02 (Conversion), 03 (Quality)
│   ├── eda.py                  # Section 04 (EDA) & Section 05 (Annotation Review)
│   ├── training.py             # Section 06 (Architecture) & Section 07 (Smoke Pilot)
│   ├── testing.py              # Section 08 (Synthetic Tests) & Section 09 (Reproducibility)
│   └── status.py               # Section 10 (What Remains) & Future Phase 2/3 Roadmap
├── utils/                      # Safe data & artifact loaders
│   ├── __init__.py
│   ├── artifact_loader.py      # Path resolution, safe JSON/YAML/CSV/hash loaders
│   └── data_loader.py          # Typed getters for metrics, splits, and tables
├── assets/
│   └── styles.css              # Custom AI research lab dark theme stylesheet
└── README_STREAMLIT.md         # Documentation
```

---

## Faculty Presentation Walkthrough

When presenting to evaluators:
1. **Hero & Status Banner:** Point out *"Engineering Pipeline Operational — Baseline in Progress"*.
2. **Key Metrics:** Highlight 26,646 images and 316,220 objects across 14 preserved classes.
3. **Data Engineering:** Explain strict COCO $\to$ YOLO translation without heuristic box clipping.
4. **Exploratory Data Analysis:** Show the heavy-tailed class imbalance (Two-wheelers 47.35% vs Others 0.11%).
5. **Annotation Review:** Walk through the contact sheets and explain honest source annotation limitations.
6. **Smoke Test Pilot:** Show the 1-epoch execution telemetry and explain that it proves pipeline correctness.
7. **Test Suite:** Click **"Run Pytest Suite"** live to demonstrate 49 passing tests.
8. **Roadmap & Phase 2/3:** Conclude with the planned 30-epoch 8k/2k baseline and downstream tracking objectives.
