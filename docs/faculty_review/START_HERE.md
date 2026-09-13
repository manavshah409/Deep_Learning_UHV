# Faculty review package

Phase 1 proper YOLOv8n subset baseline: 30 completed epochs, best epoch 30, standalone 2,000-image validation and synchronized batch-one MPS timing.

1. Open `output/pdf/UVH26_Faculty_Progress_Report.pdf`.
2. Read `docs/faculty_review/PRESENTATION_SCRIPT.md`.
3. Run `python3 scripts/show_progress.py` from this folder; standard library only, no downloads.
4. For the dashboard, install `requirements.txt` and `requirements-dashboard.txt`, then run `python -m streamlit run app.py`.
5. With project dependencies installed, run `python -m pytest -q`.

The ZIP includes source, configuration, summarized audits, measured metrics, plots, tests and documentation. It excludes dataset images, generated labels, model weights, full runs and machine-local paths. Local inference/reverification of original data requires the original project machine; offline demonstration reads saved evidence.

These are subset validation results, not test-set or full-dataset results. Full catalogue annotation audit and 10,000-image local pixel audit are distinct. Unselected-image integrity remains incomplete. No real-time video, production-readiness or Phase 2 completion claim is made. Current entry-gate and Git delivery status are saved under `reports/audit/`.
