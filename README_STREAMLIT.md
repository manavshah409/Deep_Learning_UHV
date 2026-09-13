# Artifact-based Phase 1 dashboard

The Streamlit dashboard reads the completed proper 30-epoch baseline provenance, fresh best-checkpoint evaluation, per-class scores, curves and synchronized MPS timing from saved artifacts. The historical smoke pilot remains explicitly labelled separately.

```bash
python -m pip install -r requirements.txt -r requirements-dashboard.txt
python -m streamlit run app.py
```

Launch from the project root or extracted faculty package. No training or download starts on launch. The optional test button runs pytest. Dataset images/weights are deliberately absent from the portable ZIP; metrics, figures and PDF still work. The animated hero boxes are illustrative, not model predictions.

Present the proper baseline section first, then class results, timing definitions, audit boundaries and limitations. Full 26,646-image catalogue audit does not imply full pixel verification. All selected 10,000 images passed integrity; unselected acquisition remains incomplete. End-to-end timing is still-image file-to-result processing, not video capture/display FPS. No Phase 2 training is included.
