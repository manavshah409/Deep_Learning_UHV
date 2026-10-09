# UVH-26 faculty dashboard

A project dashboard for presenting saved research and evaluating new road images with the selected YOLOv8s detector. Historical results, newly measured results and unlabeled predictions are clearly separated.

## Start the dashboard

In the existing project environment:

```bash
.venv/bin/python -m streamlit run dashboard/app.py
```

For a fresh installation, create a Python3.12 environment. Choose **one** dependency set:

```bash
python3.12 -m venv .venv-demo
source .venv-demo/bin/activate
# Browse historical evidence without weights:
python -m pip install -r requirements-demo.txt
# OR install the full inference and labeled-evaluation dependencies:
python -m pip install -r requirements-new-data.txt
python -m streamlit run dashboard/app.py
```

The full environment includes pycocotools, required to recompute labeled saved results. The lightweight environment supports historical presentation and unlabeled saved summaries. No packages or model weights need to be downloaded during a presentation.

## Choose your workspace

- **Browse saved results:** present project evidence or reopen an exported result.json.
- **Evaluate new images:** choose images-only, YOLO labels or COCO annotations; complete source and permission details; click Validate dataset; review the audit; then Run predictions or Run evaluation.
- **Present to faculty:** follow the Next button through the overview, comparisons and methodology, or through metrics/errors/examples when a new result is available.

Changing an upload or declaration invalidates its previous validation. Inference never starts automatically. A failed run preserves the last successful result. The progress indicator identifies model loading, warm-up, image processing and metric calculation.

## Understand the screens

Executive Overview presents the current dataset, or clearly labelled historical evidence when none exists. Prediction Explorer applies the same filters to the gallery, overlays and CSV downloads. Missed objects have no prediction confidence; they are not removed by confidence filters. Image-level metric cards remain fixed at the evaluation protocol even when display filters change.

Images-only inputs cannot produce precision, recall, F1, AP or error labels. Confidence is not accuracy. Metric definitions are available under Dataset and Methodology. Performance reports still-image throughput, not live-video FPS.

## Model and data requirements

Actual inference requires the original E1 checkpoint at `runs/E1_yolov8s_uvh26_mv_640_seed42/weights/best.pt`. Its SHA-256 is verified before execution; no fallback model is substituted. Use CPU when MPS is unavailable or fails. Model loading is lazy and session-local for each evaluation.

JPG/PNG inputs: up to100MB total,500 images,100MP decoded total; maximum10MB/20MP per image. ZIP contents must have safe, unique base filenames. YOLO needs matching TXT files, including empty files for empty images; COCO needs annotations.json. See [dataset preparation](docs/reproducibility/NEW_DATA_EVALUATION.md).

Uploads stay in session memory; Clear uploaded data and session results resets the upload workflow. Downloads are explicit. Evidence bundles contain predictions, labels, manifests and metrics, but no source images or checkpoints. Imported image galleries are therefore unavailable; their records and recomputed metrics remain inspectable. Imported provenance/timing is user-supplied, not authenticated.

## Portable faculty package

Build an exclusive evidence-only archive:

```bash
.venv/bin/python scripts/package_faculty_demo.py --output deliverables/faculty_dashboard.zip
```

Extract it and run the startup command above after installing the selected dependency set. It includes all dashboard modules, theme, guides and allowlisted historical evidence. No dataset, model weights or generated prediction images are bundled. Reusing an existing ZIP name is refused.

## Limits

No training, fusion, video tracking, counting evaluation or public deployment occurs through the dashboard. AP confidence intervals and stratum AP are not implemented; supported strata report recall. No independent-test or production-readiness claim is made. Overlap verification needs a separately supplied approved hash index; do not use the reserved split. Historical results remain unchanged.
