# Reproduce new data evaluation

Launch from repository root:

```bash
.venv/bin/python -m streamlit run dashboard/app.py
```

Choose New Data Evaluation and the Evaluate new images workspace. Validate the dataset, review findings, then explicitly run the model. Install requirements-new-data.txt for inference and labeled saved-result reconstruction. CPU is default; MPS falls back to CPU only when unavailable. The E1 checkpoint must already exist at its frozen path and pass SHA-256 verification. Saved Evidence and Faculty Presentation modes require no weights. No new-data results exist until you supply data. Historical fallback is clearly labelled.

## Prepare uploads
Use simple unique base filenames: letters, numbers, space, dot, dash, underscore. JPG/PNG only. Maximum 100MB total, 500 images/100MP total decoded, 10MB/20MP per image. ZIP folders are flattened only after path and duplicate-name checks; never use duplicate stems. Do not supply project reserved data.

1. Unlabeled: ZIP images alone, or upload multiple images. Select unlabeled. No AP/P/R/F1, confusion, FP or FN are calculated. Prediction confidence is not accuracy.
2. YOLO: ZIP images plus one identically stemmed .txt file each. Each row is `class_id x_center y_center width height`, normalized to [0,1], boxes entirely inside image. IDs0–13 follow the exact ordered mapping below. Use empty TXT files for explicitly empty images. Missing labels are errors, not assumed empty.
3. COCO: ZIP images and annotations.json, or upload images plus that JSON. Use images[{id,file_name,width,height}], categories[{id,name}], annotations[{id,image_id,category_id,bbox:[x,y,width,height]}]. Category names match exact UVH names; numeric COCO IDs may differ. File names must be base names. Unique image/annotation/category IDs; correct dimensions; crowd/ignore annotations unsupported and rejected.

Mapping: Hatchback, Sedan, SUV, MUV, Bus, Truck, Three-wheeler, Two-wheeler, LCV, Mini-bus, Tempo-traveller, Bicycle, Van, Others.

Confirm dataset name/source, licence or permission, annotation format, independent annotation creation and prior use in training/tuning. Optional metadata JSON/YAML may carry additional notes, but cannot override required declarations or mapping. Optional known-image hash index is a JSON list of SHA-256 strings from an approved non-reserved source. Do not generate an index by reading the reserved split. No supplied index means overlap verification unavailable.

## Offline command
Create metadata.json with this structure (use truthful source/permission/declarations):

```json
{
  "name": "New permitted road images",
  "source": "Describe collection source",
  "permission": "Describe permission or licence",
  "format": "yolo",
  "independent_annotations": true,
  "used_for_training_or_tuning": false,
  "names": ["Hatchback","Sedan","SUV","MUV","Bus","Truck","Three-wheeler","Two-wheeler","LCV","Mini-bus","Tempo-traveller","Bicycle","Van","Others"]
}
```

```bash
.venv/bin/python scripts/evaluate_new_data.py \
  --zip /path/to/new_images.zip --metadata /path/to/metadata.json \
  --device cpu --bootstrap 100 --output runs/new_data
```

Use `format: unlabeled` for images-only, or `format: coco` for COCO. A more expensive offline interval calculation uses `--bootstrap 1000` (reruns evaluation). All paths above are placeholders, not machine configuration. A unique run directory contains evidence.zip, with result/config/audit/metadata/environment/metrics/timing/records JSON, labeled CSV tables and COMPLETE or FAILED receipt. Browser downloads have the same structure. No images or checkpoint are exported. Do not commit bundles.

Extract result.json from your own saved bundle and load it through New Data Evaluation to present metrics without weights. Imported metrics are recomputed from records; provenance remains user-supplied and images are absent. Use Clear session to discard in-memory images and results. Browser uploads may remain in Streamlit widget memory until the session closes or widget is reset.

## Tests
```bash
.venv/bin/python -m pytest tests/test_new_data_dashboard.py tests/test_faculty_dashboard.py -q
.venv/bin/python -m pytest -q
.venv/bin/ruff check dashboard/new_data dashboard/app.py scripts/evaluate_new_data.py tests/test_new_data_dashboard.py tests/test_faculty_dashboard.py
```

See the technical report for metric denominators, matching, timing scope, reliability assumptions and unsupported annotations. No model training is required to use or demonstrate the dashboard.
