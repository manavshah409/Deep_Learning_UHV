# Phase 1 — Foundation, UVH-26 audit, YOLO preparation and baseline

## Project and objective

**Real-Time Vehicle Detection and Traffic Analytics for Indian Urban Roads Using YOLOv8 and the UVH-26 Dataset**

Fourth-year undergraduate Deep Learning / Computer Vision / Intelligent Transportation Systems project. Phase 1 must establish a reproducible data pipeline and complete genuine YOLOv8n transfer learning with separate validation. This report currently records work in progress; it does not establish Phase 1 completion.

Indian traffic presents heterogeneous vehicles, high density, occlusion and variation in scale and appearance. The project examines localization and India-specific categorization as a foundation for later vehicle counting and density analytics.

## Dataset provenance and source

- Official dataset: https://huggingface.co/datasets/iisc-aim/UVH-26
- Pinned revision: `59f82c57821e8a54dc40bc1f42e83909dbad0b70`.
- Acquisition began 2026-09-11; image download is incomplete.
- Hub metadata: ungated; public license CC BY 4.0; published total file size approximately 90 GB.
- Hub file inventory: 26,653 entries, including 26,646 PNG image paths, four consensus JSON files and repository metadata/card files.
- Annotation checksums, embedded licenses and field schemas: `reports/audit/schema.json`.
- Citation: Sharma et al., *Towards Image Annotations and Accurate Vision Models for Indian Traffic, Preliminary Dataset Release, UVH-26-v1.0*, IISc Technical Report, November 2025, https://doi.org/10.48550/arXiv.2511.02563.

Public dataset descriptions are distinguished from measured JSON counts below. Complete-dataset byte counts, decoding results and hashes remain pending. A separately audited 96-image pilot is documented below.

## Environment and repository audit

The workspace initially contained only an empty Git repository: no commits, no uncommitted files, branch `master`, and no remote. After the user supplied its URL, the empty remote was verified and configured as `origin`: `https://github.com/manavshah409/Deep_Learning_UHV.git`. No history was rewritten.

Detected hardware: Apple M5 MacBook Pro, arm64, 10 CPU cores and 24 GiB memory. Available storage at setup was approximately 767–768 GiB. System Python was 3.14.7; a separate installed Python 3.12.14 was used to create `.venv`. PyTorch, Ultralytics and Hugging Face Hub were initially absent. Exact installed packages are frozen in `requirements.txt`.

PyTorch 2.14.0 reports MPS built. The execution sandbox hid MPS; an authorized unsandboxed check reported MPS available and successfully calculated a tensor sum on the GPU. No CUDA is assumed. CPU fallback is implemented.

## Observed structure and schema

Actual image paths use `UVH-26-Train/data/<shard>/*.png` and `UVH-26-Val/data/<shard>/*.png`; the dataset card's illustrative `images/` directory is not the downloaded layout. Annotation `file_name` values are PNG basenames.

Each JSON has top-level `info`, `licenses`, `images`, `annotations` and `categories`. Image records include `id`, `file_name`, `height`, `width`, and `license`. Annotations include `id`, `image_id`, `category_id`, `bbox`, `area`, `iscrowd`, and `segmentation`. Representative records were inspected locally; field-presence counts are saved in the schema audit without copying object annotations into eligible reports. Boxes are COCO `[x, y, width, height]` in original image pixels.

No test split is present in the inspected repository inventory. The inspected image records provide no camera, sequence or timestamp fields. Pixel/filename independence cannot establish independence of nearby frames or views.

## Consensus policy and exact annotation counts

Use Majority Voting (MV) only. This follows the requested default and its downloaded annotations pass geometry/reference checks. Do not combine STAPLE with MV. STAPLE comparison is a later controlled experiment.

| Variant | Split | Images listed | Object instances |
|---|---|---:|---:|
| MV | Train | 21,349 | 252,723 |
| MV | Val | 5,297 | 63,497 |
| STAPLE | Train | 17,387 | 226,239 |
| STAPLE | Val | 4,339 | 57,163 |

MV total: **26,646 images and 316,220 objects**. STAPLE total: **21,726 images and 283,402 objects**. The annotation variants do not list the same number of images, despite sharing the repository image pool. Do not assume equivalent evaluation denominators.

Exact MV files selected:

- `UVH-26-Train/UVH-26-MV-Train.json`
- `UVH-26-Val/UVH-26-MV-Val.json`

Available STAPLE counterparts use `UVH-26-ST-Train.json` and `UVH-26-ST-Val.json`.

## Annotation integrity findings

The executed annotation-only audit found zero duplicate image IDs within each split, duplicate annotation IDs, duplicate filenames, missing mandatory fields, unknown categories, missing image references or invalid bounding boxes. Zero images have no annotations. All 316,220 boxes are finite, positive-area and within their declared image bounds. There is no shared image ID or filename across train and validation.

**Not yet established:** image readability, actual dimensions matching annotation metadata, complete image inventory, byte-identical train/validation leakage. The annotation audit explicitly sets `image_audit_executed: false`; an empty hash-overlap array at this stage is not evidence that hash leakage was checked.

## Class mapping and EDA

Original numeric IDs are sorted and mapped to contiguous zero-based YOLO IDs in configs/class_mapping.yaml and reports/tables/class_mapping.csv. Names remain unchanged; no classes are merged. `Others` is retained provisionally because it is a valid supplied category; visual quality review remains required.

| Original ID | YOLO ID | Name | MV instances |
|---:|---:|---|---:|
| 1 | 0 | Hatchback | 30,290 |
| 2 | 1 | Sedan | 15,950 |
| 3 | 2 | SUV | 13,175 |
| 4 | 3 | MUV | 6,523 |
| 5 | 4 | Bus | 9,286 |
| 6 | 5 | Truck | 13,011 |
| 7 | 6 | Three-wheeler | 52,428 |
| 8 | 7 | Two-wheeler | 149,730 |
| 9 | 8 | LCV | 17,345 |
| 10 | 9 | Mini-bus | 873 |
| 11 | 10 | Tempo-traveller | 1,680 |
| 12 | 11 | Bicycle | 3,391 |
| 13 | 12 | Van | 2,186 |
| 14 | 13 | Others | 352 |

Source: executed EDA and `reports/tables/eda_class_distribution.csv`. Class share percentages by split are in that CSV. Two-wheelers dominate; `Others`, Mini-bus and Tempo-traveller are rare, so aggregate metrics can obscure poor minority-class performance.

Objects per image: mean 11.8674, median 10, minimum 1 and maximum 66. Box area median is 14,688 original pixels². Using COCO-style area boundaries on original-resolution boxes gives 4,041 small, 112,887 medium and 199,292 large instances. These counts are not equivalent to object sizes after resizing to 640 pixels.

Generated figures cover split counts, classes, objects per image, metadata resolution, box width/height/area/aspect ratios and object centre spatial distribution. The notebook reads these generated artifacts; its cells have not been executed as a separate notebook run.

## Conversion and validation procedure

Implementation is present; full-dataset execution is pending download and image audit. Raw images/annotations remain unchanged. Conversion builds separate label files and relative image symlinks under `uvh26_mv_yolo_v1`. For `[x,y,w,h]`, normalized labels are `[(x+w/2)/W,(y+h/2)/H,w/W,h/H]` plus the mapped class ID.

Repair policy: reject invalid boxes and record reasons; zero clipped and repaired boxes by design. Unknown references/categories and structural ambiguity block conversion. Empty retained objects yield an empty label file. Ordering is deterministic. Annotation hashes and policy identify the processed version; differing or incomplete versions cannot be overwritten. The manifest records source image, label, split and object count.

YOLO validation checks five fields per row, integral class range, finite normalized coordinates, positive sizes, box extent, image-label pairing, manifest counts and category frequencies. It then calls Ultralytics' dataset resolver and scans both splits. This full validation has not run yet.

Visual verification will render at least 20 seed-42 random annotated images plus class coverage, dense/sparse and small-object cases. Occlusion is a visual judgment, not a field in these annotations. Training is gated on a recorded manual visual review.

## Subset decision, baseline and smoke test

Proposed baseline: 8,000 train / 2,000 validation images, seed 42, preserving official boundaries. Selection covers rare classes then performs seeded random fill. Immutable image-ID manifests and full/subset class shares will quantify representativeness. Final feasibility will be assessed from the one-epoch smoke run; no reduced dataset results will be called full-dataset results.

Planned model: official COCO-pretrained YOLOv8n. Planned settings: 640 image size, 30 epochs, batch 8, MPS, 4 workers, seed 42, deterministic mode, patience 10, optimizer auto, no cosine schedule, close mosaic 5, AMP requested, plots and checkpoint saving enabled. Smoke settings explicitly change to one epoch, zero workers and close mosaic zero on 64 train / 32 val images. Configuration and device overrides are recorded in run provenance.

Pretrained inference is a separate smoke test and cannot establish UVH-26 baseline accuracy. Proper training has not started. Finite-loss checks and checkpoint existence are required after smoke training.

## Evaluation, qualitative analysis and limitations

| Required outcome | Current evidence |
|---|---|
| Completed proper baseline | Pending |
| Training duration / best epoch / stopping status | Not measured |
| Precision / recall / F1 | Not measured |
| mAP@0.5 / mAP@0.5:0.95 | Not measured |
| Per-class AP, precision, recall | Not measured |
| Training curves / confusion matrix / PR and F1 curves | Pending training and validation |
| Fine-tuned prediction examples / error analysis | Pending |
| Fine-tuned inference time / FPS / weight checksum | Not measured |

Evaluation code measures the best checkpoint on validation data, records macro per-class F1, class metrics, speed and parameter count, and copies genuine plots. Reported FPS will be validation-batch throughput, with inference-only and pipeline timing distinguished; it excludes end-to-end application overhead.

Diagnostic matching uses class-agnostic greedy IoU matching at 0.5 with confidence 0.1 to identify unmatched objects and class confusions. This is for qualitative sampling and is not the COCO AP evaluator. Occlusion-related conclusions require visual evidence. No best/worst class claims are made before evaluation.

## Tests, commands and reproduction

The saved `reports/audit/pytest.txt` currently records **47 passed**. Tests use synthetic images and annotations; no dataset images are committed. Coverage includes path resolution, JSON/schema loading, IDs, categories, normalization, invalid boxes, label validation, backgrounds, manifest/split integrity, coverage-aware sampling, idempotent conversion and hash-leakage blocking. Diagnostic matching tests check one-to-one matching and empty detections.

Executed project commands include:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip freeze > requirements.txt
.venv/bin/hf download iisc-aim/UVH-26 --repo-type dataset --revision 59f82c57821e8a54dc40bc1f42e83909dbad0b70 --local-dir data/raw/UVH-26 --max-workers 8
# Resumed with 32 workers and HF_XET_HIGH_PERFORMANCE=1.
.venv/bin/python -m src.data.inspect_uvh26
.venv/bin/python -m src.data.validate_raw --annotations-only
.venv/bin/python -m src.data.eda
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src tests
```

Initial dependency installation used the project libraries listed in the request; the resulting exact dependency versions are in `requirements.txt`. Raw transfer logs and Hub metadata remain in ignored `data/interim/`. Ruff installation initially failed on network errors, then succeeded after resuming; source formatting and undefined/unused-name checks were executed. README documents the remaining conversion, validation, training, evaluation and inference commands in execution order.

## Files and Git checkpoint

Created configuration/environment files, package modules under `src/`, synthetic tests, EDA notebook, genuine audit/EDA outputs, README, CHANGELOG, data/model policies and this report. No existing user files were overwritten. No dataset files, machine-specific local path config or weights are eligible for Git.

Commit: pending the requested baseline-completion checkpoint. Branch: `master`. Remote: verified `origin` pointing to the user-supplied repository. Push: not attempted. Phase 1 remains incomplete.

## Recommended Phase 2 objective

After a verified Phase 1 baseline, improve minority-class and small-vehicle recall using controlled imbalance/augmentation experiments, then compare MV and STAPLE on a carefully aligned evaluation population. Add vehicle counting only after detector errors and measured inference throughput are understood.

## Early visual review and pretrained inference update

Pretrained YOLOv8n inference executed on MPS and produced nine COCO-class detections on one downloaded image; the saved speed is a single-image smoke measurement and is not a benchmark. See `reports/audit/pretrained_inference.json`.

Thirty-two early training images were manually inspected as contact sheets, covering all 14 categories, dense/sparse scenes, small objects and occlusion. No systematic scaling/orientation error was observed. Original annotation quality is imperfect: image 355 lacks a foreground motorbike box, image 20275 has an excessively tall truck box, while source verification confirmed the large box in image 834 is correctly assigned to MUV (annotation 7986); reused preview colors had made it ambiguous. Image 1890 contains a construction vehicle marked Others, supporting retention. These findings do not certify the yet-unbuilt full YOLO dataset. Evidence and sampling limitations are in `reports/audit/early_visual_review.json`; raw images and labels were not altered.

## Executed smoke-training update

The separately downloaded and audited pilot contains 64 official training images and 32 official validation images. `reports/audit/smoke_data_audit.json` records decoding, actual dimensions, SHA-256 and label frequencies. Ultralytics scanned both splits with zero corrupt images, and first samples loaded as 3×640×640 tensors (`smoke_loading.json`). Four pilot validation annotation previews were additionally inspected with readable, distinct class colors.

The one-epoch MPS run `yolov8n_uvh26_mv_smoke_seed42` completed with finite training and validation losses and saved best/last checkpoints. Wall duration including setup and final validation: 35.7282 seconds. The epoch CSV elapsed field was 12.7281 seconds. Effective selected optimizer in the executed log: AdamW, learning rate 0.000556, momentum 0.9. Peak displayed GPU memory was approximately 2.17 GB. MPS warned that scatter-reduce and index-put-with-accumulate do not have deterministic implementations; exact numerical reproducibility is not guaranteed despite deterministic settings.

The saved epoch CSV reports precision 0.02237, recall 0.00238, mAP50 0.00063 and mAP50–95 0.00043. These are **smoke-only**, rounded CSV values, not proper baseline results. All 32 saved pilot predictions had zero detections at confidence 0.01; two images were manually inspected. One source image (100173.png) is heavily blurred/redacted. The one-epoch checkpoint is not a useful detector.

Smoke checkpoint: `runs/yolov8n_uvh26_mv_smoke_seed42/weights/best.pt`, 6,224,938 bytes, SHA-256 `4808f80e387a019bee01a3dceaea255320f125c1c8161bd53d85b4c1e71c9753`. Provenance and actual epoch metrics are under `reports/tables/`. No smoke checkpoint is being represented as the required completed baseline.

Based on steady smoke batches around 0.6 seconds, the proposed 8,000-image, 30-epoch training is estimated to require several hours plus validation and startup. This is an extrapolation, not measured baseline duration. The full dataset download failed in Xet after DNS/network errors; a resumable HTTP fallback is in progress. Completed raw files are retained.

## Acquisition blocker

Full image acquisition is blocked by repeated external network failures: Hugging Face Xet DNS/reconstruction errors, followed by HTTP responses ending before their declared length and repeated read timeouts. PyPI also failed DNS resolution during an optional formatter download. The project stop condition for dataset acquisition applies. Completed raw files and resumable transfer metadata are preserved. No full-dataset conversion or proper baseline completion is claimed; no commit or push was attempted because the requested pre-commit baseline-completion condition is unmet. See `reports/audit/acquisition_status.json` for the final observed local image inventory.

## Faculty progress deliverable

Prepared a five-page PDF, speaking script, offline evidence demo and portable project ZIP for a faculty progress review. The pack labels all smoke results explicitly and does not claim completed baseline accuracy. Source/tests/configuration and generated evidence are included; raw images, annotations, weights, local paths and credentials are excluded. The 8,000/2,000 subset selection contains 94,484/24,342 objects; maximum class-share difference from the parent split is 0.306231 percentage points. Full and prioritized downloads are running after resumption.

## Authorized subset recovery (12 September 2026)

The user explicitly redefined completion as a UVH-26 Majority Voting **8,000/2,000 subset baseline**. Full annotation-catalog checks remain applicable, but acquisition and pixel auditing of unselected images are no longer prerequisites. Full acquisition is retained as a separate future resumable task. No full-dataset training, full pixel-integrity or test-set claim is made.

The original 10,000 candidate images were fully decoded, dimension checked, hashed and checked for box validity against both metadata and actual dimensions. One mismatch was found: training ID 21818 (`803489.png`), 1620×1080 actual versus 1920×1080 metadata. Its three boxes all end within actual width (rightmost 1486.5). Raw-coordinate overlays roughly align the three-wheeler but miss both motorcycles; horizontal scaling improves the motorcycles but shifts the three-wheeler incorrectly. No interpretation consistently aligns every box. No existing YOLO label was present, so the third diagnostic is explicitly a hypothetical metadata-normalized roundtrip, not an existing converted label. Raw data remains unchanged; the image is quarantined from the baseline.

An initial exact-class-exposure replacement (training ID 5235, `341297.png`) passed structural checks but final visual review revealed severe gray artifacts. Validation ID 20260 (`986228.png`) also showed severe visual degradation despite successful PNG decoding. The first frozen candidate `baseline_seed42` is preserved and superseded before training; its review is recorded as failed. Version `baseline_seed42_v2` is being prepared with training ID 25440 (`233625.png`) and validation ID 6452 (`81395.png`). Both replacements exactly preserve the removed per-class object counts. All final images are re-audited before freezing. A near-mid-gray fraction filter is used only to reject replacement candidates; it is not claimed as a validated full-image corruption detector.

The baseline optimizer is explicitly AdamW with `lr0=0.000556`, `weight_decay=0.0005`, betas `(0.9, 0.999)`, `nbs=64`, `lrf=0.01`, three warm-up epochs and zero initial bias warm-up LR. This is a conservative documented baseline choice, not an optimum from a tuning study. Installed Ultralytics 8.4.146 accepts these settings; its explicit AdamW path preserves the requested learning rate. A startup callback verifies the actual optimizer and parameter-group learning rates before training. Requested workers remain four; the installed MPS/CPU trainer unconditionally uses zero workers. After warm-up, nominal gradient accumulation is eight at batch eight, giving nominal effective batch 64; warm-up accumulation varies.

Proper training, evaluation, latency measurement and the initial Git checkpoint remain pending. The final one-epoch preflight will use the final 8,000/2,000 manifests under its own experiment ID and is not the proper baseline.


## Initial repository checkpoint requested during training

The user requested committing all eligible current changes before training finalization. This supersedes the earlier timing restriction on the initial commit, but does not establish Phase 1 completion. The proper run is `yolov8n_uvh26_mv_baseline_seed42_v1`; its last observed saved epoch was 28 and epoch 29 was active. All 10,000 image and label hashes were reverified before launch. The final training subset has 94,609 objects (not the earlier snapshot count 94,484), and validation has 24,342. All 58 tests passed at the repository checkpoint. Final evaluation, latency, updated faculty deliverables and the revised Phase 2 gate remain outstanding.
