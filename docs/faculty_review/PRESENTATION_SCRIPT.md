# Faculty presentation script: completed subset baseline

## 1. Problem and scope (45 seconds)

"Ma'am, our project detects 14 vehicle categories in Indian road scenes using YOLOv8n and IISc's UVH-26 Majority Voting annotations. Today I am presenting the completed Phase 1 subset baseline. This is object detection; traffic tracking and counting are future work."

## 2. Data work and integrity (60 seconds)

"We audited the full catalogue of 26,646 image records and 316,220 boxes. Separately, we decoded, dimension-checked and hashed the actual 8,000 training and 2,000 validation images used here. These have 94,609 and 24,342 objects, with all 14 classes and no selected train-validation content overlap. Unselected-image acquisition and integrity are still incomplete."

"One image declared 1920 pixels width but was actually 1620. Visual comparison could not justify a simple coordinate repair, so we quarantined it without changing the source. We also excluded two degraded candidate files and froze deterministic replacements before training."

## 3. What actually trained (45 seconds)

"The COCO-pretrained YOLOv8n trained for all 30 epochs on Apple M5 MPS, using image size 640, batch 8, seed 42 and frozen AdamW settings. It finished with exit code zero, in approximately 4 hours 27 minutes. Best epoch was 30; early stopping did not occur. Both checkpoints are readable and checksummed. MPS nondeterminism warnings mean we do not promise bitwise identical replay."

## 4. Measured accuracy (75 seconds)

"We evaluated best.pt again, independently, on the same frozen 2,000-image validation set. Precision is 0.6100, Recall 0.5435, mAP50 0.5600, and mAP50 to 95 0.4584. Harmonic aggregate F1 is 0.5749; macro per-class F1 is 0.5394. They differ because averaging and taking a harmonic mean are different operations."

"Three-wheelers perform best, with AP50 to 95 of 0.7403. Mini-bus is 0.1383 and Others 0.0242. Others has zero recall at the selected F1 threshold; the precision value of one is an evaluator convention, not perfect performance. Rare categories have only 31 and 58 validation examples. These are validation results used for checkpoint selection, not independent test accuracy."

## 5. Visual findings and timing (75 seconds)

"In our local paired ground-truth review, the model detects many motorcycles and auto-rickshaws well. It misses distant small vehicles, duplicates boxes in dense scenes, calls a Mini-bus a Bus, and calls a construction vehicle a Truck instead of Others. Some visible vehicles are unlabelled in the source, so not every unmatched prediction is a hallucination."

"We timed the proper checkpoint with MPS synchronization, batch one, 10 warm-ups and 100 images. Inference averaged 4.96 milliseconds. File-to-result latency had median 31.73 and p95 34.36 milliseconds, corresponding to 31.89 still images per second. This excludes camera capture and display; I am not claiming real-time video or production readiness."

## 6. Evidence and next step (30 seconds)

"The repository includes measured results, PR/F1 curves, confusion matrix, immutable provenance, tests, dashboard and this faculty report. Controlled Phase 2 experiments can begin only after the saved subset entry gate and Git checkpoint pass. No Phase 2 training has been run in this closeout. The next experiments should change one factor at a time and compare accuracy, rare-class behavior and identically defined latency."

## Two-minute demonstration (safe offline commands)

From the project or extracted ZIP root:

```bash
python3 scripts/show_progress.py
```

With the project environment installed:

```bash
python -m streamlit run app.py
python -m pytest -q
```

Open the PDF, show the current baseline table, per-class values, PR/F1 curves and timing definitions. The ZIP works as an artifact review without images or weights. On the original project machine only, show local `reports/predictions/error_analysis/review_pair_4232.jpg`, `review_pair_10518.jpg`, `review_pair_4711.jpg` and `review_pair_954.jpg` for successes, small misses and rare-class errors. Do not run training/evaluation during a short presentation.

## Likely questions

- **Is Phase 1 the full dataset?** No. Catalogue audit is full; actual pixel integrity/training/evaluation are explicitly subset-based.
- **Why best epoch equals last?** Validation fitness peaked at epoch 30; both provenance callback and CSV agree. We still evaluated best.pt independently.
- **Why are two F1 values different?** One is harmonic mean of averaged P/R, the other averages per-class F1. Neither is micro-F1.
- **Can this count traffic yet?** No; tracking, counting, sustained video latency and deployment validation are future work.
- **Why not repair bad labels during validation?** Changing frozen ground truth after seeing predictions would compromise comparison. Source limitations are documented separately.
- **Does 31.89 FPS mean real-time?** It measures sequential local still-image processing only. A video criterion and complete video pipeline have not been tested.
