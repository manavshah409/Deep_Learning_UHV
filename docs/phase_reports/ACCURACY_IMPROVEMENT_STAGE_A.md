# Accuracy Improvement Stage A — preparation and one-epoch preflight

## Faculty request and controlled plan

Prepare a second architecture, imbalance treatment, eventual model combination, justified epoch budget and durable per-epoch timing. This stage implements only E3 preparation and a one-epoch pipeline preflight. E1 remains the selected detector. E0/E1/E2 and Phase 3 evidence are preserved. No full E3 training, E4 fusion, E5 YOLO fine-tuning or further Phase 3 work is authorized here.

E3 uses the official Torchvision Faster R-CNN ResNet-50-FPN COCO_V1 initializer: a two-stage region-proposal detector and multiscale feature pyramid, providing a meaningful architectural contrast to single-stage YOLOv8s. Replace only the prediction head for15 outputs: background 0 and the unchanged14 UVH classes at1–14. No Inception, custom ResNet-18 backbone or installed-library patch. Official initializer URL/SHA and package versions are recorded under `reports/accuracy_stage_a/` and the run snapshot.

Future E4 is parallel inference with calibrated class-aware fusion, not a sequential cascade. Its weights/thresholds may be chosen only on calibration data; no fusion implementation or tuning occurs now. E5 weighted YOLO fine-tuning is optional and unstarted. Weighted E3 versus E1 changes both architecture and sampling, so it is **not an architecture-only ablation**. A future unweighted E3 control is needed to isolate the weighting effect.

## Dataset separation and provenance

Seed 42 shuffles the existing2,000 validation rows after canonical image-ID/filename ordering. The first 500 form calibration and the remaining 1,500 form reserved final comparison. Both contain all14 classes, with zero image-ID, filename and source-content-hash overlap. Parent manifests and class mapping remain unchanged. Per-class object counts and file/canonical SHA-256 values are in `reports/accuracy_stage_a/protocol_v2/protocol.json`.

The 1,500 images are not used for StageA model inference, training or parameter tuning. Structural label counts were read only to verify class coverage. They are **not a newly unseen test set**: historical E0/E1/E2 already used the original2,000 validation images. Future comparative claims must disclose this selection history; a genuinely independent test set remains desirable.

128 training images and64 calibration images are deterministic seed-42 samples of their allowed pools. Every loaded preflight image/label is rehashed against its manifest. The one-epoch preflight may omit some rare classes; its metrics are pipeline diagnostics, never model-performance estimates.

Preparation v1 wrote partial split exports before a NumPy-int JSON serialization error in weighting diagnostics. Evidence is preserved locally with FAILURE.json and excluded from use. Corrected v2 regenerated identical deterministic splits and complete diagnostics; no dataset or historical artifact was modified.

## Imbalance method

Class weights from **training annotations only**: `min(3.0,sqrt(Nmax/Nc))`. Classes absent from training receive 0 and cannot be synthesized by sampling. Image weight is the arithmetic mean of weights of its **unique present classes**, so repeated boxes of one class do not multiply the image weight. Empty images receive 1.

The sampler makes exactly N weighted draws per epoch and removes an image from eligibility after 3 draws. This hard exposure cap avoids unlimited rare-image repetition; some images are omitted in a given epoch. Seed 42+epoch controls reproducibility. Sampling changes joint class exposure in mixed-class images and is not equivalent to per-object loss weighting. Training class weights, all 8,000 image weights and proposed-epoch distribution diagnostics are exported. No weighted classification loss or targeted augmentation is implemented; standard Torchvision losses remain intact. No full weighted run has started.

## Model and preflight configuration

`configs/accuracy/E3_fasterrcnn.yaml`: official COCO-pretrained ResNet-50-FPN; random15-class head; aspect-preserving short side 480, maximum side 640; standard Torchvision normalization; no added augmentation; physical batch 1, workers 0, seed 42; SGD lr .005,momentum .9,weight_decay .0005; float32; no mixed precision or gradient accumulation. These controls differ from E1 AdamW/augmentation and must be disclosed in comparisons. The resize has a 640 long-side ceiling; it is not an assertion that all tensors match YOLO's letterbox geometry.

The MPS NMS/ROIAlign operator check passed. A single real-batch forward/backward/optimizer memory check chooses conservative batch 1; it is not a maximum-batch search or absolute peak-memory measurement. Its update is discarded and a seeded pretrained model rebuilt before the measured epoch. Any unsupported-MPS failure is preserved separately and triggers a documented whole-run CPU fallback only when configured. No silent per-operation fallback is enabled.

## Timing and integrity

Reusable `src/training/epoch_timing.py` creates `epoch_timing.csv` exclusively and flushes+fsyncs each completed row. Schema includes epoch, UTC start/end, perf_counter train/validation/total/cumulative seconds, LR, total/classifier/box/RPN-objectness/RPN-box losses, calibration P/R/AP50/AP50:95, saved checkpoint, device, batch and resize policy. Total includes checkpoint save/readback verification; train/validation times are separate. A failed epoch writes FAILURE.json and is not appended as completed. Mac sleep/calendar time can differ from active timers.

Validation converts original-pixel xyxy targets to COCO xywh for pycocotools AP (maxDet300). Foreground IDs remain1–14 with no category 0 GT. Fixed-threshold P/R use confidence .25, IoU .5, macro over represented GT classes. These differ from E1 max-F1 operating-point metrics and must not be compared directly as a controlled final result. Detector AP confidence floor .001, NMS .5, max_det 300 are explicitly fixed for this preflight.

Checkpoints include model, optimizer, epoch and provenance; SHA256checked before loading; all model tensors checked finite after save. Immutable run directories, COMPLETE hashes and historical preservation snapshot provide audit evidence. Full training is guarded by a separate explicit CLI switch; only `--preflight` was executed.

## Evidence and outcome

Run `E3_stageA_preflight_seed 42_v1` completed one epoch and exited **0** on MPS, physical batch **1**, **128 training / 64 calibration images**. Forward/backward/optimizer, finite losses/gradients, calibration validation, readable finite checkpoint, durable timing row and COMPLETE hash validation passed. No CPU fallback was needed. The one-batch memory check took **6.40 s**, sampled allocated memory **477.4 MiB** (not absolute peak). Its update was discarded before the measured epoch.

| Timing | Seconds |
| --- | ---: |
| Training | 39.622 |
| Calibration validation | 10.966 |
| Completed epoch including checkpoint verification | 51.996 |

Saved checkpoint: `runs/E3_stageA_preflight_seed 42_v1/epoch_001.pth`, **330,578,251 bytes**, SHA-256 `7e8bcc54a8d8e85efc02c0879554f1998f6768df88319c94cf5c4025d42bc8c9`. This is a pipeline-test checkpoint, not a replacement for E1.

Full local suite: **125 passed**; repository-deliverable suite: **121 passed** (excludes four pre-existing uncommitted Phase 3 event-evaluation tests, preserved outside this commit). Ruff F checks passed. Preservation/protocol/result validator passed. Evidence: `reports/accuracy_stage_a/tests.txt`, `validation.json`, `preflight_epoch_timing.csv`, `preflight_summary.json`, `preflight_memory_check.json`, `preflight_COMPLETE.json`. Detailed logs and checkpoints remain ignored.

Linear extrapolation scales train time by 8000/128 and calibration time by 500/64, retaining measured checkpoint overhead. It predicts roughly **42.72 minutes/epoch**, **14.24 hours for 20 epochs**, or **21.36 hours for 30 epochs**. This is not a measured full-run duration; warm caches/compilation, image mix, MPS scheduling, sleep and weighted repeats limit accuracy. Twenty epochs is a proposed budget, not a proven convergence requirement.

Exact proposed command (not executed; separate authorization required):

```bash
.venv/bin/python -m src.experiments.train_frcnn --config configs/accuracy/E3_fasterrcnn.yaml --run-id E3_fasterrcnn_weighted_seed 42_full_v1 --allow-full-training
```

Reproduction instructions: `docs/reproducibility/ACCURACY_STAGE_A.md`. No full E3, E4 fusion, E5 fine-tuning or new Phase 3 work was started. Existing pending Phase 3 files remain separate and unstaged. Git delivery is on the existing master branch; exact commit is identified by the delivery message `Prepare weighted Faster R-CNN experiment and timing preflight`.


Sampling clarification: the preflight selected pool contains 128 images; the measured epoch draws 128 samples with replacement under the cap, covering **80 unique images**, at most 3 exposures per image. This is intentional weighted sampling, not an assertion that all 128 pool images were visited. Seed control does not guarantee bitwise MPS determinism.
