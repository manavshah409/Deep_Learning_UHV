# Faculty presentation script

**Project:** Real-Time Vehicle Detection and Traffic Analytics for Indian Urban Roads Using YOLOv8 and the UVH-26 Dataset  
**Presenter:** Manav Shah  
**Purpose:** Phase 1 progress review, 11 September 2026  
**Length:** Approximately 6-7 minutes, followed by a short demonstration.

This script describes completed work at the time of the faculty pack. Do not describe the one-epoch smoke model as the final baseline. The full audit and proper baseline are still in progress.

## 1. Introduction - 40 seconds

“Good morning, ma'am. My project is on vehicle detection and traffic analytics for Indian urban roads using YOLOv8 and the UVH-26 dataset.

“The problem is that Indian traffic has many different vehicle types, including two-wheelers, auto-rickshaws and light commercial vehicles. A general object detector may recognize broad categories such as cars and trucks, but this project aims to distinguish the finer vehicle categories that matter in Indian traffic.

“My current work is Phase 1: building a reliable dataset pipeline and verifying that model training and evaluation work on my MacBook.”

**Show:** First page of the faculty report.

## 2. Dataset and ground truth - 60 seconds

“I am using the official UVH-26 dataset from IISc AIM. I inspected the downloaded annotation files instead of relying only on the published dataset description.

“The Majority Voting annotations contain exactly 26,646 images and 316,220 vehicle boxes. The official training split has 21,349 images, and validation has 5,297 images. There are 14 vehicle categories.

“Majority Voting combines multiple annotators' judgments. The dataset also includes STAPLE annotations, but I have kept the two versions separate. The inspected STAPLE files list fewer images, so directly combining them would make the experiment difficult to interpret.

“I also pinned the dataset revision so that the same input version can be downloaded again.”

**Show:** Dataset table and `reports/audit/schema.json`. Say “images listed in the annotations,” not “all images downloaded.”

## 3. Data engineering - 70 seconds

“The original annotations use COCO format. Each bounding box is stored as its top-left x and y coordinates, width and height. YOLO needs the box centre, width and height normalized by the image dimensions.

“I implemented that conversion with checks for invalid coordinates and unknown classes. I preserve the original class names and map their IDs to contiguous values from zero to thirteen. No vehicle classes are merged.

“The annotation audit found no invalid boxes, duplicate IDs within a split or shared train-validation filenames. However, checking the actual image files is a separate task. Full image decoding, dimension matching and content-hash checks are still pending acquisition of the remaining files.

“The raw dataset is never edited. Processed labels and manifests are separate, and images are linked rather than copied. This saves storage and protects the original data.”

**Show:** `src/data/convert_to_yolo.py`, `configs/class_mapping.yaml`, and the audit summaries. Do not claim that complete-dataset conversion has already run.

## 4. What the exploratory analysis showed - 60 seconds

“The class distribution is very uneven. There are 149,730 two-wheeler instances but only 352 instances in Others. An overall score could therefore hide weak results on rare vehicle classes.

“An image contains about 11.9 objects on average, with a median of ten and a maximum of 66. This confirms that dense scenes are important in the dataset.

“I manually inspected 32 early annotation previews covering all fourteen classes. The coordinate conversion looked aligned, but the source labels are not perfect. For example, one sparse scene has an unlabelled foreground motorbike, and another has an unusually tall truck box. I recorded these observations rather than silently changing the ground truth.”

**Show:** Class-distribution chart. If presenting on this Mac, optionally show the local annotation previews; they are excluded from the portable ZIP.

## 5. Training work completed - 70 seconds

“I chose YOLOv8n because it is a small detection model that is feasible on an Apple Silicon laptop. It starts from COCO-pretrained weights and is adapted to the fourteen UVH-26 classes.

“I have completed a genuine one-epoch smoke-training run on 64 training and 32 validation images. Those 96 images were independently checked, and Ultralytics loaded them without reporting corrupt images. The run used MPS, batch size eight and image size 640.

“The losses were finite, validation completed, and best and last checkpoints were saved. The total wall time was about 35.7 seconds. I also ran the separate evaluator and generated its per-class metrics and plots.

“This is a pipeline test, not a useful final detector. Its scores are very low, and the saved predictions had no detections at a confidence threshold of 0.01. I am not presenting those numbers as the final project accuracy.”

**Show:** Smoke-run provenance, saved epoch CSV and the clearly labeled smoke results in the PDF. Never call this the completed 30-epoch baseline.

## 6. Reliability, current status and next step - 60 seconds

“The code now has 47 passing tests using synthetic fixtures. These cover bounding-box conversion, class mapping, invalid inputs, empty labels, split overlap, content leakage and reproducibility of subset selection.

“The main remaining dependency is the image download. It previously stopped because of network errors and has now resumed. I am prioritizing a deterministic subset of 8,000 training images and 2,000 validation images. It preserves all fourteen classes, and its class shares differ from the original splits by at most about 0.31 percentage points.

“The planned proper baseline is thirty epochs at image size 640. After it completes, I will report validation precision, recall, F1, mAP, per-class results and measured inference speed. Full image integrity checks and the final report are also required before I mark Phase 1 complete.

“After that, Phase 2 can focus on small vehicles, minority classes and eventually vehicle counting and density analytics.”

## Short demonstration - 2 minutes

1. Open `output/pdf/UVH26_Faculty_Progress_Report.pdf`.
2. Run `python scripts/show_progress.py` from the project root to display saved evidence.
3. Run `python -m pytest -q` in the installed project environment.
4. Open the class-distribution chart and point out the imbalance.
5. Show the smoke CSV and checkpoint provenance; distinguish execution success from detection quality.

Do not launch a large download or training run during the presentation. The ZIP is sufficient for the report, source-code walkthrough and saved evidence. Local traffic images and weights remain on this Mac.

## Likely questions and honest answers

**Why YOLOv8n?**  
It is a lightweight starting point for transfer learning on the available Apple Silicon hardware. It is a baseline choice, not a claim that it is the most accurate architecture.

**What is your contribution if you use a pretrained model?**  
The contribution is the reproducible UVH-26 audit, class mapping, conversion, validation, experiment setup and evaluation of India-specific categories. It is an applied deep-learning project, not a new network architecture.

**What accuracy have you achieved?**  
The proper baseline has not completed. Only smoke metrics are available, and they are not meaningful final performance. Object detection will be assessed with precision, recall and mAP rather than a single classification-accuracy number.

**Is it already real-time?**  
No end-to-end real-time claim has been established. Timing the smoke checkpoint does not demonstrate a useful real-time traffic application. The final detector must be evaluated for both quality and throughput.

**Why use a subset?**  
To make the initial experiment feasible on a laptop. The 8,000/2,000 selection stays inside the official splits, covers all classes and has documented distribution differences. Results will be labeled as subset results.

**How do you prevent leakage?**  
The code preserves official splits and checks filenames and image hashes. The pilot passed its checks. Full-dataset hash checks are pending, and the inspected metadata does not establish camera-level or temporal independence.

**Can the experiment be reproduced exactly?**  
The dataset revision, package versions, seed, manifests and configuration are recorded. However, PyTorch warns that some MPS operations are not deterministic, so bit-for-bit replay is not guaranteed.

**Why are there no weights or dataset images in the ZIP?**  
They are large local artifacts excluded by the project policy. The ZIP contains the code, configurations, reports and reproduction instructions. The local smoke checkpoint has a recorded checksum.

## Thirty-second version

“Ma'am, I have completed the project setup, inspected all four annotation files, audited the Majority Voting annotations, generated EDA, implemented the YOLO data pipeline and passed 47 tests. The annotations contain 26,646 images, 316,220 objects and fourteen classes. A genuine one-epoch smoke run completed on MPS with saved checkpoints and evaluation outputs. The full image download and proper thirty-epoch baseline are still in progress, so I am presenting verified progress rather than claiming final accuracy.”
