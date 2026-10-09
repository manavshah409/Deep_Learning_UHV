# New Data Model Evaluation and Error Analysis Dashboard

The existing faculty app now focuses on the most recent session evaluation. Historical E1/E3 evidence remains separately labelled and is the fallback when no new data exists. No new-data accuracy experiment has been performed as part of implementation.

## Navigation
Executive Overview; New Data Evaluation; Deep Error Analysis; Prediction Explorer; Performance and Latency; Historical Model Comparison; Dataset and Methodology; Reproducibility and Limitations. Faculty Presentation Mode disables inference and presents a guided overview → errors → examples sequence. Saved result.json can be reconstructed without weights; source images are intentionally not bundled, so imported galleries remain unavailable.

## Reused and added components
The app retains dashboard/pages.py and services.py, frozen checkpoint verification, historical loaders, EDA and training charts. dashboard/new_data separates ingestion, analysis, execution/export and UI; scripts/evaluate_new_data.py supplies offline reproduction. Uploaded images/predictions stay in the user's Streamlit session and are never cached across users. Explicit CLI exports go to ignored runs/new_data; browser evaluation persists only if downloaded.

## Frozen evaluation
E1 YOLOv8s epoch22 SHA-256 9f1045381024445b50e33539791da224b732d61781c73b347013477ebb077fab; 640, float32, batch1; AP floor .001; NMS .7; max300. Confidence .34 is checked against the sealed Stage E operating-threshold record. Matching IoU .50; COCO AP averages .50:.05:.95 with 101 recall points. Source, configuration, image, annotation-input and manifest hashes are exported. All 14 classes retain their ordering.

Precision/recall are macro means over classes represented by GT; harmonic aggregate F1 uses those means. Macro class F1 and micro counts are separate. Classes without GT have undefined AP; zero-denominator fixed metrics follow existing zero convention. AP must not be compared directly with the historical Ultralytics validation2000 evaluator. A compatible calibration500 difference table is enabled only with matching settings, metric definitions and all fourteen GT classes represented; different image pools remain conspicuously labelled.

## Matching and error taxonomy
Fixed metrics use stable confidence-descending, same-class, one-to-one greedy IoU≥.5 matches. Unmatched predictions: same-class overlap≥.5 → duplicate; otherwise any-class overlap≥.5 → classification; otherwise any overlap≥.1 → localization; otherwise background. Unmatched GT is missed. Subthreshold predictions with an eligible correct overlap are low-confidence candidates, not TPs at the chosen operating point. High-confidence background candidates (≥.8) are annotation-review candidates, never confirmed bad annotations. Confusion uses separate class-agnostic greedy matching, rows GT/columns predictions, final background row/column.

Error-event percentages use all taxonomy events as denominator; predictions and missed GT are different event units. Plots/tables include support, class errors, confusion pairs, confidence curves and AP by IoU. Size/density/aspect/position/frequency strata report GT matched recall, not stratum AP. Original area cutoffs 32² and 96²; scene density ≤5/≤15/>15; aspect ratio <.8/>1.25; edge within normalized .2; low class support <20 GT. These are documented analysis conventions, not externally validated categories. Weather, occlusion and truncation are not inferred. Optional metadata is retained; specialized occlusion evaluation is not implemented.

## Reliability
Seed42 image-resampling percentile intervals (100 draws default) cover fixed macro/micro P/R/F1. They do not cover AP, do not test significance and assume sampled images are representative; correlated video frames and rare classes undermine this assumption. The CLI allows up to10,000 replicates. Classes with fewer than20 objects are excluded from automatic strongest/weakest ranking, while their metrics remain visible with support.

## Timing
Ten warmups, synchronized outer per-image timing. Decode is measured during ingestion and added to the outer processing interval. Preprocess includes decode, array conversion and Ultralytics preprocessing; model inference and postprocess use Ultralytics stage timers. End-to-end includes Python dispatch/result transfer but excludes upload transfer, model load, charts and export. Loading is reported separately; stage sums may differ from the outer interval. Mean/median/p90/p95 and still-image throughput are provided. This is not browser response latency or live-video FPS. CPU fallback occurs when MPS is unavailable; runtime operator failures are recorded, not silently retried under a different device.

## Validation and privacy
100MB compressed/expanded archive, 2000 entries, 500 images, 100MP decoded total, 10MB/20MP per image. Paths/symlinks, duplicate basenames/stems, invalid boxes, unknown mapping, orphan/missing labels and unsupported crowd/ignore annotations are rejected. Empty labels are valid. Duplicate byte and decoded-image hashes are audited. A user-supplied safe hash index enables overlap findings; no local dataset traversal occurs. Without that index overlap verification is unavailable, never assumed negative. An independence declaration cannot prove independence.

Immutable UUID/timestamp IDs, exclusive output-directory creation and atomic evidence.zip publication prevent overwrite. Failure receipts distinguish incomplete work. Imported results have recomputed metrics and checked configuration/manifest hashes, but prediction/timing/source provenance remains explicitly user-supplied; they are not authenticated scientific runs. Source images are absent after import.

## Boundaries and incident
No training, fine-tuning, fusion, video work, downloads or deployment. Historical results remain unchanged. An initial overbroad report-preservation snapshot hashed report files including reserved manifest metadata. No reserved images were opened or evaluated; subsequent checks exclude these paths. See progress record for exact verification and delivery evidence. Consequently, this task does not claim that reserved metadata was never accessed.

## Professional presentation update 2026-10-09
The dashboard now uses a consistent light/blue theme, one executive overview, plain-language workspaces and guided navigation. New data follows validate → review audit → run, with stale validation invalidated whenever inputs change. Failed runs preserve prior results. Explorer filters now govern overlays and CSVs consistently. Metric definitions, scope labels and actionable empty states clarify what the user can conclude. See docs/progress/2026-10-09_DASHBOARD_REVAMP.md for286 passing tests, real checkpoint functionality smoke and portable package verification. Install requirements-new-data.txt for full evaluation functionality; requirements-demo.txt supports lightweight historical browsing.
