# Manual counting review required

Status: **not annotated; no counting accuracy measured**. The empty worksheet is a template, not zero ground truth. Model-generated events must not be relabelled as manual truth.

Review the local `data/videos/derived/bangalore01_manual_review_v1.mp4` (30seconds,900frames,30FPS) before consulting predictions. It contains source imagery, a red line and frame/time overlay only. Derived from Gpkp's Commons clip01 under CC BY-SA4.0; retain attribution in PHASE_3A_VIDEO_SOURCES.md if sharing. Original selected derivative SHA is pinned in `configs/phase3a_bangalore01.yaml`.

Record every visible vehicle whose **bottom-center** passes through the finite red line (x153.6–1382.4,y648 in1920×1080). Downward crossing is A_to_B, upward B_to_A. Include vehicles missed by the detector. Use original14class names, record `uncertain=true` for subtype, visibility or boundary ambiguity; do not guess from the predicted label. Tracker IDs are unnecessary for human annotation.

Fill `reports/annotations/phase3a_bangalore01_MANUAL_TEMPLATE.csv` into a separate reviewed file; set video_id `bangalore01_0_30`, frame0–899 and timestamp frame/30. Direction and class must be independent. A second reviewer should resolve uncertainty where possible. Document annotation author/date and whether predictions were seen.

Preregistered comparison tolerance: ±0.5seconds (15frames), same video, direction and class, deterministic chronological one-to-one matching. Use `src.video.evaluate_events.evaluate(predictions,annotations,reviewed=True,tolerance_seconds=.5)` after parsing CSV types. For uncertain records, Python boolean values must be used, not text strings. Uncertain GT is excluded explicitly and resulting unmatched predictions remain provisional. Absolute error=|predicted−actual|; percentage error undefined when actual=0. Report matched/false/missed events as well as total/directional/class errors: equal totals do not imply correct events. No IDF1/HOTA/ID-switch scores without tracking GT.

Why handoff: headlight glare, occlusion, small vehicles and class ambiguity make an exhaustive, independent manual ground-truth pass unreliable through the sampled-frame inspection performed here. The user explicitly allowed this handoff instead of fabricated annotation. A valid pilot accuracy claim requires the completed worksheet.
