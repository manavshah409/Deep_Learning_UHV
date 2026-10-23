# Faculty briefing — Stage F completed, 2026-09-24

We tested whether combining YOLOv8s with Faster R-CNN improves vehicle detection. We reused existing predictions on500 calibration images and used five-fold cross-validation, selecting settings on400 images and checking them on the remaining100. We tested19 bounded configurations across four families, with no additional training.

The strongest combination, asymmetric weighted-box fusion, improved AP50:95 from48.28% to49.24% and macro class F1 from58.62% to59.47%. Its precision/recall were62.92%/57.63%. These gains did not meet the improvement rule fixed before evaluation: at least1 percentage point AP or2 points macro F1, alongside other safeguards. **We rejected E4 and retained YOLOv8s. Neither precision nor recall reached70%.**

Actual sequential model inference plus fusion averaged156.78ms per image, or6.38 still images per second. This includes both detector pipelines and fusion; it is not live-video performance. The benchmark used a fixed representative of the rejected family, not a newly selected model.

The useful project outcome is a reproducible negative result: adding a larger second detector brings considerable computational cost without the required accuracy gain in this search. The next proposal is better training coverage and targeted label review, followed by a controlled YOLOv8s fine-tuning pilot. No improvement is guaranteed.

The reserved1,500 images remain sealed, and no retraining, classifier or Phase3 work was started. [Full report](../phase_reports/E4_FUSION_FEASIBILITY.md) · [All models](../ALL_MODEL_PERFORMANCE.md).
