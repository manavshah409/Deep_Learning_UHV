# Faculty briefing: E3 Stage D completed

We trained Faster R-CNN ResNet-50-FPN for all 20 epochs on 8,000 images. It uses 14 vehicle classes plus background, a conservative batch-one MPS configuration and unweighted sampling. A checkpoint-publication failure after epoch 6 was repaired without restarting training; the recovered run exited successfully and its checkpoints/provenance passed audit.

Epoch 13 is selected. A fresh evaluation on the same 500 calibration images exactly reproduced AP50 **56.36%** and AP50:95 **42.91%**. Macro precision is **45.62%**, macro recall **63.49%**, harmonic aggregate F1 **53.09%**, and mean class F1 **52.63%**. These are calibration results, not held-out test performance and not a fair comparison with historical YOLO scores.

Three-wheelers (1,035 annotated objects) and two-wheelers (2,920) are strongest. Others has zero recall with only 10 objects; Mini-bus also has limited evidence (19 objects). Reviewing 13 scenes showed correct foreground detections alongside duplicate boxes, small/occluded misses and passenger-vehicle confusions. Suspected annotation ambiguities were documented without changing labels.

The 100-image batch-one MPS benchmark measured **112.90 ms median / 122.45 ms p95** end-to-end latency and **8.75 still images/s**. It includes file loading/preprocessing, model inference and postprocessing, but not live-video capture/display. No real-time or production claim is justified. Active training took **10 h 56 m 48 s**; calendar duration was **11 h 35 m 22 s**.

The next step requires separate authorization: freeze a matched E1/E3 comparison protocol before using the reserved set. The reserved 1,500 images were not accessed here; no fusion or new training was started. The reserved pool also has historical validation exposure, so it is not a pristine independent test set.

Developer evidence: [technical report](../phase_reports/E3_STANDALONE_CALIBRATION_CLOSEOUT.md), [commands](../reproducibility/E3_STANDALONE_CALIBRATION.md), [metrics](../../reports/evaluations/E3_fasterrcnn_best_calibration500_v1/metrics.json), [training plots](../../reports/evaluations/E3_fasterrcnn_best_calibration500_v1/training_curves.png). The dashboard has a separate calibration-only E3 artifact section.
