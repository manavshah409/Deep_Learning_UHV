# Model artifacts

Weights remain local and ignored. `yolov8n.pt` is the official Ultralytics COCO-pretrained initialization, downloaded from https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt.

Fine-tuned checkpoints will be saved under `runs/<experiment>/weights/`. Completed run provenance records file size, SHA-256, experiment ID, dataset manifest checksum and configuration in `reports/tables/<experiment>_provenance.json`.

A pretrained checkpoint is not a trained UVH-26 baseline. No baseline results are claimed until training and separate validation complete. No weights are uploaded to GitHub, Git LFS or external model hosting.
