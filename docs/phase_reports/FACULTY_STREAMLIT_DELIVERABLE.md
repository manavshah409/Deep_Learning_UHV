# Faculty Streamlit deliverable — 2026-09-25

## Implemented

New modular entrypoint `dashboard/app.py`; the historical root dashboard remains unchanged. Eight sidebar pages: overview, image detection, recorded video, analytics, comparison, training, dataset insights, documentation. Saved Evidence Mode defaults on and requires no model or dataset. Optional Live Inference Mode uses existing `src.video.pipeline.FrozenDetector`, verified E1 YOLOv8s checkpoint (image size 640). Mapping is checked against the existing YAML and frozen detector names. Hash is checked before requests and on lazy cached loading. Model access is serialized; uploaded bytes and results are session-local, no permanent uploads or temporary files.

Dark-blue sidebar, light content, restrained orange metric accents, responsive columns, clear verified/experimental/planned/unavailable states, artifact charts and technical expanders. Image controls cover confidence, NMS, max detections, device, labels and scores. Outputs: original/annotated image, counts, boxes, measured request timing, PNG/CSV/JSON. MPS-unavailable fallback uses CPU. Dashboard defaults never import torch. Requirements are separate from the training lock.

## Evidence and scope

Comparison reads original JSON artifacts: validation2000 E0/E1 AP50:95 .458407/.524760; matched calibration500 E1/E3 .482829/.429070. These pools are separate. Epoch histories are read without filling missing values; selected epochs E0=30, E1=22, E3=13. EDA uses saved charts only. Full catalogue audit (26,646 images/316,220 boxes) is distinct from local 10,000-image integrity. Unselected acquisition/integrity remains incomplete.

Stage F fusion was already rejected, so the app reports that verified outcome rather than the outdated request wording “ongoing”. No fusion is run here. No reserved1500 manifest/image was accessed. No new experiment, training, dataset mutation or historical result regeneration occurred.

## Validation

- Focused dashboard tests: 24 passed; all eight Saved Evidence pages render without exceptions/errors.
- Full local repository suite: 243 passed (includes 15 pre-existing untracked Stage F tests and 4 unrelated untracked Phase 3 cases).
- Ruff: clean for new dashboard, package builder and dashboard tests.
- Historical preservation: 840 report files byte-identical to the pre-edit snapshot. Existing Stage F validator passes (138 earlier historical files preserved).
- Actual CPU inference smoke: generated blank 64×64 image, verified E1, 0 detections, valid 91-byte PNG. This is a functionality check, not an accuracy or performance result; no dataset/media accessed.
- Manual local server starts on 127.0.0.1:8511. Browser overview inspected; metric contrast issue corrected. AppTest verifies page content. An initial browser refresh hit a permission-review timeout; the visible Rerun control subsequently succeeded and a final screenshot confirmed readable metric contrast.
- Portable package uses an explicit small-file allowlist, exclusive creation and ZIP CRC check. Contains source, mapping, histories, original metrics and five lightweight charts; no checkpoint, dataset or uploaded/generated media. Archive receipt and extraction smoke results are recorded in the completion progress note.

Initial test failures were a test-harness relative-path issue; fixed by resolving the app path absolutely. No existing test was weakened.

## Limitations

Video upload is explicitly coming next: existing detector-only pipeline lacks progress events, stored per-class frame predictions and browser codec guarantees. Saved synthetic 20-frame smoke is shown with its scope. No tracking/unique counting is implied. Image analytics use the current actual prediction only; frame time-series is unavailable until video integration. The density rule (0–5 low, 6–15 moderate, >15 high) is configurable and unofficial.

No permitted photographic demo selection was supplied, so no dataset photos or saved annotated photo outputs are bundled. Users may upload their own permitted image. Model load/hash checks, browser and network time are excluded from request timing; no live-video FPS claim. The 70–80% precision/recall target remains unmet.

## Delivery

See `README_DEMO.md` and `docs/reproducibility/STREAMLIT_DEMO.md`. Launch: `.venv/bin/python -m streamlit run dashboard/app.py`. Git delivery is limited to faculty-app code, tests, requirements, docs and its changelog entry. Pre-existing Stage F and unrelated Phase 3 work is preserved and not staged. The delivery commit is the commit introducing this report; no circular self-hash is embedded. No public deployment.
