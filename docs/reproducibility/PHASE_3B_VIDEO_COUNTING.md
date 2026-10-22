# Phase 3B reproduction and human handover

Run commands from repository root in the existing environment. Do not train a detector. The permitted video and verified E1 weights are local and excluded from Git. See PHASE_3A_VIDEO_SOURCES.md for Gpkp CC BY-SA4.0 source attribution. Keep adaptations local or retain share-alike attribution when distributing.

## Open the dashboard

```bash
.venv/bin/python -m streamlit run dashboard/app.py
```

Choose **Video Traffic Analytics**, **protocol_v2.json**, **Manual annotation**. Avoid Saved evidence before the independent annotation pass. Use a reviewer code such as reviewer01; enter no names or personal information. The displayed line is fixed to the frozen protocol. To change it, use the new-protocol command shown in the line panel, then create a new processing run and new annotations; never reuse v2 predictions with a changed line.

1. Review all900 frames of the30-second source clip. Use previous/next, timestamp jump and slow playback. Annotate each vehicle bottom-center passing through the finite red line, including model-missed vehicles. A_to_B is downwards; B_to_A upwards.
2. Select the original14-class label, direction and crossing frame. Mark uncertain objects instead of guessing. Record brief factual notes without personal information. Each add saves a new draft revision.
3. Edit/delete mistakes in the table and save a new draft. Resume only a completed revision after an interruption. Unsaved table edits are not persisted.
4. Resolve uncertainty, ideally with a second reviewer. Confirm the whole video was independently reviewed and predictions were not seen first. Explicitly confirm zero events only if appropriate. Finalize to a new revision.
5. Copy the saved relative revision directory displayed by the tool. CSV and JSON exports are available, but the evaluator uses the completion-marked revision directory for integrity checks.

## After human review only

Replace the revision placeholder with the actual path printed by the dashboard:

```bash
.venv/bin/python scripts/publish_phase3b_review.py \
  --annotations runs/phase3b_annotations/ff25e26a085b/REVISION_DIRECTORY \
  --result-id phase3b_reviewed_v1
```

This refuses unreviewed, uncertain, prior-prediction-exposed or incompatible annotations. It independently evaluates phase3b_complete_v2, writes an immutable local evaluation directory and publishes small metrics to reports/phase3b/reviewed_result.json. Public output strips notes and reviewer codes. The script refuses to overwrite an earlier reviewed result. Inspect outcomes and visual error candidates before any future Git commit. Human visual diagnostic categories must not be inferred from matching alone.

## Recorded processing commands

The following IDs already exist; preserve them. Reproduction requires fresh IDs and compatible source hashes. All four benchmark modes were run separately, not simulated from one timing table.

```bash
.venv/bin/python -m src.video.phase3b freeze --protocol reports/phase3b/protocol_v2.json --device mps
.venv/bin/python -m src.video.phase3b run --protocol reports/phase3b/protocol_v2.json --run-id phase3b_detection_v2 --mode detection
.venv/bin/python -m src.video.phase3b run --protocol reports/phase3b/protocol_v2.json --run-id phase3b_tracking_v2 --mode tracking
.venv/bin/python -m src.video.phase3b run --protocol reports/phase3b/protocol_v2.json --run-id phase3b_counting_v2 --mode counting
.venv/bin/python -m src.video.phase3b run --protocol reports/phase3b/protocol_v2.json --run-id phase3b_complete_v2 --mode complete
.venv/bin/python scripts/validate_phase3b.py
.venv/bin/python scripts/prepare_phase3b_preview.py
```

The freeze and preview commands refuse existing outputs. MPS must be available; no silent CPU fallback. In a sandbox, native GPU access may require execution approval. Protocolv1, the30-frame actual-checkpoint smoke and failed v1 tracking evidence remain preserved. A browser-compatible H264 copy is made in a separate ignored preview directory after measurement. Original measured run files are unchanged.

## Tests and validators

```bash
.venv/bin/python -m pytest -q tests/test_phase3b.py tests/test_tracking.py tests/test_counting.py
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src/video/phase3b*.py dashboard/video_analytics.py dashboard/app.py scripts/*phase3b*.py tests/test_phase3b.py
.venv/bin/python scripts/validate_phase3b.py
.venv/bin/python scripts/validate_e3_stage_d.py
.venv/bin/python scripts/validate_stage_e.py
```

Synthetic tests use fake detectors and generated images/video fixtures; no road footage or weights required. Manual annotation is a human task. Stop at this gate rather than invent precision/recall or marking Phase 3B complete.
