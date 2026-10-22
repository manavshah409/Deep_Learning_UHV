# Phase 3B video counting validation

Status: implementation in progress; human annotations NOT reviewed; no counting accuracy claim.

## Stage 0 inspection and plan

Reuse the permitted `data/videos/derived/bangalore_0_30_v2.mp4`: 900 fully decoded frames, 1920×1080, 30 FPS, 30 seconds, mp4v. SHA-256 `42833cd01527ee03c6bee7b9c6caba5baa75532c57ca3262757dd51f0347dab2`. Existing provenance identifies Gpkp, Wikimedia Commons, CC BY-SA 4.0; see PHASE_3A_VIDEO_SOURCES.md. Sampled beginning/end frames show a mostly stable elevated view, modest handheld motion and severe glare. Suitable for a limited fixed-image-line pilot, not a stabilized road-coordinate benchmark.

E1 weights verified SHA-256 `9f1045381024445b50e33539791da224b732d61781c73b347013477ebb077fab`. Existing tracker/counting tests and code inspected. Preserve Phase 3A's chronological evaluator unchanged; Phase 3B uses closest-time deterministic matching. Existing E4 and Phase 3A uncommitted files remain separate.

Plan: freeze versioned protocol; extend track histories and count displacement guards; add revisioned human annotation tool; add gated evaluator and review candidates; benchmark four modes; integrate saved evidence; test, document and deliver eligible changes. Stop at human-review gate.

Historical result summaries were fingerprinted before edits without reading dataset manifests or reserved1500. The earlier claim that reserved1500 was never opened is not adopted: historical YOLO validation included those images; this task does not access them.

## Implementation and protocol freeze

Reused ByteTrack and finite-line counter. Added confidence/observed-class/resolved-class histories, explicit minimum 10-pixel displacement, and public identity generation on late raw-tracker revival. Track exports distinguish observed tracks from lost tracks with last-observed geometry. Event class is confidence-weighted vote at crossing and stays frozen; final track vote is recorded separately. Counting requires three observations, 5-pixel dead-band, finite-line intersection, maximum gap one observed frame, and at most one event per track per direction. No same-direction recrossing reset; no cooldown is required for this policy. Detection instances and track IDs are not physical vehicle counts.

Inherited detector confidence 0.25, NMS IoU 0.7, imgsz640, max300, seed42. ByteTrack high/new 0.25, low0.1, buffer30, match0.8, fuse_score true, maximum absence30. Because inherited detector confidence is0.25, detections below that threshold do not reach ByteTrack's low-confidence association stage. This limitation is preserved explicitly rather than silently retuning the frozen configuration.

Line endpoints (153.6,648) to (1382.4,648), normalized [(0.08,0.6),(0.72,0.6)]. A_to_B is negative-to-positive signed side (downward), B_to_A upward. Event timestamp is first observed opposite-side point beyond dead-band; interpolated time retained separately. Human reviewer records visible crossing frame; ±0.5s tolerance is frozen. Matching is global nearest absolute time pair first, same video/class/direction, one-to-one; ties GT timestamp then GT ID then prediction ID. This is deliberately not the previous Phase 3A chronological matcher and does not guarantee maximum-cardinality matching.

Protocol v1 and its failed tracking run are preserved. At frame32 the strict lifetime guard rejected a raw ByteTrack ID returned after maximum absence. Protocol v2 assigns a new public ID and records the transition, never joining terminated histories. Full v2 tracking/counting/complete runs each recorded two such transitions. No detector was retrained or retuned. Final source hashes are frozen inside protocol_v2.json; v1 source hashes are historical, not expected to match the corrected working tree.

## Annotation and evaluation gate

Dashboard manual annotation shows only source frames plus the frozen line. It supports exact sequential frame seeking, previous/next, timestamp jump, slow 30-frame playback, class/direction/uncertainty/notes, editing/deleting, immutable CSV/JSON revision saves and resume. Completion requires explicit whole-video independent human review, no prior prediction inspection, resolved uncertain events, and explicit zero-event confirmation if empty. An incomplete revision has no completion marker and cannot replace a valid revision. Reviewer code and notes must contain no personal information. UI attestations are not external proof that a human watched the footage.

Evaluator produces TP/FP/FN, P/R/F1, per-class and per-direction scores, all 28 class-direction count groups, signed/absolute/percentage error, MAE, WAPE, over/undercount rates and crossing-time error. Zero-GT percentage denominators yield null. Count over/undercount rates use positive/negative net count error divided by GT, not event FP/FN rates. Pilot acceptance was frozen before measurements: event F1≥0.80 and WAPE ≤20%, with complete human review. These targets do not certify deployment.

No reviewed manual ground truth exists. Counting precision, recall, F1, count error, acceptance and visually established error-category counts are unavailable, not zero. Correct/false/missed gallery matching support is implemented; fragmentation, duplicates, ID switches, occlusion, jitter, parallel motion and annotation limitations need human visual review. No formal MOT scores are claimed. Final reviewed-video evaluation was not executed.

## Complete pipeline benchmark

All four modes processed the same900 frames independently, batch1 MPS, ten warm-up iterations excluded, no frame dropping. Startup/model loading and first decode, warm-up and setup recorded separately. Outer detector timing includes device synchronization and output extraction; internal preprocess/inference/NMS timers are nested diagnostics, not additional additive stages. Wall time includes fresh decoding, tracking/counting as applicable, event evidence image writes, rendering, mp4v encoding and encoder flush, detailed CSV/JSON exports. Final summary serialization and integrity hashing are outside this boundary. Live camera capture, browser display/network and optional H264 presentation transcode are not measured.

| Mode | FPS | Median frame ms | p90 ms | p95 ms | Processing wall seconds |
|---|---:|---:|---:|---:|---:|
| detection | 85.686 | 11.472 | 12.734 | 13.003 | 10.503 |
| tracking | 85.431 | 11.126 | 12.613 | 12.922 | 10.535 |
| counting | 83.433 | 11.316 | 12.999 | 13.582 | 10.787 |
| complete | 60.043 | 16.406 | 18.487 | 19.186 | 14.989 |

The complete pipeline predicted 40 crossings: {'A_to_B': 12, 'B_to_A': 28}. Frame detection instances: 10194; maximum observed active tracks: 22. These are distinct quantities. Complete processing meets the prerecorded-file target of30 FPS (2.001×), but counting correctness is not established and this is one short clip/session. No causal speed advantage is inferred from small timing differences between modes.

## Verification and delivery

Focused coverage 43 tests; final full repository 311 passed. New modules, dashboard and delivery scripts pass Ruff; legacy tracking/counting pass Pyflakes with pre-existing compact formatting retained. StageD and StageE artifact validators passed without local-manifest options. Sixteen historical result summary fingerprints match. Final checks are recorded in reports/phase3b/verification.json. Runs, video, previews, annotation drafts, notes and detailed trajectories remain ignored. No reserved1500 access, detector training, fine-tuning, new fusion or public deployment.

Faculty dashboard AppTest rendered both actual local annotation and saved-evidence pages; H264 preview has900 frames and is stored separately from immutable measured output. Portable package allowlist includes Phase 3B code and small summaries, never road footage or weights. Saved evidence imports remain lightweight; manual annotation requires the full local environment.

The isolated eligible staged repository passed **292 tests**. The local working tree passed **311 tests**, including preserved uncommitted Phase 3A/E4 tests that are deliberately outside this delivery. Working-tree differences in shared documents are staged selectively; unrelated edits remain local.
