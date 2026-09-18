# Phase 3A tracking and counting implementation

## Completed synthetic stage

One frozen detector execution per frame feeds installed Ultralytics8.4.146 ByteTrack. `lap==0.5.12` is pinned separately; detector packages/checkpoints are unchanged. Tracker adapter validates finite Nx6 detections and Nx8 association outputs, unique positive IDs, class range and geometry. Frames must be sequential; repeated/nonmonotonic input does not accumulate observations. Empty detections advance the tracker. Lost tracks are retained30frames and expired deterministically.

Each track stores observation age, first/last frame, latest confidence, complete box/bottom-center history, accumulated class scores and count state. Class policy sums confidences by class; ties choose the smallest frozen class ID. Event class is copied at crossing and cannot change with later votes. IDs are run-local; fragmentation can still overcount physical vehicles with new IDs.

The finite-line counter uses normalized signed cross product distance in pixels,5pixel hysteresis, minimum3 observations, per-direction duplicate suppression (optional total-once mode), and optional ROI. Endpoint intersection is checked. The pilot counts only transitions across consecutive observed frames (maximum gap1); it does not invent crossings hidden by misses. Hysteresis-zone observations retain the last stable side. Tracks can count each direction once. No physical speed/distance or lane identity is estimated.

Pilot line is horizontal from(0.08W,0.60H) to(0.72W,0.60H), selected from unlabelled scene geometry before reporting any counting error. Bottom-center negative→positive is A_to_B (down the image); reverse is B_to_A. These are image directions, not calibrated incoming/outgoing road semantics. Config: `configs/phase3a_bangalore01.yaml`.

Outputs are isolated by immutable run ID: annotated MP4, timing CSV, configuration, event CSV/JSON, histories, flow and summary. Each export uses temp+rename; consumers require final COMPLETE.json with hashes for bundle integrity. Failures preserve partial evidence without COMPLETE. Statistics distinguish confirmed track IDs and crossings, with class/direction/minute composition, rolling10second flow, track durations and observed occupancy. Manual density thresholds10/25 observed tracks are demonstration labels, not calibrated congestion.

109 synthetic/full-suite tests passed before real-video evaluation. Added tests cover IDs, misses, expiry, empty frames, malformed outputs, confidence votes/ties, forward/reverse/diagonal crossings, jitter/no crossing/finite endpoints, duplicate frames, reappearance, ROI, simultaneous tracks, frozen event classes, atomic publication and generated-video integration. Ordinary tests use no checkpoint/network. Test collection now lazily imports Ultralytics tracking to avoid its global Pillow patch changing earlier corruption tests.

No manual counting GT is yet available. Smoke and full-pilot observations/results will be documented separately. No dashboard is included.

## Smoke review and frozen pilot decision

Actual-checkpoint MPS smoke processed300frames/10seconds successfully. Four sampled annotated frames (1,4,7,9seconds) were inspected. Visible issues include oversized false boxes on the gantry, headlight-region confusion, car/two-wheeler class instability and fragmented IDs. This is a substantial night-domain shift from UVH-26 validation; no accuracy claim follows from event generation. Line/settings remain unchanged for the full30second pilot. Detector confidence stays.25 as requested, so ByteTrack's below.25 low-confidence recovery stage receives few/no detections; this limitation is documented rather than silently changing detector threshold. No GT-driven tuning occurred.
