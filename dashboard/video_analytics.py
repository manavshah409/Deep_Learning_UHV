"""Saved Phase 3B evidence and independent human annotation, with no inference."""

import functools
import hashlib
import json
import operator
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data.common import ROOT, sha256


def load_protocol(path):
    value = json.loads(Path(path).read_text())
    digest = hashlib.sha256(
        json.dumps(value["payload"], sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if digest != value["sha256"]:
        raise ValueError("Protocol integrity check failed")
    return value


def saved_evidence(expected_protocol=None):
    st.subheader("Saved Evidence Mode")
    st.caption(
        "Reading recorded results does not run a detector. Predictions are not manual ground truth."
    )
    path = ROOT / "reports/phase3b/benchmark_summary.json"
    if not path.exists():
        st.info("No completed Phase 3B benchmark is available yet.")
        return
    evidence = json.loads(path.read_text())
    if expected_protocol and evidence["protocol_sha256"] != expected_protocol:
        st.warning(
            "Saved benchmark belongs to a different protocol. Select its protocol before viewing."
        )
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "mode": r["mode"],
                    "frames": r["frames"],
                    "device": r["device"],
                    "fps": r["effective_fps"],
                    "p95_ms": r["latency_ms"]["total_ms"]["p95"],
                }
                for r in evidence["runs"]
            ]
        ),
        hide_index=True,
    )
    row = next((r for r in evidence["runs"] if r["mode"] == "complete"), None)
    if not row:
        return
    a, b, c = st.columns(3)
    a.metric("Frame detection instances", row["detection_instances"])
    b.metric("Maximum observed active tracks", row["maximum_observed_active_tracks"])
    c.metric("Unique predicted crossing events", row["predicted_crossings"])
    st.write(
        "One event per track per direction. A returning track may contribute two directions; this is not a guaranteed count of unique physical vehicles."
    )
    st.write(
        f"Complete pipeline: {row['effective_fps']:.2f} FPS; source {row['source_fps']:.0f} FPS. Target {'passed' if row['realtime_target_pass'] else 'not passed'}."
    )
    st.dataframe(pd.DataFrame(row["class_direction"]), hide_index=True)
    st.bar_chart(pd.Series(row["direction_counts"], name="Predicted crossings"))
    run_dir = ROOT / "runs/phase3b" / row["run_id"]
    if (run_dir / "frame_timings.csv").exists():
        timeline = pd.read_csv(run_dir / "frame_timings.csv")
        st.line_chart(timeline.set_index("timestamp_seconds")[["cumulative_crossings"]])
    preview = ROOT / "runs/phase3b_previews" / row["run_id"] / "annotated_browser.mp4"
    if preview.exists():
        st.video(str(preview))
        st.caption(
            "Browser-compatible presentation copy; transcoding is separate from measured mp4v pipeline encoding."
        )
    elif (run_dir / "annotated.mp4").exists():
        st.download_button(
            "Download annotated video for local playback",
            (run_dir / "annotated.mp4").read_bytes(),
            "annotated.mp4",
        )
        st.caption(
            "Measured mp4v output may require a local player. Browser transcoding is not part of measured processing."
        )
    pointer = ROOT / "reports/phase3b/reviewed_result.json"
    if pointer.exists():
        info = json.loads(pointer.read_text())
        if (
            info["protocol_sha256"] != evidence["protocol_sha256"]
            or info["run_id"] != row["run_id"]
        ):
            raise ValueError("Saved reviewed result protocol/run mismatch")
        st.subheader("Manually reviewed counting evaluation")
        st.json(info["total"])
        st.dataframe(pd.DataFrame(info["class_direction"]), hide_index=True)
        st.subheader("Error gallery review candidates")
        for item in info.get("review_gallery", []):
            with st.expander(item["category"]):
                st.json(item)
                event = item["predicted"] or item["ground_truth"]
                st.caption(
                    f"Inspect source frame {event['frame']}; {item['review_status']}. Do not infer ID switches without visual evidence."
                )
    else:
        st.warning(
            "Human ground truth is pending. Crossing precision, recall, F1 and count error are unavailable."
        )
        st.info(
            "Error gallery categories such as duplicate counts, fragmentation, ID switches, class confusion, jitter and occlusion require reviewed evidence. Predicted events are review candidates only."
        )
        events_path = run_dir / "crossing_events.json"
        if events_path.exists():
            events = json.loads(events_path.read_text())
            if events:
                event = st.selectbox(
                    "Predicted event to inspect",
                    events,
                    format_func=lambda e: (
                        f"Frame {e['frame']} · track {e['track_id']} · {e['class_name']} · {e['direction']}"
                    ),
                )
                image = run_dir / "evidence" / f"{event['event_id']}.jpg"
                if image.exists():
                    st.image(
                        str(image),
                        caption="Predicted crossing source frame; not a human-reviewed correct event",
                    )
    with st.expander("Technical timing and limitations"):
        st.json(row)
    st.download_button(
        "Download small benchmark evidence", path.read_bytes(), "phase3b_benchmark.json"
    )


def frame_view(path, frame, endpoints):
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Local permitted video is missing or unreadable")
        # Sequential grab establishes exact frame numbering, independent of codec seeking.
        for _ in range(frame):
            if not cap.grab():
                raise ValueError("Cannot reach requested frame")
        ok, image = cap.read()
        if not ok:
            raise ValueError("Frame decode failed")
        points = [
            (int(x * image.shape[1]), int(y * image.shape[0])) for x, y in endpoints
        ]
        cv2.line(image, *points, (0, 0, 255), 3)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def annotation_tool(protocol):
    st.subheader("Independent manual crossing annotation")
    st.warning(
        "Annotate the entire source clip before viewing model predictions. Use a reviewer code, not personal information. Resolve uncertain events before final evaluation."
    )
    p = protocol["payload"]
    v = p["video"]
    source = ROOT / v["filename"]
    if not source.exists():
        st.info(
            "The permitted local video is required for annotation; it is not bundled in Git."
        )
        return
    from src.video.phase3b_annotations import (
        DIRECTIONS,
        as_csv,
        load_revision,
        save_revision,
    )
    from src.video.pipeline import CLASS_NAMES

    if st.session_state.get("phase3b_verified_source") != protocol["sha256"]:
        if sha256(source) != v["sha256"]:
            raise ValueError("Annotation video hash mismatch")
        st.session_state.phase3b_verified_source = protocol["sha256"]
    key = "phase3b_document_" + protocol["sha256"]
    if key not in st.session_state:
        st.session_state[key] = {
            "protocol_sha256": protocol["sha256"],
            "annotator": "reviewer01",
            "events": [],
            "human_review_complete": False,
            "whole_video_reviewed": False,
            "predictions_seen_before_annotation": False,
            "confirmed_zero_events": False,
        }
    doc = st.session_state[key]
    reviewer = st.text_input("Anonymous reviewer code", value=doc["annotator"])
    parent = ROOT / "runs/phase3b_annotations" / protocol["sha256"][:12]
    revisions = (
        sorted(
            [x for x in parent.glob("*") if (x / "COMPLETE.json").exists()],
            reverse=True,
        )
        if parent.exists()
        else []
    )
    if revisions:
        revision = st.selectbox(
            "Completed annotation revision", revisions, format_func=lambda x: x.name
        )
        if st.button("Resume this revision"):
            st.session_state[key] = load_revision(revision, protocol)
            st.rerun()
    st.caption(
        "Frozen line: "
        + str(p["counting"]["line_normalized"])
        + " normalized. A→B is downward; B→A upward for this horizontal line. Record the first frame bottom-center passes the line, including missed vehicles."
    )
    with st.expander("Choose a different line for a new protocol"):
        coords = st.columns(4)
        vals = [
            c.number_input(
                label,
                min_value=0.0,
                max_value=1.0,
                value=float(value),
                key="line_" + label,
            )
            for c, label, value in zip(
                coords,
                ["x1", "y1", "x2", "y2"],
                functools.reduce(operator.iadd, p["counting"]["line_normalized"], []),
            )
        ]
        st.code(
            ".venv/bin/python -m src.video.phase3b freeze --protocol reports/phase3b/protocol_v2.json --line "
            + " ".join(map(str, vals))
        )
        st.caption(
            "A new line requires a new protocol, new run and new annotation set. The current protocol is not overwritten."
        )
    frame = st.number_input(
        "Frame number",
        min_value=0,
        max_value=v["frames"] - 1,
        value=0,
        step=1,
        key="p3b_frame",
    )
    c1, c2, c3 = st.columns(3)

    def step(delta):
        st.session_state.p3b_frame = max(
            0, min(v["frames"] - 1, st.session_state.p3b_frame + delta)
        )

    c1.button("Previous frame", on_click=step, args=(-1,))
    c2.button("Next frame", on_click=step, args=(1,))
    seconds = c3.number_input(
        "Jump timestamp in seconds", 0.0, (v["frames"] - 1) / v["fps"], 0.0
    )

    def jump():
        st.session_state.p3b_frame = round(seconds * v["fps"])

    c3.button("Jump", on_click=jump)
    st.image(
        frame_view(source, frame, p["counting"]["line_normalized"]),
        caption=f"Frame {frame}, {frame / v['fps']:.3f}s. Source plus line only; no predictions.",
    )
    if st.button("Play next 30 frames slowly"):
        display = st.empty()
        for i in range(frame, min(frame + 30, v["frames"])):
            display.image(
                frame_view(source, i, p["counting"]["line_normalized"]),
                caption=f"Frame {i}, {i / v['fps']:.3f}s",
            )
            time.sleep(0.1)
    with st.form("event_form"):
        cls = st.selectbox("Vehicle class", CLASS_NAMES)
        direction = st.selectbox("Direction", DIRECTIONS)
        uncertain = st.checkbox("Uncertain event")
        notes = st.text_input("Notes without personal information")
        if st.form_submit_button("Record event and save draft"):
            doc.update(
                annotator=reviewer,
                human_review_complete=False,
                whole_video_reviewed=False,
            )
            for e in doc["events"]:
                e.update(annotator=reviewer, review_status="draft")
            doc["events"].append(
                {
                    "event_id": uuid.uuid4().hex[:12],
                    "video_id": v["video_id"],
                    "frame": int(frame),
                    "timestamp_seconds": frame / v["fps"],
                    "class_id": CLASS_NAMES.index(cls),
                    "class_name": cls,
                    "direction": direction,
                    "review_status": "draft",
                    "annotator": reviewer,
                    "uncertain": uncertain,
                    "notes": notes,
                }
            )
            saved = save_revision(parent, doc, protocol)
            st.success("Draft safely saved as " + saved.name)
    if doc["events"]:
        edited = st.data_editor(
            pd.DataFrame(doc["events"]),
            num_rows="dynamic",
            disabled=[
                "event_id",
                "video_id",
                "timestamp_seconds",
                "class_id",
                "review_status",
                "annotator",
            ],
            column_config={
                "class_name": st.column_config.SelectboxColumn(options=CLASS_NAMES),
                "direction": st.column_config.SelectboxColumn(options=DIRECTIONS),
            },
            key="p3b_editor",
        )
        if st.button("Save edits and deletions as new draft revision"):
            doc["events"] = edited.to_dict("records")
            doc.update(
                annotator=reviewer,
                human_review_complete=False,
                whole_video_reviewed=False,
            )
            for e in doc["events"]:
                e.update(
                    frame=int(e["frame"]),
                    timestamp_seconds=int(e["frame"]) / v["fps"],
                    class_id=CLASS_NAMES.index(e["class_name"]),
                    review_status="draft",
                    annotator=reviewer,
                    uncertain=bool(e["uncertain"]),
                )
            save_revision(parent, doc, protocol)
            st.success("New draft revision saved")
    st.caption(
        "Save edits before finalizing. An interrupted partial revision cannot replace a completed revision."
    )
    whole = st.checkbox(
        "I independently reviewed all 900 frames and resolved every uncertain event"
    )
    unseen = st.checkbox(
        "I did not inspect model predictions before this annotation pass"
    )
    zero = st.checkbox(
        "I explicitly confirm zero crossings if this event list is empty"
    )
    if st.button("Finalize human review as a new revision"):
        import copy

        final = copy.deepcopy(doc)
        final.update(
            annotator=reviewer,
            human_review_complete=True,
            whole_video_reviewed=whole,
            predictions_seen_before_annotation=not unseen,
            confirmed_zero_events=zero,
        )
        for e in final["events"]:
            e.update(annotator=reviewer, review_status="reviewed")
        saved = save_revision(parent, final, protocol)
        st.session_state[key] = final
        st.success("Human review revision saved: " + str(saved.relative_to(ROOT)))
    st.download_button(
        "Export annotation JSON",
        json.dumps(st.session_state[key], indent=2),
        "annotations.json",
    )
    st.download_button(
        "Export annotation CSV", as_csv(st.session_state[key]), "annotations.csv"
    )


def render():
    st.header("Video Traffic Analytics")
    protocols = sorted((ROOT / "reports/phase3b").glob("protocol_*.json"))
    if not protocols:
        st.info("Phase 3B protocol has not been frozen yet.")
        return
    selected = st.selectbox(
        "Frozen video protocol",
        protocols,
        index=len(protocols) - 1,
        format_func=lambda x: x.name,
    )
    protocol = load_protocol(selected)
    st.write(protocol["payload"]["video"])
    st.caption(
        "Local footage only. No training or inference starts from this page. Counting validation remains pending until independent human review."
    )
    choice = st.radio("Video workspace", ["Manual annotation", "Saved evidence"])
    try:
        if choice == "Manual annotation":
            annotation_tool(protocol)
        else:
            saved_evidence(protocol["sha256"])
    except (ValueError, OSError, KeyError, TypeError) as e:
        st.error(str(e))
