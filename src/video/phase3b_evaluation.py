"""Frozen closest-time event comparison; no automatic manual ground truth."""

import math
from collections import Counter

from src.video.phase3b_annotations import DIRECTIONS, validate
from src.video.pipeline import CLASS_NAMES


def counts(predicted, actual, tp):
    fp, fn = predicted - tp, actual - tp
    precision = tp / predicted if predicted else 0.0
    recall = tp / actual if actual else 0.0
    return {
        "ground_truth": actual,
        "predicted": predicted,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "signed_error": predicted - actual,
        "absolute_error": abs(predicted - actual),
        "percentage_error": 100 * abs(predicted - actual) / actual if actual else None,
        "overcount_rate": max(predicted - actual, 0) / actual if actual else None,
        "undercount_rate": max(actual - predicted, 0) / actual if actual else None,
    }


def evaluate(predictions, document, protocol):
    validate(document, protocol, final=True)
    actual = document["events"]
    video = protocol["payload"]["video"]
    tolerance = protocol["payload"]["matching"]["tolerance_seconds"]
    ids = set()
    for p in predictions:
        if p["event_id"] in ids:
            raise ValueError("Duplicate prediction event ID")
        ids.add(p["event_id"])
        if (
            p["video_id"] != video["video_id"]
            or type(p["class_id"]) is not int
            or not 0 <= p["class_id"] < 14
            or p["direction"] not in DIRECTIONS
            or not math.isfinite(p["timestamp_seconds"])
            or not 0 <= p["timestamp_seconds"] < video["duration_seconds"]
        ):
            raise ValueError("Invalid prediction event")
    # Greedy global nearest pair. Ties: GT timestamp, GT event ID, prediction ID.
    # Explicitly not a maximum-cardinality assignment algorithm.
    candidates = []
    for ai, a in enumerate(actual):
        for pi, p in enumerate(predictions):
            delta = abs(p["timestamp_seconds"] - a["timestamp_seconds"])
            if (
                p["class_id"] == a["class_id"]
                and p["direction"] == a["direction"]
                and delta <= tolerance
            ):
                candidates.append(
                    (
                        delta,
                        a["timestamp_seconds"],
                        a["event_id"],
                        p["event_id"],
                        ai,
                        pi,
                    )
                )
    used_a, used_p, matches = set(), set(), []
    for delta, _, _, _, ai, pi in sorted(candidates):
        if ai not in used_a and pi not in used_p:
            used_a.add(ai)
            used_p.add(pi)
            matches.append(
                {
                    "prediction_index": pi,
                    "annotation_index": ai,
                    "absolute_time_error_seconds": delta,
                }
            )
    result = {
        "status": "human_reviewed_pilot",
        "total": counts(len(predictions), len(actual), len(matches)),
        "matches": matches,
    }
    for key, values, label in [
        ("class_id", range(14), "per_class"),
        ("direction", DIRECTIONS, "per_direction"),
    ]:
        result[label] = {
            str(v): counts(
                sum(p[key] == v for p in predictions),
                sum(a[key] == v for a in actual),
                sum(actual[m["annotation_index"]][key] == v for m in matches),
            )
            for v in values
        }
    groups = []
    for cls in range(14):
        for direction in DIRECTIONS:
            pc = sum(
                p["class_id"] == cls and p["direction"] == direction
                for p in predictions
            )
            ac = sum(
                a["class_id"] == cls and a["direction"] == direction for a in actual
            )
            groups.append(
                {
                    "class_name": CLASS_NAMES[cls],
                    "direction": direction,
                    "predicted": pc,
                    "ground_truth": ac,
                    "signed_error": pc - ac,
                    "absolute_error": abs(pc - ac),
                    "percentage_error": 100 * abs(pc - ac) / ac if ac else None,
                }
            )
    result["class_direction"] = groups
    sae = sum(g["absolute_error"] for g in groups)
    result["mae_28_class_direction_groups"] = sae / 28
    result["weighted_absolute_percentage_error"] = (
        100 * sae / len(actual) if actual else None
    )
    result["crossing_time_mae_seconds"] = (
        sum(m["absolute_time_error_seconds"] for m in matches) / len(matches)
        if matches
        else None
    )
    review = []
    for m in matches:
        p, a = predictions[m["prediction_index"]], actual[m["annotation_index"]]
        review.append(
            {
                "category": "correct_crossing",
                "predicted": p,
                "ground_truth": a,
                "review_status": "matching_result_not_visual_review",
            }
        )
    for i, p in enumerate(predictions):
        if i not in used_p:
            review.append(
                {
                    "category": "false_positive_crossing",
                    "predicted": p,
                    "ground_truth": None,
                    "review_status": "visual_review_required",
                }
            )
    for i, a in enumerate(actual):
        if i not in used_a:
            review.append(
                {
                    "category": "missed_crossing",
                    "predicted": None,
                    "ground_truth": a,
                    "review_status": "visual_review_required",
                }
            )
    result["review_gallery"] = review
    result["error_category_counts"] = dict(Counter(r["category"] for r in review))
    result["diagnostics"] = {
        "duplicate_counts": None,
        "fragmentation": None,
        "id_switches": None,
        "class_confusion": None,
        "direction_errors": None,
        "note": "Require visual review; event labels alone do not establish identity errors. No formal MOT score.",
    }
    return result
