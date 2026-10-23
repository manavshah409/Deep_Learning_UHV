import pytest

from src.video.evaluate_events import evaluate


def event(t=1,cls=7):return {'video_id': 'v','timestamp_seconds': t,'direction': 'A_to_B','class_id': cls}


def test_requires_manual_review():
 with pytest.raises(ValueError):evaluate([],[])


def test_event_matching_not_only_total():
 r=evaluate([event(1),event(3)],[event(1.1),event(10)],True)
 assert r['total']['absolute_error']==0 and r['false_crossing_events']==1 and r['missed_crossings']==1


def test_zero_actual_and_uncertain():
 r=evaluate([event()],[dict(**event(),uncertain=True)],True)
 assert r['total']['percentage_error'] is None and r['uncertain_annotations_excluded']==1


def test_class_and_direction_matching():
 r=evaluate([event(1,6)],[event(1,7)],True)
 assert r['matched_crossings']==0 and r['class_wise']['6']['absolute_error']==1
