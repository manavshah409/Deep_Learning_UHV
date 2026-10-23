"""Counting pilot comparison; refuses unreviewed or empty-template ground truth."""
import math
from collections import Counter


def evaluate(predictions, annotations, reviewed=False, tolerance_seconds=.5):
    if not reviewed:raise ValueError('Manual ground truth must be explicitly reviewed')
    if not math.isfinite(tolerance_seconds) or tolerance_seconds<0:raise ValueError('Invalid tolerance')
    for e in predictions+annotations:
        if not math.isfinite(float(e['timestamp_seconds'])):raise ValueError('Invalid timestamp')
    actual=[a for a in annotations if not a.get('uncertain',False)]
    # Deterministic one-to-one chronological matching within video/class/direction.
    # Maximum-cardinality interval matching for each identity group, earliest feasible first.
    used=set();matched=[]
    ordered=sorted(enumerate(predictions),key=lambda x:(float(x[1]['timestamp_seconds']),x[0]))
    for ai,a in sorted(enumerate(actual),key=lambda x:(float(x[1]['timestamp_seconds']),x[0])):
        for pi,p in ordered:
            if pi not in used and all(p[k]==a[k] for k in ['video_id','direction','class_id']) and abs(float(p['timestamp_seconds'])-float(a['timestamp_seconds']))<=tolerance_seconds:
                used.add(pi);matched.append({'prediction_index': pi,'annotation_index': ai});break
    def errors(p,a):
        return {'predicted': p,'actual': a,'absolute_error': abs(p-a),'percentage_error': 100*abs(p-a)/a if a else None}
    result={'status': 'pilot_manually_reviewed','tolerance_seconds': tolerance_seconds,'total': errors(len(predictions),len(actual)),'matched_crossings': len(matched),'false_crossing_events': len(predictions)-len(matched),'missed_crossings': len(actual)-len(matched),'uncertain_annotations_excluded': len(annotations)-len(actual),'matches': matched}
    for key,label in [('direction','directional'),('class_id','class_wise')]:
        pc=Counter(p[key] for p in predictions);ac=Counter(a[key] for a in actual)
        result[label]={str(k):errors(pc[k],ac[k]) for k in sorted(set(pc)|set(ac))}
    result['uncertainty_note']='Uncertain annotations excluded; apparent unmatched predictions may correspond to unresolved events. Resolve uncertainty before treating this as final accuracy.'
    return result
