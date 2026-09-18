"""Finite-line hysteresis counting on observed confirmed track anchors."""
import csv
from pathlib import Path
import cv2
import numpy as np
from src.video.pipeline import CLASS_NAMES,atomic_json


class LineCounter:
    def __init__(self, endpoints, minimum_age=3, hysteresis=5, maximum_gap=1, mode='per_direction', roi=None, tolerance=0):
        self.a,self.b=np.asarray(endpoints,dtype=float)
        self.vector=self.b-self.a
        self.length=float(np.linalg.norm(self.vector))
        if self.length<=0 or not np.isfinite([self.a,self.b]).all():raise ValueError('Invalid line')
        if minimum_age<1 or hysteresis<0 or maximum_gap<1 or mode not in ['per_direction','total']:raise ValueError('Invalid counting configuration')
        self.minimum_age=minimum_age;self.hysteresis=hysteresis;self.maximum_gap=maximum_gap;self.mode=mode;self.roi=roi;self.tolerance=tolerance
        self.states={};self.events=[]

    def side(self,p):
        delta=np.asarray(p)-self.a
        return float((self.vector[0]*delta[1]-self.vector[1]*delta[0])/self.length)

    def update(self,track,frame,fps,identity):
        p=np.asarray(track.anchor)
        old=self.states.get(track.track_id)
        if old and frame<=old['seen']:return None
        if self.roi is not None and cv2.pointPolygonTest(np.asarray(self.roi,dtype=np.float32),tuple(map(float,p)),False)<0:
            self.states.pop(track.track_id,None);return None
        if old and frame-old['seen']>self.maximum_gap:
            old=None;self.states.pop(track.track_id,None)
        if old:old['seen']=frame
        distance=self.side(p)
        if abs(distance)<=self.hysteresis:return None
        sign=1 if distance>0 else -1
        self.states[track.track_id]=dict(sign=sign,p=p.copy(),frame=frame,seen=frame)
        if not old or old['sign']==sign or track.observations<self.minimum_age:return None
        direction='A_to_B' if old['sign']<sign else 'B_to_A'
        if direction in track.counted or (self.mode=='total' and track.counted):return None
        d0=self.side(old['p']);alpha=d0/(d0-distance)
        cross=old['p']+alpha*(p-old['p'])
        along=float(np.dot(cross-self.a,self.vector)/self.length)
        if not -self.tolerance<=along<=self.length+self.tolerance:return None
        track.counted.add(direction)
        cls=track.class_id
        event=dict(**identity,event_id=f'{identity["run_id"]}_{len(self.events)+1:06}',frame=frame,timestamp_seconds=frame/fps,interpolated_crossing_timestamp_seconds=(old['frame']+alpha*(frame-old['frame']))/fps,track_id=track.track_id,class_id=cls,class_name=CLASS_NAMES[cls],direction=direction,confidence=track.confidence,crossing_x=float(cross[0]),crossing_y=float(cross[1]))
        track.frozen_events.append(event.copy());self.events.append(event)
        return event


def export_events(directory,events):
    """Both files published before run COMPLETE marker; consumers must require marker."""
    directory=Path(directory)
    fields=['event_id','run_id','video_id','video_sha256','frame','timestamp_seconds','interpolated_crossing_timestamp_seconds','track_id','class_id','class_name','direction','confidence','crossing_x','crossing_y','line_id','tracker','checkpoint_sha256']
    atomic_json(directory/'crossing_events.json',events)
    temporary=directory/'crossing_events.csv.tmp'
    with temporary.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(events)
    temporary.replace(directory/'crossing_events.csv')
