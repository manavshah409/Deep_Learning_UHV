"""ByteTrack adapter with observation-only history and deterministic class votes."""
from dataclasses import dataclass, field
from types import SimpleNamespace
import numpy as np


@dataclass
class Track:
    track_id: int
    first_frame: int
    last_frame: int
    observations: int = 0
    scores: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    counted: set = field(default_factory=set)
    frozen_events: list = field(default_factory=list)
    confidence: float = 0.0

    @property
    def class_id(self):
        return min(self.scores, key=lambda c: (-self.scores[c], c))

    @property
    def anchor(self):
        return self.history[-1]['anchor']

    @property
    def box(self):
        return self.history[-1]['box']

    def observe(self, frame, box, confidence, cls):
        self.last_frame = frame
        self.observations += 1
        self.scores[cls] = self.scores.get(cls, 0.) + confidence
        self.confidence = confidence
        self.history.append(dict(frame=frame,box=list(map(float,box)),anchor=[float((box[0]+box[2])/2),float(box[3])]))


class VehicleTracker:
    def __init__(self, config, maximum_absence=30):
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker
        self.boxes_type = Boxes
        self.tracker = BYTETracker(SimpleNamespace(**config))
        self.maximum_absence = maximum_absence
        self.tracker.max_frames_lost = maximum_absence
        self.active = {}
        self.finished = []
        self.last_frame = -1

    def update(self, detections, frame, shape):
        if frame <= self.last_frame:
            return []  # duplicate/nonmonotonic frame cannot accumulate votes/events
        if self.last_frame >= 0 and frame != self.last_frame + 1:
            raise ValueError('Tracker expects sequential frames, including empty frames')
        detections = np.asarray(detections, dtype=np.float32)
        if detections.ndim != 2 or detections.shape[1] != 6 or not np.isfinite(detections).all():
            raise ValueError('Expected finite Nx6 detections')
        if len(detections) and (np.any(detections[:,2:4] <= detections[:,:2]) or np.any((detections[:,4]<0)|(detections[:,4]>1)) or np.any((detections[:,5]<0)|(detections[:,5]>13)|(detections[:,5]!=np.floor(detections[:,5])))):
            raise ValueError('Invalid detection geometry/confidence/class')
        self.last_frame = frame
        result = self.tracker.update(self.boxes_type(detections, shape))
        return self.accept(result, frame)

    def accept(self, result, frame):
        result = np.asarray(result)
        if result.size == 0:
            result = np.empty((0,8))
        if result.ndim != 2 or result.shape[1] != 8 or not np.isfinite(result).all():
            raise ValueError('Malformed ByteTrack output; expected xyxy,id,score,class,index')
        ids = result[:,4]
        if len(set(ids)) != len(ids) or np.any(ids<1) or np.any(ids!=np.floor(ids)):
            raise ValueError('Invalid or duplicate track IDs')
        for row in result:
            x1,y1,x2,y2,tid,conf,cls,_ = row
            if x2<=x1 or y2<=y1 or not 0<=conf<=1 or not 0<=cls<14 or cls!=int(cls):
                raise ValueError('Invalid track output fields')
        for tid,t in list(self.active.items()):
            if frame-t.last_frame>self.maximum_absence:
                self.finished.append(self.active.pop(tid))
        observed=[]
        for row in result:
            box,tid,conf,cls=row[:4],int(row[4]),float(row[5]),int(row[6])
            if tid not in self.active:
                self.active[tid]=Track(tid,frame,frame)
            t=self.active[tid]
            t.observe(frame,box,conf,cls)
            observed.append(t)
        return observed

    def all_tracks(self):
        return self.finished + list(self.active.values())
