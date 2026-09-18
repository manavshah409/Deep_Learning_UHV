"""Streaming frozen-detector + ByteTrack + finite-line pilot. No dashboard."""
import argparse
import csv
import json
import time
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import yaml
from src.data.common import ROOT,sha256
from src.video.pipeline import FrozenDetector,validate_config,atomic_json,CLASS_NAMES
from src.video.tracking import VehicleTracker
from src.video.counting import LineCounter,export_events


def stats(values):
    return dict(mean=float(np.mean(values)),median=float(np.median(values)),p95=float(np.percentile(values,95))) if values else None


def run_traffic(cfg, detector_factory=FrozenDetector):
    validate_config(cfg)
    source=ROOT/cfg['input_video']
    if not source.is_file():raise FileNotFoundError(source)
    run=ROOT/cfg['output_root']/cfg['run_id'];run.mkdir(parents=True,exist_ok=False)
    cap=writer=None;rows=[]
    try:
        atomic_json(run/'config.json',cfg)
        detector=detector_factory(cfg)
        source_hash=sha256(source)
        if cfg.get('input_sha256') and source_hash!=cfg['input_sha256']:raise ValueError('Video hash mismatch')
        cap=cv2.VideoCapture(str(source))
        if not cap.isOpened():raise ValueError('Video failed to open')
        fps=float(cap.get(cv2.CAP_PROP_FPS));total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w,h=int(cap.get(3)),int(cap.get(4))
        if not np.isfinite(fps) or fps<=0 or min(w,h)<=0:raise ValueError('Invalid video metadata')
        start=int(cfg['start_seconds']*fps)
        if start>=total:raise ValueError('Start past end')
        for _ in range(start):
            if not cap.grab():raise ValueError('Failed seeking start by decode')
        tracker_cfg=yaml.safe_load((ROOT/cfg['tracker']['config']).read_text())
        if cfg['tracker']['type']!='bytetrack':raise ValueError('Only ByteTrack supported')
        cc=cfg['counting'];tracker=VehicleTracker(tracker_cfg,cc['maximum_absence_frames'])
        endpoints=np.asarray(cc['line_normalized'],dtype=float)*[w,h]
        roi=np.asarray(cc['roi_normalized'])*[w,h] if cc.get('roi_normalized') else None
        counter=LineCounter(endpoints,cc['minimum_confirmed_age'],cc['hysteresis_pixels'],cc.get('maximum_crossing_gap_frames',1),cc.get('mode','per_direction'),roi,cc.get('endpoint_tolerance_pixels',0))
        identity=dict(run_id=cfg['run_id'],video_id=cfg['video_id'],video_sha256=source_hash,line_id='L1',tracker='ByteTrack',checkpoint_sha256=cfg['checkpoint_sha256'])
        if cfg['render']:
            writer=cv2.VideoWriter(str(run/cfg['output_video']),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
            if not writer.isOpened():raise ValueError('Encoder unavailable')
        stages=['decode_ms','detector_ms','tracker_ms','analytics_ms','render_encode_ms','total_ms']
        begin=time.perf_counter();frame_id=start;max_active=0
        with (run/'frame_timings.csv').open('w',newline='') as tf:
            tw=csv.DictWriter(tf,fieldnames=['frame','timestamp_seconds','active_tracks','occupancy','crossings']+stages);tw.writeheader()
            while frame_id<total:
                if cfg.get('max_frames') and len(rows)>=cfg['max_frames']:break
                if cfg.get('end_seconds') is not None and frame_id/fps>=cfg['end_seconds']:break
                t0=time.perf_counter();ok,frame=cap.read();t1=time.perf_counter()
                if not ok or frame is None or frame.shape[:2]!=(h,w):raise ValueError(f'Frame decode failure at {frame_id}')
                detector.sync();d0=time.perf_counter();detections=detector(frame);detector.sync();t2=time.perf_counter()
                observed=tracker.update(detections,frame_id,(h,w));t3=time.perf_counter()
                confirmed=[t for t in observed if t.observations>=cc['minimum_confirmed_age']]
                # Feed all observed tracks to establish pre-confirmation side history.
                for t in observed:counter.update(t,frame_id,fps,identity)
                live_ids=set(tracker.active)
                counter.states={tid:s for tid,s in counter.states.items() if tid in live_ids}
                occupancy=sum(roi is None or cv2.pointPolygonTest(np.asarray(roi,dtype=np.float32),tuple(t.anchor),False)>=0 for t in confirmed)
                max_active=max(max_active,len(confirmed));t4=time.perf_counter()
                if writer:
                    cv2.line(frame,tuple(endpoints[0].astype(int)),tuple(endpoints[1].astype(int)),(0,0,255),2)
                    for t in confirmed:
                        x1,y1,x2,y2=map(int,t.box)
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,220,0),2)
                        cv2.putText(frame,f'ID{t.track_id} {CLASS_NAMES[t.class_id]}',(x1,max(15,y1)),0,.5,(0,220,0),1)
                    cv2.putText(frame,f'Crossings {len(counter.events)} | frame {frame_id}',(15,30),0,.7,(0,255,255),2)
                    writer.write(frame)
                t5=time.perf_counter()
                row=dict(frame=frame_id,timestamp_seconds=frame_id/fps,active_tracks=len(confirmed),occupancy=occupancy,crossings=len(counter.events),decode_ms=1000*(t1-t0),detector_ms=1000*(t2-d0),tracker_ms=1000*(t3-t2),analytics_ms=1000*(t4-t3),render_encode_ms=1000*(t5-t4),total_ms=1000*(t5-t0))
                tw.writerow(row);rows.append(row);frame_id+=1
        if writer:writer.release();writer=None
        streaming_seconds=time.perf_counter()-begin
        if not rows:raise ValueError('No processed frames')
        export_events(run,counter.events)
        all_tracks=tracker.all_tracks()
        track_summary=[dict(track_id=t.track_id,class_id=t.class_id,class_name=CLASS_NAMES[t.class_id],class_scores=t.scores,first_frame=t.first_frame,last_frame=t.last_frame,observations=t.observations,duration_seconds=(t.last_frame-t.first_frame+1)/fps,counted_directions=sorted(t.counted),frozen_events=t.frozen_events,history=t.history) for t in all_tracks]
        atomic_json(run/'tracks.json',track_summary)
        cls=Counter(e['class_name'] for e in counter.events);directions=Counter(e['direction'] for e in counter.events);minute=Counter(str(int(e['timestamp_seconds']//60)) for e in counter.events)
        rolling=[]
        window=cfg.get('rolling_window_seconds',10)
        for r in rows:
            t=r['timestamp_seconds'];exposure=min(window,(r['frame']-start+1)/fps)
            count=sum(max(start/fps,t-window)<e['timestamp_seconds']<=t for e in counter.events)
            rolling.append(dict(frame=r['frame'],timestamp_seconds=t,crossings_last_window=count,exposure_seconds=exposure,flow_per_minute=count*60/exposure,active_tracks=r['active_tracks'],occupancy=r['occupancy']))
        atomic_json(run/'flow.json',rolling)
        confirmed_all=[t for t in all_tracks if t.observations>=cc['minimum_confirmed_age']]
        thresholds=cfg.get('density_thresholds',dict(medium=10,high=25))
        average=float(np.mean([r['occupancy'] for r in rows]))
        processing=time.perf_counter()-begin
        summary=dict(streaming_seconds=streaming_seconds,final_aggregation_export_seconds=processing-streaming_seconds,status='complete',scope='video_tracking_counting_pilot_unverified_ground_truth',run_id=cfg['run_id'],video_id=cfg['video_id'],source_sha256=source_hash,checkpoint_sha256=cfg['checkpoint_sha256'],checkpoint_verified=isinstance(detector,FrozenDetector),device=detector.device,tracker='ByteTrack',tracker_config=tracker_cfg,confidence=cfg['confidence'],nms_iou=cfg['nms_iou'],max_det=cfg['max_det'],imgsz=640,resolution=[w,h],source_fps=fps,source_frames=total,processed_frames=len(rows),processed_video_seconds=len(rows)/fps,processing_seconds=processing,processing_fps=len(rows)/processing,encoding_included=cfg['render'],model_loading_included=False,failed_frames=0,dropped_frames=0,total_unique_confirmed_track_ids=len(confirmed_all),total_crossings=len(counter.events),direction_counts={d:directions[d] for d in ['A_to_B','B_to_A']},class_counts={c:cls[c] for c in CLASS_NAMES},per_minute_counts=minute,composition_percent={c:100*cls[c]/len(counter.events) if counter.events else 0 for c in CLASS_NAMES},maximum_simultaneously_observed_confirmed_tracks=max_active,occupancy=dict(scope='ROI' if roi is not None else 'whole frame',mean=average,maximum=max(r['occupancy'] for r in rows)),density=dict(label='high' if average>=thresholds['high'] else 'medium' if average>=thresholds['medium'] else 'low',thresholds=thresholds,method='manually chosen average observed confirmed-track occupancy; not physical traffic density'),track_duration_seconds=stats([(t.last_frame-t.first_frame+1)/fps for t in confirmed_all]),latency_ms={s:stats([r[s] for r in rows]) for s in stages},warnings=detector.warnings,timing_note='Sequential streaming; sync MPS; detector includes preprocess/NMS; first-frame startup included; wall includes streaming, encoding finalization, event/track/flow export and aggregation; model load/hash/seek, final summary serialization and completion hashing excluded. No inference-only claim.',privacy='No face/plate analysis; IDs are run-local association labels')
        atomic_json(run/'summary.json',summary)
        files=['config.json','frame_timings.csv','crossing_events.csv','crossing_events.json','tracks.json','flow.json','summary.json']
        if cfg['render']:files.append(cfg['output_video'])
        atomic_json(run/'COMPLETE.json',dict(files_sha256={f:sha256(run/f) for f in files}))
        return run,summary
    except Exception as e:
        atomic_json(run/'FAILURE.json',dict(error_type=type(e).__name__,message=str(e),processed_frames=len(rows)))
        raise
    finally:
        if cap is not None:cap.release()
        if writer is not None:writer.release()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run-id');p.add_argument('--max-frames',type=int);p.add_argument('--no-render',action='store_true')
    a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text())
    if a.run_id:cfg['run_id']=a.run_id
    if a.max_frames:cfg['max_frames']=a.max_frames
    if a.no_render:cfg['render']=False
    _,summary=run_traffic(cfg);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
