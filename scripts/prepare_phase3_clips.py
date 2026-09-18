"""Decode verified local Commons originals; create new 30-second MP4 derivatives."""
from pathlib import Path
import json
import sys
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.common import ROOT,sha256


def main():
    originals=json.loads((ROOT/'data/videos/external/accepted_metadata.json').read_text())
    out=ROOT/'data/videos/derived';out.mkdir(exist_ok=True)
    records=[]
    for m in originals:
        source=ROOT/'data/videos/external'/m['filename']
        target=out/(m['video_id']+'_0_30_v2.mp4')
        if target.exists():raise FileExistsError(target)
        cap=cv2.VideoCapture(str(source)); writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),m['fps'],(m['width'],m['height']))
        assert writer.isOpened()
        count=0;changes=[];previous=None;timestamps=[]
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                if count<int(30*m['fps']):writer.write(frame)
                if count%int(m['fps'])==0:
                    small=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(160,90))
                    if previous is not None:changes.append(float(cv2.absdiff(small,previous).mean()))
                    previous=small
                count+=1
        finally:cap.release();writer.release()
        import numpy as np
        intervals=np.diff(timestamps)
        if count<int(30*m['fps']) or min(intervals)<=0 or max(intervals)>1.5*1000/m['fps']:
            raise ValueError('Insufficient frames or non-contiguous timestamps')
        records.append(dict(video_id=m['video_id'],source_sha256=m['sha256'],source_decoded_frames=count,reported_frame_count=m['frames'],timestamp_interval_ms=[float(min(intervals)),float(max(intervals))],last_timestamp_ms=timestamps[-1],start_seconds=0,end_seconds=30,filename=target.name,sha256=sha256(target),reencoded=True,audio_removed=True,command='.venv/bin/python scripts/prepare_phase3_clips.py',codec='mp4v',max_one_second_gray_change=max(changes)))
    (ROOT/'data/videos/derived/provenance.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
