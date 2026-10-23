"""Create a source-only line/frame overlay for independent manual review."""
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.common import ROOT


def main():
    cfg=yaml.safe_load((ROOT/'configs/phase3a_bangalore01.yaml').read_text())
    target=ROOT/'data/videos/derived/bangalore01_manual_review_v1.mp4'
    if target.exists():raise FileExistsError(target)
    cap=cv2.VideoCapture(str(ROOT/cfg['input_video']))
    fps=cap.get(cv2.CAP_PROP_FPS);w,h=int(cap.get(3)),int(cap.get(4))
    writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not cap.isOpened() or not writer.isOpened():raise RuntimeError('Video open/encode failed')
    line=[tuple(int(x*s) for x,s in zip(p,[w,h])) for p in cfg['counting']['line_normalized']]
    count=0
    try:
        while True:
            ok,f=cap.read()
            if not ok:break
            cv2.line(f,*line,(0,0,255),2)
            cv2.putText(f,f'Frame {count} | {count/fps:.3f}s | No model predictions',(20,35),0,.8,(0,255,255),2)
            writer.write(f);count+=1
    finally:cap.release();writer.release()
    print(f'Created independent review video: {count} frames')


if __name__=='__main__':main()
