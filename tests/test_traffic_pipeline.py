import cv2
import numpy as np
import yaml
import json
from pathlib import Path
from src.video.traffic import run_traffic
from src.data.common import sha256
import pytest


class MovingDetector:
 device='cpu';warnings=['Synthetic detector; no real checkpoint']
 def __init__(self,cfg):self.i=0
 def sync(self):pass
 def __call__(self,frame):
  self.i+=1;y=10+self.i*2
  return np.array([[20,y-15,40,y,.9,7]],dtype=float)


def test_generated_video_tracking_and_publication(tmp_path):
 p=tmp_path/'test.avi';w=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'MJPG'),10,(100,100))
 assert w.isOpened()
 for _ in range(30):w.write(np.zeros((100,100,3),np.uint8))
 w.release()
 cfg=yaml.safe_load(Path('configs/phase3_video.yaml').read_text())
 cfg.update(input_video=str(p),output_root=str(tmp_path),run_id='integrated',video_id='synthetic')
 cfg['counting']['hysteresis_pixels']=2
 run,s=run_traffic(cfg,MovingDetector)
 assert s['total_crossings']==1 and s['direction_counts']['A_to_B']==1
 assert s['total_unique_confirmed_track_ids']==1
 marker=json.loads((run/'COMPLETE.json').read_text())
 assert all(sha256(run/f)==h for f,h in marker['files_sha256'].items())
 assert not s['checkpoint_verified']
 with pytest.raises(FileExistsError):run_traffic(cfg,MovingDetector)
