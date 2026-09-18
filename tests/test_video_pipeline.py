import copy
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import yaml
from src.video.pipeline import process_video, validate_config, FrozenDetector


class FakeDetector:
    device='cpu'
    warnings=['Synthetic test double: no checkpoint inference']
    def __init__(self,cfg): pass
    def sync(self): pass
    def __call__(self,frame): return np.array([[10,10,30,30,.8,7]],dtype=float)


@pytest.fixture
def video_config(tmp_path):
    source=tmp_path/'synthetic.avi'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,48))
    assert writer.isOpened()
    for i in range(12):
        frame=np.zeros((48,64,3),np.uint8)
        cv2.rectangle(frame,(i+1,10),(i+10,20),(255,255,255),-1)
        writer.write(frame)
    writer.release()
    cfg=yaml.safe_load(Path('configs/phase3_video.yaml').read_text())
    cfg.update(input_video=str(source),output_root=str(tmp_path/'runs'),run_id='test')
    return cfg


def test_streaming_output_and_no_overwrite(video_config):
    run,s=process_video(video_config,FakeDetector)
    assert s['processed_frame_count']==12 and s['failed_reads']==0
    assert not s['checkpoint_verified']
    cap=cv2.VideoCapture(str(run/'annotated.mp4'))
    count=0
    while cap.read()[0]: count+=1
    assert count==12 and cap.get(cv2.CAP_PROP_FPS)==10
    cap.release()
    assert len((run/'frame_timings.csv').read_text().splitlines())==13
    with pytest.raises(FileExistsError): process_video(video_config,FakeDetector)


def test_time_window_no_render(video_config):
    video_config.update(start_seconds=.2,end_seconds=.7,render=False)
    run,s=process_video(video_config,FakeDetector)
    assert s['processed_frame_count']==5
    assert not (run/'annotated.mp4').exists()


def test_max_frames(video_config):
    video_config['max_frames']=3
    _,s=process_video(video_config,FakeDetector)
    assert s['processed_frame_count']==3


def test_bad_detection_preserves_failure(video_config):
    class Bad(FakeDetector):
        def __call__(self,frame): return np.array([[float('nan')]*6])
    with pytest.raises(ValueError,match='Malformed detector'):
        process_video(video_config,Bad)
    failure=Path(video_config['output_root'])/'test/FAILURE.json'
    assert json.loads(failure.read_text())['processed_frames']==0


def test_invalid_video(video_config):
    Path(video_config['input_video']).write_bytes(b'not a video')
    with pytest.raises(ValueError,match='decoder'):
        process_video(video_config,FakeDetector)


@pytest.mark.parametrize('key,value',[('run_id','../escape'),('imgsz',960),('confidence',2),('max_frames',0),('checkpoint_sha256','bad'),('output_video','../a.mp4')])
def test_config_rejections(video_config,key,value):
    cfg=copy.deepcopy(video_config);cfg[key]=value
    with pytest.raises(ValueError): validate_config(cfg)


def test_checkpoint_hash_rejection(video_config,tmp_path):
    checkpoint=tmp_path/'fake.pt';checkpoint.write_bytes(b'not checkpoint')
    video_config['checkpoint']=str(checkpoint)
    with pytest.raises(ValueError,match='SHA-256'): FrozenDetector(video_config)
