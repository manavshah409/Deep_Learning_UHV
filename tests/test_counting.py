import json
import pytest
from src.video.tracking import Track
from src.video.counting import LineCounter,export_events

ID=dict(run_id='test',video_id='synthetic',video_sha256='test',line_id='L1',tracker='ByteTrack',checkpoint_sha256='test')

def move(counter,points,tid=1,start=0):
 t=Track(tid,start,start)
 for f,(x,y) in enumerate(points,start):
  t.observe(f,[x-2,y-4,x+2,y],.9,7);counter.update(t,f,10,ID)
 return t

@pytest.mark.parametrize('points,direction',[
 ([(5,-5),(5,-4),(5,5)],'A_to_B'), ([(5,5),(5,4),(5,-5)],'B_to_A')])
def test_direction(points,direction):
 c=LineCounter([[0,0],[10,0]],hysteresis=2);move(c,points)
 assert len(c.events)==1 and c.events[0]['direction']==direction

@pytest.mark.parametrize('points',[
 [(5,-5),(5,-4),(5,-3)],[(5,0),(6,0),(7,0)],[(5,-1),(5,1),(5,-1),(5,1)],[(20,-5),(20,-4),(20,5)]])
def test_no_crossing(points):
 c=LineCounter([[0,0],[10,0]],hysteresis=2);move(c,points);assert not c.events


def test_diagonal():
 c=LineCounter([[0,0],[10,10]],hysteresis=1)
 move(c,[(6,2),(6,3),(2,6)]);assert len(c.events)==1


def test_duplicate_and_reverse_and_total():
 points=[(5,-5),(5,-4),(5,5),(5,-5),(5,5)]
 c=LineCounter([[0,0],[10,0]],hysteresis=2);t=move(c,points)
 assert len(c.events)==2
 c.update(t,4,10,ID);assert len(c.events)==2
 c=LineCounter([[0,0],[10,0]],hysteresis=2,mode='total');move(c,points);assert len(c.events)==1


def test_disappearance_no_inferred_crossing():
 c=LineCounter([[0,0],[10,0]],minimum_age=1,hysteresis=2)
 t=move(c,[(5,-5)])
 t.observe(4,[3,1,7,5],.9,7);c.update(t,4,10,ID);assert not c.events


def test_multiple_and_frozen_classes(tmp_path):
 c=LineCounter([[0,0],[10,0]],hysteresis=2)
 t=move(c,[(5,-5),(5,-4),(5,5)])
 move(c,[(6,5),(6,4),(6,-5)],tid=2)
 for i in range(4,20):t.observe(i,[3,1,7,5],1,6)
 assert t.class_id==6 and c.events[0]['class_id']==7
 export_events(tmp_path,c.events)
 assert len(json.loads((tmp_path/'crossing_events.json').read_text()))==2
 assert not list(tmp_path.glob('*.tmp'))


def test_roi():
 c=LineCounter([[0,0],[10,0]],hysteresis=2,roi=[[0,-10],[3,-10],[3,10],[0,10]])
 move(c,[(5,-5),(5,-4),(5,5)]);assert not c.events
