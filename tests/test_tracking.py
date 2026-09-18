import numpy as np
import pytest
from src.video.tracking import VehicleTracker,Track


def tracker():
 return VehicleTracker(dict(track_buffer=3,track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,match_thresh=.8,fuse_score=True),3)

def box(x=0,cls=7): return np.array([[x,0,x+40,40,.9,cls]],dtype=float)


def test_stable_ids_and_misses():
 t=tracker(); a=t.update(box(),0,(100,100))[0]
 assert t.update(box(1),1,(100,100))[0].track_id==a.track_id
 assert t.update(np.empty((0,6)),2,(100,100))==[]
 assert t.update(box(2),3,(100,100))[0].track_id==a.track_id
 assert a.observations==3


def test_expiration_empty_and_repeat():
 t=tracker();a=t.update(box(),0,(100,100))[0]
 assert t.update(box(),0,(100,100))==[]
 for i in range(1,5):t.update(np.empty((0,6)),i,(100,100))
 assert a.track_id not in t.active and a in t.finished


def test_votes_and_tie():
 t=Track(1,0,0)
 for i,c in enumerate([7,6]):t.observe(i,[0,0,10,10],.5,c)
 assert t.class_id==6
 t.observe(2,[0,0,10,10],.9,7)
 assert t.class_id==7 and t.anchor==[5,10]


@pytest.mark.parametrize('bad',[np.ones((1,3)),np.array([[0,0,1,1,1,.9,20,0]]),np.array([[0,0,1,1,1,float('nan'),7,0]])])
def test_bad_output(bad):
 with pytest.raises(ValueError):tracker().accept(bad,0)


def test_simultaneous_opposite_tracks():
 t=tracker();r=t.update(np.concatenate([box(0),box(100)]),0,(200,200))
 assert len({x.track_id for x in r})==2
 r2=t.update(np.concatenate([box(1),box(99)]),1,(200,200))
 assert {x.track_id for x in r}=={x.track_id for x in r2}
