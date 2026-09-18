"""Deterministic protocol, capped sampling and Torchvision detection data."""
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,Sampler
from src.data.common import sha256


def split_rows(rows, calibration_size=500, seed=42):
    if not 0<calibration_size<len(rows):raise ValueError('Invalid split size')
    for key in ['image_id','image','source_sha256']:
        if len({r[key] for r in rows})!=len(rows):raise ValueError(f'Duplicate {key}')
    ordered=sorted(rows,key=lambda r:(r['image_id'],r['image']))
    random.Random(seed).shuffle(ordered)
    return ordered[:calibration_size],ordered[calibration_size:]


def assert_disjoint(a,b):
    for key in ['image_id','image','source_sha256']:
        if {r[key] for r in a}&{r[key] for r in b}:raise ValueError(f'Overlap {key}')


def class_weights(counts,cap=3.):
    counts=np.asarray(counts,dtype=float)
    if cap<1 or not math.isfinite(cap) or np.any(counts<0) or not np.isfinite(counts).all():raise ValueError('Invalid counts/cap')
    maximum=float(max(counts,default=0))
    # Absent classes cannot be created by sampling: explicit zero weight.
    return [min(cap,math.sqrt(maximum/n)) if n else 0. for n in counts]


def image_weight(classes,weights):
    unique=sorted(set(classes))
    if any(c<0 or c>=len(weights) for c in unique):raise ValueError('Invalid class')
    return float(np.mean([weights[c] for c in unique])) if unique else 1.


class CappedWeightedSampler(Sampler):
    """Exactly N draws; max_repeats caps exposure per image in each epoch."""
    def __init__(self,weights,seed=42,max_repeats=3):
        self.weights=list(weights);self.seed=seed;self.max_repeats=max_repeats;self.epoch=0
        if not self.weights or max_repeats<1 or any(not math.isfinite(w) or w<=0 for w in weights):raise ValueError('Invalid sampler')
    def __len__(self):return len(self.weights)
    def __iter__(self):
        rng=random.Random(self.seed+self.epoch);weights=self.weights.copy();counts=[0]*len(weights)
        for _ in weights:
            i=rng.choices(range(len(weights)),weights=weights,k=1)[0]
            counts[i]+=1
            if counts[i]>=self.max_repeats:weights[i]=0
            yield i


def convert_labels(text,width,height):
    boxes=[];labels=[]
    for line in text.splitlines():
        if not line.strip():continue
        values=list(map(float,line.split()))
        if len(values)!=5 or not all(math.isfinite(x) for x in values):raise ValueError('Malformed label')
        c,x,y,w,h=values
        if c!=int(c) or not 0<=c<14 or w<=0 or h<=0:raise ValueError('Invalid label class/size')
        xyxy=[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height]
        if xyxy[0]<-1e-4 or xyxy[1]<-1e-4 or xyxy[2]>width+1e-4 or xyxy[3]>height+1e-4:raise ValueError('Box out of bounds')
        boxes.append(xyxy);labels.append(int(c)+1)
    return torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),torch.tensor(labels,dtype=torch.int64)


class VehicleDataset(Dataset):
    def __init__(self,root,rows,verify=True):self.root=Path(root);self.rows=rows;self.verify=verify
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index];image=self.root/r['image'];label=self.root/r['label']
        if self.verify and (sha256(image)!=r['source_sha256'] or sha256(label)!=r['label_sha256']):raise ValueError('Dataset content hash mismatch')
        with Image.open(image) as im:
            im=im.convert('RGB');w,h=im.size;arr=np.asarray(im).copy()
        boxes,labels=convert_labels(label.read_text(),w,h)
        target=dict(boxes=boxes,labels=labels,image_id=torch.tensor(r['image_id']),area=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]),iscrowd=torch.zeros(len(labels),dtype=torch.int64))
        return torch.from_numpy(arr).permute(2,0,1).float()/255,target


def collate(batch):return tuple(zip(*batch))


def rows_hash(rows):return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
