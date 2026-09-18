import csv
import numpy as np
import pytest
import torch
from PIL import Image
from src.data.common import sha256
from src.experiments.accuracy_data import split_rows,assert_disjoint,class_weights,image_weight,CappedWeightedSampler,convert_labels,VehicleDataset
from src.training.epoch_timing import EpochLogger,FIELDS
from src.experiments.faster_rcnn import build_model,load_checkpoint


def rows(n):return [dict(image_id=i,image=f'{i}.png',source_sha256=str(i)) for i in range(n)]


def test_deterministic_split_and_disjoint():
 a,b=split_rows(rows(20),5);assert len(a)==5 and len(b)==15
 assert (a,b)==split_rows(list(reversed(rows(20))),5)
 assert_disjoint(a,b)
 with pytest.raises(ValueError):assert_disjoint(a,a)
 bad=rows(20);bad[1]['source_sha256']=bad[0]['source_sha256']
 with pytest.raises(ValueError):split_rows(bad,5)


def test_boxes_mapping_background():
 b,c=convert_labels('0 .5 .5 .2 .4\n13 .5 .5 .1 .1',100,200)
 assert c.tolist()==[1,14] and torch.allclose(b[0],torch.tensor([40.,60.,60.,140.]))
 b,c=convert_labels('',100,200);assert b.shape==(0,4) and c.dtype==torch.int64


@pytest.mark.parametrize('text',['14 .5 .5 .1 .1','0 0 0 1 1','0 .5 .5 0 1','0 nan .5 1 1','0 .5 .5 .2'])
def test_invalid_boxes(text):
 with pytest.raises(ValueError):convert_labels(text,100,100)


def test_capped_weights_and_multiclass():
 w=class_weights([100,25,1,0]);assert w==[1,2,3,0]
 assert image_weight([0,1,1],w)==1.5 and image_weight([],w)==1
 with pytest.raises(ValueError):class_weights([-1,2])
 sampler=CappedWeightedSampler([1,3,3,1],max_repeats=2)
 a=list(sampler);assert a==list(sampler) and len(a)==4 and max(np.bincount(a))<=2


def test_dataset_structure(tmp_path):
 im=tmp_path/'a.png';Image.new('RGB',(100,100)).save(im)
 lab=tmp_path/'a.txt';lab.write_text('6 .5 .5 .2 .2\n')
 r=dict(image='a.png',label='a.txt',image_id=5,source_sha256=sha256(im),label_sha256=sha256(lab))
 image,t=VehicleDataset(tmp_path,[r])[0]
 assert image.shape==(3,100,100) and image.dtype==torch.float32
 assert t['labels'].tolist()==[7] and t['area'].tolist()==[400.]
 lab.write_text('')
 with pytest.raises(ValueError,match='hash'):VehicleDataset(tmp_path,[r])[0]


def test_timing_persistence_and_immutable(tmp_path):
 p=tmp_path/'epoch_timing.csv';logger=EpochLogger(p)
 row={k:0 for k in FIELDS};row.update(epoch=1,checkpoint_saved='epoch_001.pth')
 logger.append(row)
 with p.open() as f:read=list(csv.DictReader(f))
 assert len(read)==1 and set(read[0])==set(FIELDS)
 with pytest.raises(FileExistsError):EpochLogger(p)
 with pytest.raises(ValueError):logger.append({'epoch':2})
 assert len(p.read_text().splitlines())==2


def test_checkpoint_hash_rejection(tmp_path):
 p=tmp_path/'bad.pth';p.write_bytes(b'bad')
 with pytest.raises(ValueError,match='SHA'):load_checkpoint(p,'wrong',None)


def test_official_model_output_structure_without_download():
 model=build_model(pretrained=False,min_size=64,max_size=64).eval()
 assert model.roi_heads.box_predictor.cls_score.out_features==15
 with torch.no_grad():out=model([torch.zeros(3,64,64)])[0]
 assert set(out)=={'boxes','labels','scores'} and out['boxes'].shape[1]==4
 assert torch.isfinite(out['scores']).all()
