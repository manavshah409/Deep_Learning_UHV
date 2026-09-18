"""Official ResNet-50-FPN detector; explicit background and frozen foreground map."""
from pathlib import Path
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn,FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.data.common import ROOT,sha256


def build_model(pretrained=True,min_size=480,max_size=640):
    torch.hub.set_dir(str(ROOT/'models/torchvision'))
    model=fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None,weights_backbone=None,min_size=min_size,max_size=max_size,box_score_thresh=.001,box_nms_thresh=.5,box_detections_per_img=300)
    model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,15)
    return model


def save_checkpoint(path,model,optimizer,epoch,metadata):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    temp=path.with_suffix('.tmp')
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),epoch=epoch,metadata=metadata),temp)
    temp.replace(path)
    return sha256(path)


def load_checkpoint(path,expected_sha,model):
    if sha256(path)!=expected_sha:raise ValueError('Checkpoint SHA-256 mismatch')
    data=torch.load(path,map_location='cpu',weights_only=True)
    model.load_state_dict(data['model'])
    return data
