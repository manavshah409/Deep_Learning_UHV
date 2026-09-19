"""Stage B calibration-only COCO AP and per-class metrics."""
import numpy as np
import torch
from torchvision.ops import box_iou
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(model,loader,device):
    model.eval();gt=dict(info={},images=[],annotations=[],categories=[dict(id=i,name=str(i)) for i in range(1,15)]);pred=[]
    tp=np.zeros(14);fp=np.zeros(14);fn=np.zeros(14);support=np.zeros(14)
    with torch.no_grad():
        for images,targets in loader:
            outputs=model([im.to(device) for im in images])
            for im,target,out in zip(images,targets,outputs):
                out={k:v.detach().cpu() for k,v in out.items()};iid=int(target['image_id'])
                if not all(torch.isfinite(v).all() for v in out.values()):raise ValueError('Nonfinite model output')
                gt['images'].append(dict(id=iid,height=im.shape[1],width=im.shape[2]))
                for box,c in zip(target['boxes'],target['labels']):
                    x1,y1,x2,y2=box.tolist();gt['annotations'].append(dict(id=len(gt['annotations'])+1,image_id=iid,category_id=int(c),bbox=[x1,y1,x2-x1,y2-y1],area=(x2-x1)*(y2-y1),iscrowd=0))
                for box,c,score in zip(out['boxes'],out['labels'],out['scores']):
                    x1,y1,x2,y2=box.tolist();pred.append(dict(image_id=iid,category_id=int(c),bbox=[x1,y1,x2-x1,y2-y1],score=float(score)))
                for c in range(1,15):
                    gb=target['boxes'][target['labels']==c];mask=(out['labels']==c)&(out['scores']>=.25);pb=out['boxes'][mask];scores=out['scores'][mask]
                    used=set();matches=0
                    for b in pb[torch.argsort(scores,descending=True)]:
                        overlaps=box_iou(b[None],gb)[0]
                        eligible=[j for j in range(len(gb)) if j not in used and overlaps[j]>=.5]
                        if eligible:used.add(max(eligible,key=lambda j:float(overlaps[j])));matches+=1
                    support[c-1]+=len(gb);tp[c-1]+=matches;fp[c-1]+=len(pb)-matches;fn[c-1]+=len(gb)-matches
    coco=COCO();coco.dataset=gt;coco.createIndex()
    if pred:dt=coco.loadRes(pred)
    else:
        dt=COCO();dt.dataset={**gt,'annotations':[]};dt.createIndex()
    ev=COCOeval(coco,dt,'bbox');ev.params.maxDets=[1,10,300];ev.evaluate();ev.accumulate()
    p=ev.eval['precision'][:,:,:,0,-1]
    ap=float(p[p>-1].mean()) if (p>-1).any() else 0.
    p50=p[0];ap50=float(p50[p50>-1].mean()) if (p50>-1).any() else 0.
    valid=support>0
    precision=np.divide(tp,tp+fp,out=np.zeros(14),where=tp+fp>0)
    recall=np.divide(tp,tp+fn,out=np.zeros(14),where=tp+fn>0)
    per_class = [dict(class_id=i, support=int(support[i]), precision=float(precision[i]), recall=float(recall[i]), ap50=float(p50[:,i][p50[:,i]>-1].mean()) if (p50[:,i]>-1).any() else None, ap50_95=float(p[:,:,i][p[:,:,i]>-1].mean()) if (p[:,:,i]>-1).any() else None) for i in range(14)]
    return dict(per_class=per_class, precision=float(precision[valid].mean()) if valid.any() else 0.,recall=float(recall[valid].mean()) if valid.any() else 0.,map50=ap50,map50_95=ap,images=len(gt['images']),objects=len(gt['annotations']),prediction_count=len(pred),class_support=support.astype(int).tolist(),method='COCO bbox AP maxDet300; P/R macro over represented GT classes, conf.25 IoU.5 greedy; Stage B configuration-selection only; AP is macro over represented classes, not final results')
