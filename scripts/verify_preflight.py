"""Verify frozen preflight integrity and export predictions for separate visual review."""
from concurrent.futures import ThreadPoolExecutor
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import yaml
from src.data.common import ROOT, sha256, save_json, category_mapping, annotation_paths, load_coco, paths

NAME='yolov8n_uvh26_mv_preflight_seed42_v1'
DATA=ROOT/'data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2'
REPORT=ROOT/'reports/audit/preflight_integrity_verification.json'


def main():
    result=dict(experiment_id=NAME,status='checking')
    try:
        prov=json.loads((ROOT/'reports/tables'/f'{NAME}_provenance.json').read_text())
        assert prov['status']=='completed' and prov['preflight'] and prov['epochs_completed']==1
        checks={}
        for key,p,expected in [('config',ROOT/'configs/baseline_yolov8n.yaml',prov['config_sha256']),('class_mapping',ROOT/'configs/class_mapping.yaml',prov['class_mapping_sha256']),('manifest',DATA/'manifest.json',prov['dataset_manifest_sha256']),('train_manifest',DATA/'train_manifest.json',prov['split_manifest_sha256']['train']),('val_manifest',DATA/'val_manifest.json',prov['split_manifest_sha256']['val']),('pretrained',ROOT/prov['model'],prov['pretrained_sha256']),('training_source',ROOT/'src/training/train_baseline.py',prov['source_sha256']),('requirements',ROOT/'requirements.txt',prov['requirements_sha256'])]:
            checks[key]=sha256(p)==expected
        for key in ['weights','last_weights']:
            w=prov[key];p=ROOT/w['path'];checks[key]=p.stat().st_size==w['bytes'] and sha256(p)==w['sha256']
        assert all(checks.values()),f'Checksum failures: {[k for k,v in checks.items() if not v]}'
        spec=yaml.safe_load((DATA/'dataset.yaml').read_text())
        assert Path(spec['path']).resolve()==DATA.resolve()
        assert spec['train']=='images/train' and spec['val']=='images/val'
        args=yaml.safe_load((ROOT/'runs'/NAME/'args.yaml').read_text())
        assert Path(args['data']).resolve()==(DATA/'dataset.yaml').resolve()
        frozen=yaml.safe_load((ROOT/'configs/baseline_yolov8n.yaml').read_text())
        for key,val in prov['config'].items():
            assert val == (1 if key=='epochs' else frozen[key]),f'Config differs: {key}'
        for key in ['optimizer','lr0','weight_decay','imgsz','batch','seed','epochs']:
            assert args[key]==prov['config'][key],f'Effective args differ: {key}'
        from ultralytics import YOLO
        model=YOLO(ROOT/prov['weights']['path'])
        assert model.names==spec['names'],'Checkpoint taxonomy mismatch'
        import torch
        assert all(bool(torch.isfinite(v).all()) for v in model.model.state_dict().values()),'Nonfinite checkpoint values'
        catalogue=load_coco(annotation_paths(paths()['raw'])['train'])
        assert spec['names']=={c['yolo_id']:c['name'] for c in category_mapping(catalogue['categories'])}
        rows=json.loads((DATA/'manifest.json').read_text());raw=paths()['raw']
        def check(row):
            image=DATA/row['image'];label=DATA/row['label']
            return image.resolve()==(raw/row['source']).resolve() and sha256(image)==row['source_sha256'] and sha256(label)==row['label_sha256']
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i,okay in enumerate(pool.map(check,rows),1):
                assert okay,f'Image/label/path integrity failure at manifest row {i}'
                if i%1000==0:print('Verified image and label hashes',i,flush=True)
        losses=pd.read_csv(ROOT/'runs'/NAME/'results.csv');losses.columns=losses.columns.str.strip();assert len(losses)==1
        assert np.isfinite(losses[[c for c in losses if 'loss' in c]].to_numpy()).all()
        review=json.loads((ROOT/'reports/audit/visual_review.json').read_text())
        assert review['status'] in ['passed','passed_with_documented_source_limitations'] and review['manifest_sha256']==sha256(DATA/'manifest.json')
        for s in ['train','val']:
            assert len([r for r in rows if r['split']==s])=={'train':8000,'val':2000}[s]
        result.update(status='passed',checksums=checks,images_and_labels_verified=len(rows),dataset_paths='passed',class_mapping='passed',checkpoint_tensor_values='finite',losses='finite',annotation_review=review['status'],checkpoint_sha256=prov['weights']['sha256'],manifest_sha256=sha256(DATA/'manifest.json'),preflight_epoch_metrics=losses.iloc[0].to_dict(),prediction_review='pending')
        save_json(REPORT,result)
        lookup={(r['split'],r['image_id']):r for r in rows}
        out=ROOT/'reports/predictions/preflight_seed42_v1'
        if out.exists():raise ValueError('Prediction directory already exists; preserve it')
        out.mkdir(parents=True)
        predictions=[]
        for r in review['reviewed_images']:
            row=lookup[(r['split'],r['image_id'])]
            pred=model.predict(str(DATA/row['image']),device='mps',imgsz=640,conf=.25,iou=.7,max_det=300,verbose=False)[0]
            filename=f"{r['split']}_{r['image_id']}.jpg";pred.save(filename=str(out/filename))
            predictions.append(dict(split=r['split'],image_id=r['image_id'],file=filename,ground_truth_objects=row['objects'],predictions=len(pred.boxes),detections=[dict(class_id=int(c),confidence=float(conf),xyxy=[float(v) for v in xy]) for c,conf,xy in zip(pred.boxes.cls.cpu(),pred.boxes.conf.cpu(),pred.boxes.xyxy.cpu())]))
        save_json(out/'predictions.json',predictions)
        result.update(prediction_review='awaiting_visual_inspection',prediction_images=len(predictions),total_predictions=sum(r['predictions'] for r in predictions),prediction_confidence=.25,prediction_nms_iou=.7)
        save_json(REPORT,result);print(json.dumps(result,indent=2))
    except BaseException as e:
        result.update(status='failed',error=str(e));save_json(REPORT,result);raise


if __name__=='__main__':main()
