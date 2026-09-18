"""Stage A preflight runner. Full training requires a separate explicit CLI switch."""
import argparse
import csv
import json
import random
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import torchvision
import yaml
from torch.utils.data import DataLoader
from src.data.common import ROOT,sha256
from src.experiments.accuracy_data import VehicleDataset,collate,CappedWeightedSampler,rows_hash
from src.experiments.faster_rcnn import build_model,save_checkpoint,load_checkpoint
from src.experiments.frcnn_metrics import evaluate
from src.training.epoch_timing import EpochLogger


def utc():return datetime.now(timezone.utc).isoformat()

def save(path,value):
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def sync(device):
    if device=='mps':torch.mps.synchronize()

def seeded(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def optimizer(model,cfg):return torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])

def step(model,opt,images,targets,device):
    model.train();opt.zero_grad(set_to_none=True)
    losses=model([x.to(device) for x in images],[{k:v.to(device) for k,v in t.items()} for t in targets])
    loss=sum(losses.values())
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite gradient')
    opt.step();sync(device)
    return {k:float(v.detach().cpu()) for k,v in losses.items()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/accuracy/E3_fasterrcnn.yaml');parser.add_argument('--run-id',required=True)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--preflight',action='store_true');mode.add_argument('--allow-full-training',action='store_true')
    a=parser.parse_args()
    if not a.run_id.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run ID')
    cfg=yaml.safe_load(Path(a.config).read_text());run=ROOT/'runs'/a.run_id;run.mkdir(exist_ok=False)
    completed=0
    try:
        protocol_root=ROOT/cfg['protocol'];protocol=json.loads((protocol_root/'protocol.json').read_text());root=ROOT/protocol['dataset_version']
        for name,h in protocol['parent_manifest_sha256'].items():
            if sha256(root/name)!=h:raise ValueError('Parent manifest changed')
        if sha256(ROOT/'configs/class_mapping.yaml')!=protocol['class_mapping_sha256']:raise ValueError('Class map changed')
        initializer=json.loads((ROOT/'reports/accuracy_stage_a/initializer.json').read_text())
        if sha256(ROOT/initializer['path'])!=initializer['sha256']:raise ValueError('Initializer hash mismatch')
        train_file=protocol_root/'preflight_train_128.json' if a.preflight else root/'train_manifest.json'
        val_file=protocol_root/('preflight_calibration_64.json' if a.preflight else 'calibration_500.json')
        for path in [train_file,val_file]:
            if path.parent==protocol_root and sha256(path)!=protocol['splits'][path.stem]['file_sha256']:raise ValueError('Split hash mismatch')
        train_rows=json.loads(train_file.read_text());val_rows=json.loads(val_file.read_text())
        cfg.update(epochs=1 if a.preflight else cfg['epochs'],preflight=a.preflight,train_images=len(train_rows),validation_images=len(val_rows),train_manifest_sha256=sha256(train_file),validation_manifest_sha256=sha256(val_file),canonical_train_sha256=rows_hash(train_rows),canonical_validation_sha256=rows_hash(val_rows),initializer_sha256=initializer['sha256'],class_mapping_sha256=protocol['class_mapping_sha256'],torch=str(torch.__version__),torchvision=str(torchvision.__version__))
        save(run/'config.json',cfg)
        dataset=VehicleDataset(root,train_rows);validation=VehicleDataset(root,val_rows)
        with (protocol_root/'image_sampling_weights.csv').open() as f:weight_map={int(r['image_id']):float(r['weight']) for r in csv.DictReader(f)}
        sampler=CappedWeightedSampler([weight_map[r['image_id']] for r in train_rows],cfg['seed'],cfg['maximum_image_repeats_per_epoch'])
        loader=DataLoader(dataset,batch_size=cfg['batch_size'],sampler=sampler,num_workers=0,collate_fn=collate)
        val_loader=DataLoader(validation,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate)
        device='mps' if cfg['device']=='mps' and torch.backends.mps.is_available() else 'cpu'
        if device!=cfg['device'] and not cfg['cpu_fallback']:raise RuntimeError('MPS unavailable')
        memory=dict(requested_device=cfg['device'],selected_device=device,batch_size=cfg['batch_size'],scope='one real batch forward/backward/optimizer; not full memory stress test')
        seeded(cfg['seed']);model=build_model(min_size=cfg['min_size'],max_size=cfg['max_size']).to(device);opt=optimizer(model,cfg)
        images,targets=next(iter(loader));check_start=time.perf_counter()
        try:memory['losses']=step(model,opt,images,targets,device)
        except (RuntimeError,NotImplementedError) as error:
            save(run/'mps_memory_check_failure.json',dict(error=str(error),traceback=traceback.format_exc()))
            if device!='mps' or not cfg['cpu_fallback']:raise
            del model,opt;torch.mps.empty_cache();device='cpu';seeded(cfg['seed'])
            model=build_model(min_size=cfg['min_size'],max_size=cfg['max_size']).to(device);opt=optimizer(model,cfg)
            memory['selected_device']=device;memory['fallback_reason']=str(error);memory['losses']=step(model,opt,images,targets,device)
        memory['seconds']=time.perf_counter()-check_start
        if device=='mps':memory.update(sampled_allocated_bytes=torch.mps.current_allocated_memory(),sampled_driver_bytes=torch.mps.driver_allocated_memory())
        save(run/'memory_check.json',memory)
        # Discard all memory-check optimizer updates before the measured epoch.
        del model,opt
        if device=='mps':torch.mps.empty_cache()
        seeded(cfg['seed']);model=build_model(min_size=cfg['min_size'],max_size=cfg['max_size']).to(device);opt=optimizer(model,cfg)
        logger=EpochLogger(run/'epoch_timing.csv');run_start=time.perf_counter();epochs=[]
        for epoch in range(1,cfg['epochs']+1):
            start_utc=utc();begin=time.perf_counter();sampler.epoch=epoch-1;losses=[]
            for images,targets in loader:losses.append(step(model,opt,images,targets,device))
            train_end=time.perf_counter();metrics=evaluate(model,val_loader,device);sync(device);val_end=time.perf_counter()
            mean={k:float(np.mean([r[k] for r in losses])) for k in losses[0]}
            checkpoint=run/f'epoch_{epoch:03}.pth';digest=save_checkpoint(checkpoint,model,opt,epoch,dict(config=cfg,metrics=metrics))
            loaded=load_checkpoint(checkpoint,digest,model)
            if loaded['epoch']!=epoch or not all(torch.isfinite(t).all() for t in loaded['model'].values()):raise ValueError('Checkpoint integrity failed')
            del loaded
            end=time.perf_counter()
            row=dict(epoch=epoch,start_utc=start_utc,end_utc=utc(),training_seconds=train_end-begin,validation_seconds=val_end-train_end,total_epoch_seconds=end-begin,cumulative_seconds=end-run_start,learning_rate=opt.param_groups[0]['lr'],total_training_loss=sum(mean.values()),classification_loss=mean['loss_classifier'],box_regression_loss=mean['loss_box_reg'],rpn_objectness_loss=mean['loss_objectness'],rpn_box_loss=mean['loss_rpn_box_reg'],validation_precision=metrics['precision'],validation_recall=metrics['recall'],validation_map50=metrics['map50'],validation_map50_95=metrics['map50_95'],checkpoint_saved=checkpoint.name,device=device,batch_size=cfg['batch_size'],resize_policy=f'short_side{cfg["min_size"]}_max_side{cfg["max_size"]}')
            logger.append(row);completed=epoch;epochs.append(dict(**row,checkpoint_sha256=digest,checkpoint_bytes=checkpoint.stat().st_size,validation=metrics))
            save(run/'summary.json',dict(status='complete' if epoch==cfg['epochs'] else 'running',scope='pipeline_preflight_not_accuracy' if a.preflight else 'calibration_training',completed_epochs=epoch,epochs=epochs,memory_check=memory))
            print(f'Completed epoch {epoch}; seconds={row["total_epoch_seconds"]:.1f}',flush=True)
        save(run/'COMPLETE.json',dict(files_sha256={p.name:sha256(p) for p in run.iterdir() if p.is_file() and p.name!='COMPLETE.json'}))
    except Exception as error:
        save(run/'FAILURE.json',dict(error=str(error),traceback=traceback.format_exc(),completed_epochs=completed,utc=utc()))
        raise


if __name__=='__main__':main()
