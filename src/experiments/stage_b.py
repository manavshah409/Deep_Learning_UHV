"""Paired Stage B protocol. Never reads the reserved comparison manifest."""
import argparse
import csv
import hashlib
import json
import random
import time
import traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.common import ROOT, sha256
from src.experiments.accuracy_data import VehicleDataset, collate, CappedWeightedSampler, assert_disjoint
from src.experiments.faster_rcnn import build_model, save_checkpoint, load_checkpoint
from src.experiments.train_frcnn import seeded, optimizer, step, sync, save, utc
from src.training.epoch_timing import EpochLogger

OUT = ROOT / 'reports/accuracy_stage_b'
DATA = ROOT / 'data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2'
A = ROOT / 'reports/accuracy_stage_a'
CONFIG = ROOT / 'configs/accuracy/E3_stage_b.json'
EXPECTED = {
    DATA / 'train_manifest.json': '8e4a72413caecc9defee68e75f498b71160b06fb33f30f6cbe2dc8b3677032cf',
    A / 'protocol_v2/calibration_500.json': 'a436b78d561516d61ae2f7eedb4a8acb4a58222d2c13df80411ae807f288c502',
    ROOT / 'configs/class_mapping.yaml': '6fd0b458fed2cb174c924d4fa9ead86db0a292dc7c55fd0f4d4b3b9b20d08ea8',
}

def lr(epoch, base=.001):
    if not 1 <= epoch <= 20: raise ValueError('Epoch outside frozen schedule')
    return base * min(1., epoch / 2) * .1 ** (int(epoch >= 13) + int(epoch >= 18))

def counts(rows):
    result = []
    for row in rows:
        p = DATA / row['label']
        if sha256(p) != row['label_sha256']: raise ValueError('Label hash mismatch')
        c = Counter(int(line.split()[0]) for line in p.read_text().splitlines() if line.strip())
        result.append([c[i] for i in range(14)])
    return result

def select(rows, size, seed=42):
    """Seeded shuffle with deterministic class-covering anchors, then fill."""
    rows = sorted(rows, key=lambda r: (r['image_id'], r['image']))
    random.Random(seed).shuffle(rows)
    labels = counts(rows)
    available = {i for c in labels for i, n in enumerate(c) if n}
    selected, covered = [], set()
    # Rarest object classes first; ordering inside each class stays seed-shuffled.
    totals = np.asarray(labels).sum(axis=0)
    for cls in sorted(available, key=lambda i: (totals[i], i)):
        if cls in covered: continue
        idx = next(i for i, c in enumerate(labels) if c[cls])
        selected.append(idx)
        covered.update(i for i,n in enumerate(labels[idx]) if n)
    if len(selected) > size: raise ValueError('Insufficient room for class coverage')
    selected += [i for i in range(len(rows)) if i not in selected][:size-len(selected)]
    if len(selected) != size: raise ValueError('Insufficient parent rows')
    return [rows[i] for i in selected]

def draws(method, weights, epoch, seed=42, cap=3):
    if method == 'weighted':
        sampler = CappedWeightedSampler(weights, seed, cap); sampler.epoch = epoch - 1
        return list(sampler)
    if method != 'unweighted': raise ValueError('Unknown sampling method')
    order = list(range(len(weights))); random.Random(seed + epoch - 1).shuffle(order)
    return order

def exposure(indices, object_counts):
    n = len(object_counts); repetitions = Counter(indices)
    hist = Counter(repetitions.get(i,0) for i in range(n))
    objects = np.asarray(object_counts, dtype=int)
    effective = objects[indices].sum(axis=0)
    image_exposure = (objects[indices] > 0).sum(axis=0)
    return dict(total_draws=len(indices), unique_images=len(repetitions), coverage_percentage=100*len(repetitions)/n,
                maximum_repetitions=max(repetitions.values(), default=0), repetition_histogram=dict(sorted(hist.items())),
                effective_per_class_objects=effective.tolist(), effective_per_class_image_exposures=image_exposure.tolist())

def verify_parents():
    for path, expected in EXPECTED.items():
        if sha256(path) != expected: raise ValueError(f'Frozen hash changed: {path.name}')
    init = json.loads((A / 'initializer.json').read_text())
    if sha256(ROOT / init['path']) != init['sha256']: raise ValueError('Initializer changed')
    return init

def prepare():
    verify_parents()
    if (OUT / 'protocol.json').exists(): raise FileExistsError('Stage B protocol already frozen')
    train = select(json.loads((DATA / 'train_manifest.json').read_text()), 1000)
    val = select(json.loads((A / 'protocol_v2/calibration_500.json').read_text()), 250)
    assert_disjoint(train, val)
    for name, rows in [('train_1000',train), ('calibration_250',val)]:
        save(OUT / f'{name}.json', rows)
        for row in rows:
            if sha256(DATA / row['image']) != row['source_sha256']: raise ValueError('Image hash mismatch')
    files = [OUT/'train_1000.json', OUT/'calibration_250.json', CONFIG, A/'protocol_v2/image_sampling_weights.csv']
    save(OUT/'protocol.json', dict(created_utc=utc(), seed=42, selection='Seeded shuffle; rare-first class-covering anchors; fill in shuffled order',
         hashes={str(p.relative_to(ROOT)):sha256(p) for p in files},
         parents={str(p.relative_to(ROOT)):h for p,h in EXPECTED.items()},
         train_objects=np.asarray(counts(train)).sum(axis=0).tolist(), calibration_objects=np.asarray(counts(val)).sum(axis=0).tolist(),
         reserved_split_accessed=False, config=json.loads(CONFIG.read_text()), lr_by_epoch={e:lr(e) for e in range(1,21)}))

def verify_protocol():
    init = verify_parents()
    protocol = json.loads((OUT/'protocol.json').read_text())
    for name, expected in protocol['hashes'].items():
        if sha256(ROOT/name) != expected: raise ValueError(f'Stage B input changed: {name}')
    return init

def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def run(method, run_id, full=False):
    from src.experiments.stage_b_metrics import evaluate
    if not run_id.replace('_','').replace('-','').isalnum(): raise ValueError('Invalid run ID')
    init = verify_protocol()
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS required; no silent CPU fallback')
    cfg = json.loads(CONFIG.read_text())
    train_file = DATA/'train_manifest.json' if full else OUT/'train_1000.json'
    val_file = A/'protocol_v2/calibration_500.json' if full else OUT/'calibration_250.json'
    train, val = json.loads(train_file.read_text()), json.loads(val_file.read_text())
    assert_disjoint(train, val)
    run_dir = ROOT/'runs'/run_id; run_dir.mkdir(exist_ok=False)
    completed = 0
    try:
        cfg.update(sampling=method, epochs=20 if full else 3, train_images=len(train), calibration_images=len(val),
                   train_sha256=sha256(train_file), calibration_sha256=sha256(val_file), initializer_sha256=init['sha256'],
                   torch=str(torch.__version__), config_sha256=sha256(CONFIG),
                   source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/experiments/stage_b_metrics.py',ROOT/'src/experiments/faster_rcnn.py',ROOT/'src/experiments/accuracy_data.py',ROOT/'src/experiments/train_frcnn.py']})
        save(run_dir/'config.json', cfg)
        with (A/'protocol_v2/image_sampling_weights.csv').open() as f:
            weights_by_id = {int(r['image_id']):float(r['weight']) for r in csv.DictReader(f)}
        weights = [weights_by_id[r['image_id']] for r in train]; object_counts = counts(train)
        dataset = VehicleDataset(DATA,train)
        val_loader = DataLoader(VehicleDataset(DATA,val),batch_size=1,shuffle=False,num_workers=0,collate_fn=collate)
        seeded(cfg['seed']); model = build_model(min_size=cfg['min_size'],max_size=cfg['max_size'])
        initial_hash = state_hash(model)
        save(run_dir/'initialization.json',dict(initializer=init,initial_model_state_sha256=initial_hash,seed=cfg['seed'],stage_a_checkpoint_used=False))
        model.to('mps'); opt = optimizer(model,cfg)
        logger = EpochLogger(run_dir/'epoch_timing.csv'); start = time.perf_counter(); epochs = []
        for epoch in range(1,cfg['epochs']+1):
            epoch_start = time.perf_counter(); start_utc = utc()
            indices = draws(method, weights, epoch, cfg['seed'],cfg['maximum_image_repeats_per_epoch'])
            sampling = exposure(indices,object_counts)
            sampling.update(epoch=epoch,draw_image_ids=[train[i]['image_id'] for i in indices])
            save(run_dir/f'sampling_epoch_{epoch:03d}.json',sampling)
            for group in opt.param_groups: group['lr'] = lr(epoch,cfg['learning_rate'])
            loader = DataLoader(dataset,batch_size=1,sampler=indices,num_workers=0,collate_fn=collate)
            losses=[]; sync('mps'); t=time.perf_counter()
            for images,targets in loader: losses.append(step(model,opt,images,targets,'mps'))
            sync('mps'); train_seconds=time.perf_counter()-t
            means={k:float(np.mean([x[k] for x in losses])) for k in losses[0]}
            save(run_dir/f'losses_epoch_{epoch:03d}.json',dict(mean=means,steps=losses,all_finite=True))
            t=time.perf_counter(); metrics=evaluate(model,val_loader,'mps'); sync('mps'); val_seconds=time.perf_counter()-t
            save(run_dir/f'metrics_epoch_{epoch:03d}.json',metrics)
            checkpoint=run_dir/f'epoch_{epoch:03d}.pth'
            checksum=save_checkpoint(checkpoint,model,opt,epoch,dict(config=cfg,metrics=metrics,initial_state_sha256=initial_hash))
            loaded=load_checkpoint(checkpoint,checksum,model)
            if loaded['epoch']!=epoch or any(not torch.isfinite(v).all() for v in loaded['model'].values()): raise ValueError('Checkpoint integrity failed')
            del loaded
            row=dict(epoch=epoch,start_utc=start_utc,end_utc=utc(),training_seconds=train_seconds,validation_seconds=val_seconds,
                     total_epoch_seconds=time.perf_counter()-epoch_start,cumulative_seconds=time.perf_counter()-start,
                     learning_rate=lr(epoch,cfg['learning_rate']),total_training_loss=sum(means.values()),
                     classification_loss=means['loss_classifier'],box_regression_loss=means['loss_box_reg'],
                     rpn_objectness_loss=means['loss_objectness'],rpn_box_loss=means['loss_rpn_box_reg'],
                     validation_precision=metrics['precision'],validation_recall=metrics['recall'],validation_map50=metrics['map50'],validation_map50_95=metrics['map50_95'],
                     checkpoint_saved=checkpoint.name,device='mps',batch_size=1,resize_policy='min480/max640 aspect preserving')
            logger.append(row); completed=epoch
            epochs.append(dict(**row,checkpoint_sha256=checksum,checkpoint_bytes=checkpoint.stat().st_size,metrics=metrics,sampling={k:v for k,v in sampling.items() if k!='draw_image_ids'}))
            save(run_dir/'summary.json',dict(run_id=run_id,completed_epochs=completed,initial_state_sha256=initial_hash,epochs=epochs,configuration_selection_only=not full,reserved_split_accessed=False))
            print(json.dumps(dict(epoch=epoch,train_s=round(train_seconds,2),val_s=round(val_seconds,2),loss=sum(means.values()),map50=metrics['map50'],map50_95=metrics['map50_95'])),flush=True)
        save(run_dir/'COMPLETE.json',dict(exit_status=0,completed_epochs=completed,end_utc=utc()))
    except BaseException as error:
        save(run_dir/'FAILURE.json',dict(error=str(error),traceback=traceback.format_exc(),completed_epochs=completed,utc=utc()))
        raise

def main():
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true');p.add_argument('--validate',action='store_true')
    p.add_argument('--sampling',choices=['unweighted','weighted']);p.add_argument('--run-id');p.add_argument('--allow-full-training',action='store_true')
    a=p.parse_args()
    if a.prepare: prepare()
    elif a.validate: verify_protocol(); print('Frozen Stage B input hashes passed; reserved split not accessed')
    elif a.sampling and a.run_id: run(a.sampling,a.run_id,a.allow_full_training)
    else: p.error('Choose prepare, validate, or sampling and run-id')
if __name__=='__main__': main()
