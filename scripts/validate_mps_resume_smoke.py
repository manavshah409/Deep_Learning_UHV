"""Two tiny synthetic MPS epochs; engineering evidence, never detector training."""
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.training.epoch_resume import EpochRun, atomic_json, active_clock
from src.training.epoch_timing import FIELDS


def make():
    torch.manual_seed(42)
    model=torch.nn.Linear(2,1).to('mps')
    return model,torch.optim.SGD(model.parameters(),lr=.001,momentum=.9)


def epoch(run,model,opt,e):
    for g in opt.param_groups:g['lr']=[.0005,.001][e-1]
    start=active_clock()
    opt.zero_grad();loss=model(torch.ones(1,2,device='mps')).square().mean()
    loss.backward();opt.step();torch.mps.synchronize()
    seconds=active_clock()-start
    row={k:0 for k in FIELDS}
    row.update(epoch=e,start_utc='synthetic',end_utc='synthetic',total_epoch_seconds=seconds,
        cumulative_seconds=run.cumulative+seconds,learning_rate=opt.param_groups[0]['lr'],
        total_training_loss=float(loss.detach().cpu()),checkpoint_saved=f'epoch_{e:03d}.pth',device='mps',batch_size=1,resize_policy='none')
    run.commit(model,opt,row,e/10)


def main():
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    path=ROOT/'runs/E3_stageC0_mps_resume_v1'
    cfg=dict(device='mps',seed=42,sampling='synthetic fixed batch',architecture='Linear(2,1)',output_classes=1,
        optimizer='SGD momentum .9',schedule=dict(definition='two-epoch engineering smoke'),lr_by_epoch=[.0005,.001],
        torch=str(torch.__version__),dataset='synthetic tensor only')
    m,o=make()
    try:
        with EpochRun(path,cfg) as run:
            run.initialize(m,o);epoch(run,m,o,1)
            expected_rng=torch.rand(3,device='mps').cpu()
            raise RuntimeError('Simulated engineering stop')
    except RuntimeError as exc:
        if str(exc)!='Simulated engineering stop':raise
    m,o=make()
    with EpochRun(path,cfg,resume=True) as run:
        restored=run.restore(m,o)
        rng_saved=restored['rng']['mps']['saved']
        rng_match=torch.equal(expected_rng,torch.rand(3,device='mps').cpu()) if rng_saved else None
        if rng_saved and not rng_match:raise ValueError('MPS RNG restoration mismatch')
        momentum_devices=sorted({str(v['momentum_buffer'].device) for v in o.state.values()})
        assert momentum_devices and all(d.startswith('mps') for d in momentum_devices)
        epoch(run,m,o,2);run.complete()
        result=dict(engineering_only=True,device='mps',model='Linear(2,1)',completed_epochs=run.epoch,
            resumed_from_epoch=restored['completed_epoch'],mps_rng_saved=rng_saved,mps_rng_roundtrip_equal=rng_match,
            optimizer_momentum_devices=momentum_devices,best_epoch=run.best_epoch,
            full_E3_started=False,reserved_split_accessed=False)
    atomic_json(ROOT/'reports/accuracy_stage_c0/mps_smoke.json',result)
    print(json.dumps(result))
if __name__=='__main__':main()
