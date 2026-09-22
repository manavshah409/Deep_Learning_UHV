"""Small CPU recovery tests. No private dataset, detector, or reserved split."""
import copy
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.experiments.stage_b import draws, lr
from src.training.epoch_resume import EpochRun, RunLock, FORMAT_VERSION, atomic_json, canonical_hash, digest, capture_rng, restore_rng
from src.training.epoch_timing import FIELDS


def config():
    return dict(device='cpu', seed=42, sampling='unweighted', architecture='synthetic Linear(2,1)', output_classes=1,
        optimizer=dict(name='SGD',momentum=.9,weight_decay=.0005), schedule=dict(definition='frozen Stage B LR'),
        lr_by_epoch=[lr(e) for e in range(1,4)], train_manifest_hash='train', calibration_manifest_hash='calibration',
        class_mapping_hash='mapping', initializer=dict(sha256='synthetic'), resize_policy='none', format=FORMAT_VERSION)


def model_and_opt():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    model=torch.nn.Linear(2,1)
    return model,torch.optim.SGD(model.parameters(),lr=.001,momentum=.9,weight_decay=.0005)


def timing(state,epoch):
    row={k:0 for k in FIELDS}
    row.update(epoch=epoch,start_utc=f'epoch-{epoch}-start',end_utc=f'epoch-{epoch}-end',
        total_epoch_seconds=float(epoch),cumulative_seconds=state.cumulative+epoch,learning_rate=lr(epoch),
        checkpoint_saved=f'epoch_{epoch:03d}.pth',device='cpu',batch_size=1,resize_policy='none')
    return row


def updates(model,opt,epoch,stop=False):
    for g in opt.param_groups:g['lr']=lr(epoch)
    for i in draws('unweighted',[1]*4,epoch):
        opt.zero_grad()
        x=torch.randn(1,2)+random.random()+float(np.random.rand())+i/10
        loss=(model(x)-.3).square().mean();loss.backward();opt.step()
        if stop:raise RuntimeError('Simulated interruption inside epoch')


def one_epoch(state,model,opt,epoch):
    updates(model,opt,epoch)
    return state.commit(model,opt,timing(state,epoch),[.1,.3,.2][epoch-1])


def prepare(path,epochs=1):
    model,opt=model_and_opt()
    with EpochRun(path,config()) as state:
        state.initialize(model,opt)
        for epoch in range(1,epochs+1):one_epoch(state,model,opt,epoch)
    return model,opt


def rows(path):
    with (path/'epoch_timing.csv').open() as stream:return list(csv.DictReader(stream))


def rewrite_csv(path,entries):
    with (path/'epoch_timing.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader();writer.writerows(entries)


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


def test_checkpoint_roundtrip_optimizer_epoch_schedule_time_best(tmp_path):
    path=tmp_path/'run';old_model,old_opt=prepare(path,2)
    model,opt=model_and_opt()
    with EpochRun(path,config(),resume=True) as state:
        p=state.restore(model,opt)
        assert equal_tree(model.state_dict(),old_model.state_dict())
        assert equal_tree(opt.state_dict(),old_opt.state_dict())
        assert any('momentum_buffer' in v for v in opt.state.values())
        assert (p['completed_epoch'],p['next_epoch'])==(2,3)
        assert (p['current_lr'],p['next_lr'])==(.001,.001)
        assert p['schedule_hash']==canonical_hash(config()['schedule'])
        assert state.cumulative==3 and state.best_epoch==2 and state.best_metric==.3
        assert digest(path/'last.pth')==digest(path/'epoch_002.pth')==digest(path/'best.pth')
        assert state.path==path


def test_rng_roundtrip():
    model_and_opt();saved=capture_rng('cpu')
    expected=(random.random(),np.random.rand(),torch.rand(5))
    random.random();np.random.rand();torch.rand(7)
    restore_rng(saved)
    assert equal_tree(expected,(random.random(),np.random.rand(),torch.rand(5)))
    assert saved['mps']['saved'] is False and 'get_api' in saved['mps']


def test_next_epoch_sample_order(tmp_path):
    path=tmp_path/'run';prepare(path,2)
    m,o=model_and_opt()
    with EpochRun(path,config(),resume=True) as state:
        p=state.restore(m,o)
        for method in ['unweighted','weighted']:
            assert draws(method,[1,2,3,1]*8,p['next_epoch'],p['sampler']['seed'])==draws(method,[1,2,3,1]*8,3,42)


def test_completed_run_refused(tmp_path):
    path=tmp_path/'run';prepare(path,3)
    m,o=model_and_opt()
    with EpochRun(path,config(),resume=True) as state:state.restore(m,o);state.complete()
    with pytest.raises(ValueError,match='Completed'):
        with EpochRun(path,config(),resume=True):pass


@pytest.mark.parametrize('field,value',[
    ('seed',43),('sampling','weighted'),('architecture','other'),('output_classes',2),
    ('optimizer',{'name':'Adam'}),('schedule',{'definition':'other'}),('resize_policy','different'),
    ('train_manifest_hash','changed'),('calibration_manifest_hash','changed'),('class_mapping_hash','changed'),
    ('initializer',{'sha256':'changed'}),('format',999),('lr_by_epoch',[.1,.1,.1])])
def test_provenance_mismatch_refused(tmp_path,field,value):
    path=tmp_path/'run';prepare(path)
    changed=config();changed[field]=value
    with pytest.raises(ValueError,match='configuration'):
        with EpochRun(path,changed,resume=True):pass


def test_csv_ahead_refused(tmp_path):
    path=tmp_path/'run';prepare(path)
    existing=rows(path);extra=dict(existing[-1],epoch='2');rewrite_csv(path,existing+[extra])
    m,o=model_and_opt()
    with pytest.raises(ValueError,match='ahead'):
        with EpochRun(path,config(),resume=True) as state:state.restore(m,o)


def test_checkpoint_one_ahead_recovers_row_idempotently(tmp_path):
    path=tmp_path/'run';prepare(path,2)
    existing=rows(path);rewrite_csv(path,existing[:1])
    for _ in range(2):
        m,o=model_and_opt()
        with EpochRun(path,config(),resume=True) as state:state.restore(m,o)
        assert rows(path)==existing


def test_gap_greater_than_one_refused(tmp_path):
    path=tmp_path/'run';prepare(path,3);rewrite_csv(path,rows(path)[:1])
    m,o=model_and_opt()
    with pytest.raises(ValueError,match='more than one'):
        with EpochRun(path,config(),resume=True) as state:state.restore(m,o)


def test_duplicate_csv_refused(tmp_path):
    path=tmp_path/'run';prepare(path);r=rows(path);rewrite_csv(path,r+r)
    m,o=model_and_opt()
    with pytest.raises(ValueError,match='Duplicate'):
        with EpochRun(path,config(),resume=True) as state:state.restore(m,o)


def test_mismatched_timing_refused(tmp_path):
    path=tmp_path/'run';prepare(path);r=rows(path);r[0]['total_training_loss']='999';rewrite_csv(path,r)
    m,o=model_and_opt()
    with pytest.raises(ValueError,match='Timing data mismatch'):
        with EpochRun(path,config(),resume=True) as state:state.restore(m,o)


def test_corrupt_checkpoint_refused_without_fallback(tmp_path):
    path=tmp_path/'run';prepare(path,2)
    with (path/'epoch_002.pth').open('r+b') as f:f.write(b'corrupt')
    m,o=model_and_opt()
    with pytest.raises(ValueError,match='hash'):
        with EpochRun(path,config(),resume=True) as state:state.restore(m,o)


def test_nonfinite_model_not_published(tmp_path):
    m,o=model_and_opt();path=tmp_path/'run'
    with pytest.raises(ValueError,match='Non-finite'):
        with EpochRun(path,config()) as state:
            state.initialize(m,o)
            with torch.no_grad():next(m.parameters()).fill_(float('nan'))
            state.commit(m,o,timing(state,1),.1)
    assert not (path/'epoch_001.pth').exists()
    assert not (path/'epoch_timing.csv').exists()
    assert (path/'FAILURE.json').exists()


def test_concurrent_process_refused(tmp_path):
    path=tmp_path/'run';path.mkdir()
    with RunLock(path,'parent'):
        code="from src.training.epoch_resume import RunLock; import sys; RunLock(sys.argv[1],'child').__enter__()"
        p=subprocess.run([sys.executable,'-c',code,str(path)],capture_output=True,text=True)
        assert p.returncode!=0 and 'actively locked' in p.stderr


def test_stale_pid_and_reused_live_pid_not_mistaken_for_lock(tmp_path):
    import os
    path=tmp_path/'run';path.mkdir()
    for pid in [999999999,os.getpid()]:
        (path/'.run.lock').write_text(json.dumps(dict(pid=pid,session_id='stale')))
        with RunLock(path,'new'):
            assert json.loads((path/'.run.lock').read_text())['session_id']=='new'


def test_new_existing_and_resume_missing_refused(tmp_path):
    path=tmp_path/'run';prepare(path)
    with pytest.raises(FileExistsError):
        with EpochRun(path,config()):pass
    with pytest.raises(FileNotFoundError):
        with EpochRun(tmp_path/'missing',config(),resume=True):pass


def test_interruption_resumption_matches_uninterrupted(tmp_path):
    reference=tmp_path/'reference';ref_model,ref_opt=prepare(reference,3)
    path=tmp_path/'resumed';model,opt=model_and_opt()
    with pytest.raises(RuntimeError,match='Simulated'):
        with EpochRun(path,config()) as state:
            state.initialize(model,opt)
            one_epoch(state,model,opt,1);one_epoch(state,model,opt,2)
            atomic_json(path/'losses_epoch_003.json',dict(partial=True))
            updates(model,opt,3,stop=True)
    assert [r['epoch'] for r in rows(path)]==['1','2']
    assert not (path/'epoch_003.pth').exists() and (path/'FAILURE.json').exists()
    model,opt=model_and_opt()
    with EpochRun(path,config(),resume=True) as state:
        state.restore(model,opt)
        assert state.epoch==2 and state.cumulative==3 and state.best_epoch==2
        assert not (path/'losses_epoch_003.json').exists()
        assert list((path/'partial_attempts').glob('*/losses_epoch_003.json'))
        one_epoch(state,model,opt,3)
        assert state.cumulative==6 and state.best_epoch==2 and state.best_metric==.3
        assert opt.param_groups[0]['lr']==.001
        state.complete()
    assert [r['epoch'] for r in rows(path)]==['1','2','3']
    assert equal_tree(model.state_dict(),ref_model.state_dict())
    assert equal_tree(opt.state_dict(),ref_opt.state_dict())
    assert {p.name for p in tmp_path.iterdir()}=={'reference','resumed'}
    assert len(list((path/'sessions').glob('*.json')))==2
    assert len(list((path/'failures').glob('*.json')))==1


def test_failure_after_checkpoint_before_csv_recovers(tmp_path,monkeypatch):
    path=tmp_path/'run';model,opt=model_and_opt()
    original=EpochRun._append_row
    with pytest.raises(OSError,match='power loss'):
        with EpochRun(path,config()) as state:
            state.initialize(model,opt)
            monkeypatch.setattr(EpochRun,'_append_row',lambda *args: (_ for _ in ()).throw(OSError('power loss')))
            one_epoch(state,model,opt,1)
    assert not (path/'epoch_timing.csv').exists()
    monkeypatch.setattr(EpochRun,'_append_row',original)
    model,opt=model_and_opt()
    with EpochRun(path,config(),resume=True) as state:
        state.restore(model,opt);assert state.epoch==1
    assert [r['epoch'] for r in rows(path)]==['1']


def test_unsealed_partial_checkpoint_is_archived(tmp_path):
    path=tmp_path/'run';prepare(path)
    (path/'epoch_002.pth').write_bytes(b'incomplete, not published')
    m,o=model_and_opt()
    with EpochRun(path,config(),resume=True) as state:
        state.restore(m,o);assert state.epoch==1
        one_epoch(state,m,o,2)
    assert list((path/'partial_attempts').glob('*/epoch_002.pth'))


def test_scheduler_object_state_restored(tmp_path):
    path=tmp_path/'run';m,o=model_and_opt();sch=torch.optim.lr_scheduler.StepLR(o,step_size=1)
    with EpochRun(path,config()) as state:state.initialize(m,o,sch)
    m2,o2=model_and_opt();sch2=torch.optim.lr_scheduler.StepLR(o2,step_size=1)
    with EpochRun(path,config(),resume=True) as state:
        p=state.restore(m2,o2,sch2)
        assert sch2.state_dict()==sch.state_dict()==p['scheduler']


def test_cli_modes_mutually_exclusive():
    p=subprocess.run([sys.executable,'-m','src.experiments.stage_b','--run-id','x','--resume-run-id','x'],capture_output=True,text=True)
    assert p.returncode==2 and 'not allowed with argument' in p.stderr


def test_decay_boundary_next_lr_restored(tmp_path):
    cfg=config();cfg['lr_by_epoch']=[lr(e) for e in range(1,21)]
    path=tmp_path/'run';m,o=model_and_opt()
    with EpochRun(path,cfg) as state:
        state.initialize(m,o)
        for epoch in range(1,13):
            for g in o.param_groups:g['lr']=lr(epoch)
            state.commit(m,o,timing(state,epoch),epoch/100)
    m,o=model_and_opt()
    with EpochRun(path,cfg,resume=True) as state:
        p=state.restore(m,o)
        assert p['completed_epoch']==12 and p['next_epoch']==13
        assert p['current_lr']==.001 and p['next_lr']==.0001


def test_nonfinite_optimizer_not_published(tmp_path):
    path=tmp_path/'run';m,o=model_and_opt()
    with pytest.raises(ValueError,match='Non-finite'):
        with EpochRun(path,config()) as state:
            state.initialize(m,o);updates(m,o,1)
            next(iter(o.state.values()))['momentum_buffer'].fill_(float('inf'))
            state.commit(m,o,timing(state,1),.1)
    assert not (path/'epoch_001.pth').exists()


def test_unknown_checkpoint_format_refused(tmp_path):
    path=tmp_path/'run';prepare(path)
    cp=path/'epoch_001.pth';p=torch.load(cp,weights_only=True);p['format_version']=999;torch.save(p,cp)
    receipt=json.loads(cp.with_suffix('.sha256.json').read_text());receipt.update(sha256=digest(cp),bytes=cp.stat().st_size)
    atomic_json(cp.with_suffix('.sha256.json'),receipt)
    m,o=model_and_opt()
    with pytest.raises(ValueError,match='Unsupported checkpoint format'):
        with EpochRun(path,config(),resume=True) as state:state.restore(m,o)


def test_partial_csv_tail_refused(tmp_path):
    path=tmp_path/'run';prepare(path)
    with (path/'epoch_timing.csv').open('a') as f:f.write('2,partial')
    m,o=model_and_opt()
    with pytest.raises(ValueError,match='Partial timing'):
        with EpochRun(path,config(),resume=True) as state:state.restore(m,o)


def test_first_epoch_interruption_restores_initial_state(tmp_path):
    path=tmp_path/'run';m,o=model_and_opt();initial=copy.deepcopy(m.state_dict())
    with pytest.raises(RuntimeError):
        with EpochRun(path,config()) as state:
            state.initialize(m,o);updates(m,o,1,stop=True)
    m,o=model_and_opt()
    with EpochRun(path,config(),resume=True) as state:
        state.restore(m,o)
        assert state.epoch==0 and state.cumulative==0 and equal_tree(m.state_dict(),initial)
        one_epoch(state,m,o,1)


def test_fsync_occurs_before_checkpoint_rename(tmp_path,monkeypatch):
    import src.training.epoch_resume as module
    calls=[];real_sync=module.os.fsync;real_replace=module.os.replace
    def sync(fd):calls.append('fsync');return real_sync(fd)
    def replace(a,b):
        if str(b).endswith('.pth'):
            assert 'fsync' in calls
            calls.append('checkpoint-rename')
        return real_replace(a,b)
    monkeypatch.setattr(module.os,'fsync',sync);monkeypatch.setattr(module.os,'replace',replace)
    prepare(tmp_path/'run')
    assert 'checkpoint-rename' in calls


def test_foreign_run_checkpoint_refused(tmp_path):
    import shutil
    a=tmp_path/'a';b=tmp_path/'b';prepare(a);prepare(b)
    for name in ['epoch_001.pth','epoch_001.sha256.json']:shutil.copyfile(a/name,b/name)
    m,o=model_and_opt()
    with pytest.raises(ValueError,match='different scientific run'):
        with EpochRun(b,config(),resume=True) as state:state.restore(m,o)


def test_plateau_best_then_improvement_alias_publication(tmp_path):
    cfg=config();cfg['lr_by_epoch']=[lr(e) for e in range(1,7)]
    m,o=model_and_opt();path=tmp_path/'run'
    with EpochRun(path,cfg) as state:
        state.initialize(m,o)
        for e,metric in enumerate([.1,.2,.3,.4,.35,.5],1):
            for group in o.param_groups:group['lr']=lr(e)
            state.commit(m,o,timing(state,e),metric)
        assert state.best_epoch==6
        for _ in range(4):state._alias(path/'epoch_006.pth','best.pth')
        assert not list(path.glob('best.pth*.tmp'))
        assert digest(path/'best.pth')==digest(path/'epoch_006.pth')


def test_reviewed_code_repair_keeps_original_provenance(tmp_path,monkeypatch):
    import src.training.epoch_resume as module
    cfg=config();cfg['source_sha256']={'engine':'old'};path=tmp_path/'run'
    m,o=model_and_opt()
    with EpochRun(path,cfg) as state:state.initialize(m,o);one_epoch(state,m,o,1)
    original_config=(path/'config.json').read_bytes();original_cp=(path/'epoch_001.pth').read_bytes()
    changed=copy.deepcopy(cfg);changed['source_sha256']={'engine':'approved-fix'}
    policy=tmp_path/'policy.json';monkeypatch.setattr(module,'REPAIR_POLICY',policy)
    atomic_json(policy,{'repairs':[dict(id='test-fix',run_id='run',original_config_hash=canonical_hash(cfg),original_source_sha256=cfg['source_sha256'],approved_execution_source_sha256=changed['source_sha256'])]})
    m,o=model_and_opt()
    with EpochRun(path,changed,resume=True) as state:
        state.restore(m,o);one_epoch(state,m,o,2)
        assert state.code_repair=='test-fix' and state.configuration==cfg
    assert (path/'config.json').read_bytes()==original_config and (path/'epoch_001.pth').read_bytes()==original_cp
    saved=torch.load(path/'epoch_002.pth',weights_only=True)
    assert saved['configuration']==cfg and saved['execution_source_sha256']==changed['source_sha256']
    for field,value in [('seed',99),('source_sha256',{'engine':'unapproved'})]:
        bad=copy.deepcopy(changed);bad[field]=value
        with pytest.raises(ValueError,match='configuration'):
            with EpochRun(path,bad,resume=True):pass
