"""Engineering-only CPU interruption demonstration using the tested tiny fixture.

No detector or dataset is loaded. Artifacts live under ignored runs; the JSON
summary contains only portable paths. Refuses an existing engineering run ID.
"""
import json
import runpy
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.training.epoch_resume import EpochRun, atomic_json, digest


def main():
    fixture=runpy.run_path(str(ROOT/'tests/test_epoch_resume.py'))
    directory=ROOT/'runs/E3_stageC0_cpu_resume_v2'
    directory.mkdir(exist_ok=False)
    reference=directory/'uninterrupted';resumed=directory/'resumed'
    config=fixture['config'];make=fixture['model_and_opt'];epoch=fixture['one_epoch'];rows=fixture['rows']
    expected_model,expected_optimizer=fixture['prepare'](reference,3)
    model,optimizer=make()
    try:
        with EpochRun(resumed,config()) as run:
            run.initialize(model,optimizer)
            epoch(run,model,optimizer,1);epoch(run,model,optimizer,2)
            fixture['updates'](model,optimizer,3,stop=True)
    except RuntimeError as error:
        assert str(error)=='Simulated interruption inside epoch'
    interrupted_epochs=[int(r['epoch']) for r in rows(resumed)]
    model,optimizer=make()
    with EpochRun(resumed,config(),resume=True) as run:
        restored=run.restore(model,optimizer)
        assert run.epoch==2
        epoch(run,model,optimizer,3);run.complete()
        result=dict(engineering_only=True,device='cpu',model='Linear(2,1)',
            interrupted_completed_epochs=interrupted_epochs,final_completed_epochs=[int(r['epoch']) for r in rows(resumed)],
            restored_completed_epoch=restored['completed_epoch'],restored_next_epoch=restored['next_epoch'],
            restored_current_lr=restored['current_lr'],restored_next_lr=restored['next_lr'],
            cumulative_completed_seconds=run.cumulative,timing_note='Synthetic fixed epoch durations 1+2+3 seconds, not a performance measurement',
            best_epoch=run.best_epoch,best_metric=run.best_metric,
            exact_model_equality=fixture['equal_tree'](model.state_dict(),expected_model.state_dict()),
            exact_optimizer_equality=fixture['equal_tree'](optimizer.state_dict(),expected_optimizer.state_dict()),
            same_scientific_run_directory=True,scientific_subdirectories=sorted(p.name for p in directory.iterdir()),
            checkpoint_sha256=digest(resumed/'last.pth'),checkpoint_bytes=(resumed/'last.pth').stat().st_size,
            rng_saved=dict(python=True,numpy=True,torch_cpu=True,mps=restored['rng']['mps']),
            full_E3_started=False,reserved_split_accessed=False)
    assert result['exact_model_equality'] and result['exact_optimizer_equality']
    assert result['final_completed_epochs']==[1,2,3] and result['best_epoch']==2
    assert (resumed/'FAILURE.json').exists() and (resumed/'COMPLETE.json').exists()
    result['failure_evidence_preserved']=True
    result['session_count']=len(list((resumed/'sessions').glob('*.json')))
    atomic_json(ROOT/'reports/accuracy_stage_c0/cpu_integration.json',result)
    print(json.dumps({k:result[k] for k in ['final_completed_epochs','exact_model_equality','exact_optimizer_equality','best_epoch','cumulative_completed_seconds']}))
if __name__=='__main__':main()
