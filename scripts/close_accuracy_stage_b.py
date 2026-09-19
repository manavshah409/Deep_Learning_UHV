"""Validate and summarize completed paired pilots without accessing reserved data."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.experiments.stage_b import ROOT,OUT,DATA,A,CONFIG,verify_protocol,counts,draws,exposure
from src.experiments.accuracy_data import assert_disjoint
from src.data.common import sha256
from src.experiments.train_frcnn import save


def decide(b1,b2,configuration):
    """Predeclared thresholds; use supported minority macro, never best rare class."""
    rule=configuration['decision']; minority=rule['minority_class_ids']
    supported=[i for i in minority if b1['epochs'][-1]['metrics']['class_support'][i]>=rule['minimum_calibration_objects_for_minority_ap']]
    def ap(run):
        return float(np.mean([run['epochs'][-1]['metrics']['per_class'][i]['ap50_95'] for i in supported])) if supported else None
    def exp(run):
        return sum(sum(e['sampling']['effective_per_class_objects'][i] for i in minority) for e in run['epochs'])
    ap1,ap2=ap(b1),ap(b2); exp1,exp2=exp(b1),exp(b2)
    d50=b2['epochs'][-1]['metrics']['map50']-b1['epochs'][-1]['metrics']['map50']
    d95=b2['epochs'][-1]['metrics']['map50_95']-b1['epochs'][-1]['metrics']['map50_95']
    finite=all(np.isfinite(e['total_training_loss']) for r in [b1,b2] for e in r['epochs'])
    stable=finite and all(r['epochs'][-1]['total_training_loss']<=r['epochs'][0]['total_training_loss'] for r in [b1,b2])
    overall_ok=min(d50,d95)>=-rule['overall_regression_limit_ap']
    ap_gain=ap1 is not None and ap2>ap1
    exposure_gain=exp2>exp1
    minority_ok=ap1 is not None and ap2>=ap1
    weighted=stable and overall_ok and (ap_gain or (exposure_gain and minority_ok))
    return dict(selected='weighted' if weighted else 'unweighted',final_epoch_map50_delta=d50,final_epoch_map50_95_delta=d95,
        supported_minority_class_ids=supported,supported_minority_ap_b1=ap1,supported_minority_ap_b2=ap2,
        aggregate_minority_object_exposure_b1=exp1,aggregate_minority_object_exposure_b2=exp2,
        finite_and_nonincreasing_endpoint_loss=stable,overall_regression_within_limit=overall_ok,
        minority_ap_gain=ap_gain,minority_exposure_gain=exposure_gain,
        interpretation='Provisional configuration selection from one seed and three epochs; no statistical superiority claim. Unsupported rare classes are excluded from AP selection. Unweighted is the lower-coverage-risk default when the weighted rule does not pass.')


def main():
    verify_protocol()
    process=json.loads((OUT/'process_status.json').read_text())
    assert len(process['processes'])==2 and all(p['returncode']==0 for p in process['processes'])
    train=json.loads((OUT/'train_1000.json').read_text());val=json.loads((OUT/'calibration_250.json').read_text())
    parents=[json.loads((DATA/'train_manifest.json').read_text()),json.loads((A/'protocol_v2/calibration_500.json').read_text())]
    for selected,parent,size in zip([train,val],parents,[1000,250]):
        assert len(selected)==size and len({r['image_id'] for r in selected})==size
        lookup={r['image_id']:r for r in parent}
        for r in selected:
            assert r==lookup[r['image_id']]
            assert sha256(DATA/r['image'])==r['source_sha256']
            assert sha256(DATA/r['label'])==r['label_sha256']
    assert_disjoint(train,val)
    object_counts=counts(train)
    with (A/'protocol_v2/image_sampling_weights.csv').open() as f: wm={int(r['image_id']):float(r['weight']) for r in csv.DictReader(f)}
    weights=[wm[r['image_id']] for r in train]
    names=json.loads((A/'protocol_v2/weighting.json').read_text())['class_names']
    summaries={}; configs={}; checkpoints=[]; flat=[]; per_class=[]; coverage=[]; warnings={}; loss_diagnostics=[]
    for method in ['unweighted','weighted']:
        run_id=f'E3_stageB_{method}_3ep_seed42_v1';run=ROOT/'runs'/run_id
        summary=json.loads((run/'summary.json').read_text());config=json.loads((run/'config.json').read_text())
        assert json.loads((run/'COMPLETE.json').read_text())['completed_epochs']==3
        assert summary['completed_epochs']==3 and len(summary['epochs'])==3
        assert config['train_images']==1000 and config['calibration_images']==250
        assert not (run/'FAILURE.json').exists()
        summaries[method]=summary;configs[method]=config
        save(OUT/f'{method}_summary.json',summary)
        save(OUT/f'{method}_config.json',config)
        (OUT/f'{method}_epoch_timing.csv').write_bytes((run/'epoch_timing.csv').read_bytes())
        warnings[method]=[line.strip() for line in (ROOT/'runs'/f'{run_id}.log').read_text().splitlines() if any(x in line.lower() for x in ['warning','error','fallback','traceback'])]
        for e in summary['epochs']:
            ep=e['epoch'];expected=exposure(draws(method,weights,ep),object_counts)
            assert all(e['sampling'][k]==json.loads(json.dumps(v)) for k,v in expected.items())
            stored=json.loads((run/f'sampling_epoch_{ep:03d}.json').read_text())
            assert stored['draw_image_ids']==[train[i]['image_id'] for i in draws(method,weights,ep)]
            losses=json.loads((run/f'losses_epoch_{ep:03d}.json').read_text())
            assert len(losses['steps'])==1000 and all(np.isfinite(v) for step in losses['steps'] for v in step.values())
            totals=np.asarray([sum(step.values()) for step in losses['steps']])
            loss_diagnostics.append(dict(method=method,epoch=ep,mean=float(totals.mean()),minimum=float(totals.min()),maximum=float(totals.max()),p95=float(np.percentile(totals,95)),p99=float(np.percentile(totals,99)),finite_steps=len(totals)))
            cp=run/e['checkpoint_saved'];assert sha256(cp)==e['checkpoint_sha256'] and cp.stat().st_size==e['checkpoint_bytes']
            payload=torch.load(cp,map_location='cpu',weights_only=True)
            assert payload['epoch']==ep and payload['metadata']['initial_state_sha256']==summary['initial_state_sha256']
            assert all(torch.isfinite(v).all() for v in payload['model'].values())
            assert all(group['lr']==e['learning_rate'] for group in payload['optimizer']['param_groups'])
            assert all(torch.isfinite(v).all() for state in payload['optimizer']['state'].values() for v in state.values() if isinstance(v,torch.Tensor));del payload
            checkpoints.append(dict(method=method,epoch=ep,path=str(cp.relative_to(ROOT)),sha256=e['checkpoint_sha256'],bytes=e['checkpoint_bytes'],readable=True))
            flat.append(dict(method=method,**{k:e[k] for k in ['epoch','training_seconds','validation_seconds','total_epoch_seconds','cumulative_seconds','learning_rate','total_training_loss','classification_loss','box_regression_loss','rpn_objectness_loss','rpn_box_loss','validation_precision','validation_recall','validation_map50','validation_map50_95']}))
            coverage.append(dict(method=method,epoch=ep,**expected))
            for c in e['metrics']['per_class']:per_class.append(dict(method=method,epoch=ep,class_name=names[c['class_id']],**c))
    x,y=configs.values();assert {k:v for k,v in x.items() if k!='sampling'}=={k:v for k,v in y.items() if k!='sampling'}
    assert summaries['unweighted']['initial_state_sha256']==summaries['weighted']['initial_state_sha256']
    for name,rows in [('epoch_comparison',flat),('per_class_metrics',per_class)]:
        with (OUT/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(OUT/'sampling_comparison.json',coverage)
    save(OUT/'loss_diagnostics.json',loss_diagnostics)
    save(OUT/'checkpoint_integrity.json',checkpoints)
    cfg=json.loads(CONFIG.read_text());decision=decide(summaries['unweighted'],summaries['weighted'],cfg)
    def projected(e):
        overhead=max(0,e['total_epoch_seconds']-e['training_seconds']-e['validation_seconds'])
        return e['training_seconds']*8+e['validation_seconds']*2+overhead
    projection={m:dict(hours_20_epochs=float(np.mean([projected(e) for e in r['epochs']]))*20/3600,
        observed_epoch_scaled_range_hours=[min(projected(e) for e in r['epochs'])*20/3600,max(projected(e) for e in r['epochs'])*20/3600]) for m,r in summaries.items()}
    save(OUT/'comparison.json',dict(decision=decision,projection=projection,projection_method='20 * mean(8*1000-image training + 2*250-image calibration + checkpoint/log overhead); planning estimate, not measured full training; no early stopping assumed',warnings=warnings,initial_state_sha256=summaries['unweighted']['initial_state_sha256'],scope='Configuration selection only; full E3/E4/reserved evaluation unstarted'))
    before=json.loads((OUT/'preservation_before.json').read_text())
    changed=[p for p,h in before['hashes'].items() if sha256(ROOT/p)!=h]
    assert not changed, f'Historical artifacts changed: {changed}'
    save(OUT/'validation.json',dict(passed=True,selected_images_hash_checked=1250,all_six_checkpoints_readable=True,sampling_only_configuration_difference=True,
        identical_initial_state=True,all_six_epoch_losses_finite=True,historical_files_unchanged=len(before['hashes']),reserved_manifest_opened=False,
        checks=['frozen parent and protocol hashes','selected row membership and disjointness','selected pixel/label integrity','actual process exits','six readable hashed checkpoints','deterministic draw replay and class exposure','identical initial model and paired configuration','historical preservation']))
    print(json.dumps(dict(decision=decision,projection=projection,validation='passed'),indent=2))
if __name__=='__main__':main()
