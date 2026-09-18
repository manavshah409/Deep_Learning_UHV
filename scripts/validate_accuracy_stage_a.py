"""Verify Stage A protocol/preflight and unchanged historical evidence."""
from pathlib import Path
import argparse
import csv
import json
import sys
import math
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.common import ROOT,sha256
from src.experiments.accuracy_data import assert_disjoint
from src.training.epoch_timing import FIELDS


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='E3_stageA_preflight_seed42_v1');a=p.parse_args()
    root=ROOT/'reports/accuracy_stage_a';protocol=root/'protocol_v2'
    d=json.loads((protocol/'protocol.json').read_text())
    splits={}
    for name,record in d['splits'].items():
        path=protocol/(name+'.json');assert sha256(path)==record['file_sha256'];splits[name]=json.loads(path.read_text());assert len(splits[name])==record['images']
    assert_disjoint(splits['calibration_500'],splits['final_evaluation_1500'])
    assert_disjoint(splits['preflight_calibration_64'],splits['final_evaluation_1500'])
    assert {r['image_id'] for r in splits['preflight_calibration_64']}<={r['image_id'] for r in splits['calibration_500']}
    run=ROOT/'runs'/a.run_id
    marker=json.loads((run/'COMPLETE.json').read_text());assert all(sha256(run/f)==h for f,h in marker['files_sha256'].items())
    with (run/'epoch_timing.csv').open() as f:timings=list(csv.DictReader(f))
    assert len(timings)==1 and set(timings[0])==set(FIELDS) and timings[0]['epoch']=='1'
    summary=json.loads((run/'summary.json').read_text());assert summary['status']=='complete' and summary['completed_epochs']==1
    assert summary['epochs'][0]['validation']['images']==64
    for key in ['training_seconds','validation_seconds','total_epoch_seconds','total_training_loss']:
        assert math.isfinite(float(timings[0][key])) and float(timings[0][key])>=0
    preservation=json.loads((root/'preservation.json').read_text())
    assert all(sha256(ROOT/f)==h for f,h in preservation['files_sha256'].items())
    result=dict(status='passed',completed_epochs=1,calibration_images=500,reserved_evaluation_images=1500,preflight_train=128,preflight_validation=64,historical_artifacts_unchanged=True,final1500_model_evaluation_started=False,full_training_started=False,model_fusion_started=False)
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
