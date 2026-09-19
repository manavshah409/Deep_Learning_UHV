"""Synthetic protocol checks; never open the reserved manifest or image pool."""
import json
from pathlib import Path
import numpy as np
import pytest
from src.experiments import stage_b as b


def test_frozen_lr_schedule():
    assert [b.lr(e) for e in range(1,4)] == [.0005,.001,.001]
    assert b.lr(12)==.001 and b.lr(13)==.0001 and b.lr(17)==.0001
    assert b.lr(18)==pytest.approx(.00001) and b.lr(20)==pytest.approx(.00001)
    with pytest.raises(ValueError): b.lr(0)


def test_unweighted_exact_deterministic_coverage():
    a=b.draws('unweighted',[1]*100,1)
    assert sorted(a)==list(range(100)) and a==b.draws('unweighted',[1]*100,1)
    assert a!=b.draws('unweighted',[1]*100,2)


def test_weighted_cap_and_exposure_accounting():
    draws=b.draws('weighted',[1,2,3,1,1,1]*20,1)
    result=b.exposure(draws,[[1]*14]*120)
    assert result['total_draws']==120 and result['maximum_repetitions']<=3
    assert sum(result['repetition_histogram'].values())==120
    assert sum(k*v for k,v in result['repetition_histogram'].items())==120
    assert result['effective_per_class_objects']==[120]*14
    assert draws==b.draws('weighted',[1,2,3,1,1,1]*20,1)


def test_selection_reproducible_and_all_classes(monkeypatch):
    rows=[dict(image_id=i,image=str(i)) for i in range(100)]
    monkeypatch.setattr(b,'counts',lambda rs:[[int(r['image_id']%14==c) for c in range(14)] for r in rs])
    a=b.select(rows,30)
    assert a==b.select(list(reversed(rows)),30)
    assert len(a)==len({r['image_id'] for r in a})==30
    assert {r['image_id']%14 for r in a}==set(range(14))


def test_unweighted_exposure():
    result=b.exposure([2,0,1],[[1,0],[0,2],[3,1]])
    assert result['coverage_percentage']==100
    assert result['maximum_repetitions']==1 and result['repetition_histogram']=={1:3}
    assert result['effective_per_class_objects']==[4,3]


def test_selection_rejects_insufficient_rows(monkeypatch):
    monkeypatch.setattr(b,'counts',lambda rows:[[1]*14]*len(rows))
    with pytest.raises(ValueError):b.select([dict(image_id=1,image='a')],2)


def test_protocol_never_opens_reserved_manifest(monkeypatch,tmp_path):
    original=Path.open
    def guarded(path,*args,**kwargs):
        assert 'final_evaluation_1500' not in str(path)
        assert path.name!='val_manifest.json'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    # Synthetic validator fixture works without the private dataset or weights.
    import hashlib
    monkeypatch.setattr(b,'ROOT',tmp_path); monkeypatch.setattr(b,'A',tmp_path); monkeypatch.setattr(b,'OUT',tmp_path)
    initializer=tmp_path/'initializer.pth';initializer.write_bytes(b'initializer')
    (tmp_path/'initializer.json').write_text(json.dumps(dict(path='initializer.pth',sha256=hashlib.sha256(b'initializer').hexdigest())))
    allowed=tmp_path/'calibration_500.json';allowed.write_text('[]')
    monkeypatch.setattr(b,'EXPECTED',{allowed:hashlib.sha256(b'[]').hexdigest()})
    (tmp_path/'protocol.json').write_text(json.dumps(dict(hashes={'calibration_500.json':hashlib.sha256(b'[]').hexdigest()})))
    b.verify_protocol()


def test_configuration_frozen_before_pilots():
    p=json.loads((b.OUT/'protocol.json').read_text())
    assert p['config']==json.loads(b.CONFIG.read_text())
    assert all(x>0 for x in p['train_objects']) and all(x>0 for x in p['calibration_objects'])
    assert p['reserved_split_accessed'] is False


def test_decision_rejects_material_overall_regression():
    from scripts.close_accuracy_stage_b import decide
    import copy
    metrics=dict(class_support=[20]*14,map50=.3,map50_95=.2,per_class=[dict(ap50_95=.1) for _ in range(14)])
    epoch=dict(total_training_loss=1.,metrics=metrics,sampling=dict(effective_per_class_objects=[10]*14))
    first=dict(epochs=[copy.deepcopy(epoch) for _ in range(3)])
    second=copy.deepcopy(first)
    for e in second['epochs']:e['sampling']['effective_per_class_objects']=[20]*14
    cfg=json.loads(b.CONFIG.read_text())
    assert decide(first,second,cfg)['selected']=='weighted'
    second['epochs'][-1]['metrics']['map50_95']=.18
    assert decide(first,second,cfg)['selected']=='unweighted'


def test_decision_does_not_cherry_pick_unsupported_class():
    from scripts.close_accuracy_stage_b import decide
    import copy
    metrics=dict(class_support=[20]*13+[2],map50=.3,map50_95=.2,per_class=[dict(ap50_95=.1) for _ in range(14)])
    epoch=dict(total_training_loss=1.,metrics=metrics,sampling=dict(effective_per_class_objects=[10]*14))
    first=dict(epochs=[copy.deepcopy(epoch) for _ in range(3)]);second=copy.deepcopy(first)
    second['epochs'][-1]['metrics']['per_class'][13]['ap50_95']=.9
    cfg=json.loads(b.CONFIG.read_text())
    assert 13 not in decide(first,second,cfg)['supported_minority_class_ids']
    assert decide(first,second,cfg)['selected']=='unweighted'


def test_per_class_coco_metrics_exclude_absent_classes():
    import torch
    from src.experiments.stage_b_metrics import evaluate
    class Perfect(torch.nn.Module):
        def forward(self,images):
            return [dict(boxes=torch.tensor([[10.,10.,30.,30.]]),labels=torch.tensor([1]),scores=torch.tensor([.9]))]
    target=dict(boxes=torch.tensor([[10.,10.,30.,30.]]),labels=torch.tensor([1]),image_id=torch.tensor(1))
    result=evaluate(Perfect(),[([torch.zeros(3,40,40)],[target])],'cpu')
    assert result['map50_95']==pytest.approx(1.) and result['map50']==pytest.approx(1.)
    assert result['precision']==1. and result['recall']==1.
    assert result['per_class'][0]['ap50_95']==pytest.approx(1.)
    assert result['per_class'][1]['ap50_95'] is None
    assert result['class_support']==[1]+[0]*13
