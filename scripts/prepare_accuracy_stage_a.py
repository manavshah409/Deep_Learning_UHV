"""Read-only historical verification and additive controlled protocol exports."""
import json
import random
import sys
from pathlib import Path
import numpy as np
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.common import ROOT,sha256
from src.experiments.accuracy_data import split_rows,assert_disjoint,class_weights,image_weight,CappedWeightedSampler,rows_hash


def main():
    out=ROOT/'reports/accuracy_stage_a/protocol_v2'
    out.mkdir(exist_ok=False)
    root=ROOT/'data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2'
    expected={'train_manifest.json':'8e4a72413caecc9defee68e75f498b71160b06fb33f30f6cbe2dc8b3677032cf','val_manifest.json':'fd23d2e417d70a8614b6312cb3eff1deb0ed98f0532269c22878269fdc5959d3'}
    for name,digest in expected.items():assert sha256(root/name)==digest
    assert sha256(ROOT/'configs/class_mapping.yaml')=='6fd0b458fed2cb174c924d4fa9ead86db0a292dc7c55fd0f4d4b3b9b20d08ea8'
    checkpoint=ROOT/'runs/E1_yolov8s_uvh26_mv_640_seed42/weights/best.pt'
    assert sha256(checkpoint)=='9f1045381024445b50e33539791da224b732d61781c73b347013477ebb077fab'
    train=json.loads((root/'train_manifest.json').read_text());val=json.loads((root/'val_manifest.json').read_text())
    mapping=yaml.safe_load((ROOT/'configs/class_mapping.yaml').read_text())
    classes={}
    for r in train+val:
        label=root/r['label'];assert sha256(label)==r['label_sha256']
        classes[(r['split'],r['image_id'])]=[int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()]
    def counts(rows):
        return np.bincount([c for r in rows for c in classes[(r['split'],r['image_id'])]],minlength=14).tolist()
    cal,final=split_rows(val);assert_disjoint(cal,final)
    if min(counts(cal))==0 or min(counts(final))==0:raise ValueError('Seed42 split lacks class coverage; stop for protocol revision')
    train_small=random.Random(42).sample(train,128);cal_small=random.Random(42).sample(cal,64)
    names={'calibration_500':cal,'final_evaluation_1500':final,'preflight_train_128':train_small,'preflight_calibration_64':cal_small}
    hashes={}
    for name,rows in names.items():
        p=out/(name+'.json');p.write_text(json.dumps(rows,indent=2)+'\n');hashes[name]=dict(file_sha256=sha256(p),canonical_rows_sha256=rows_hash(rows),images=len(rows),object_counts=counts(rows))
    weights=class_weights(counts(train));image_weights=[image_weight(classes[(r['split'],r['image_id'])],weights) for r in train]
    import csv
    with (out/'image_sampling_weights.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['image_id','weight']);writer.writerows((r['image_id'],w) for r,w in zip(train,image_weights))
    draws=list(CappedWeightedSampler(image_weights));sampled=[train[i] for i in draws]
    proposal=dict(formula='min(3,sqrt(Nmax/Nc))',cap=3,class_names=[r['name'] for r in mapping],object_counts=counts(train),class_weights=weights,image_weight_policy='Mean of weights for unique classes; empty image weight1. Max3 exposures per image per epoch; exactly N weighted draws.',image_weight_min=min(image_weights),image_weight_max=max(image_weights),seed42_proposed_epoch_object_counts=counts(sampled),seed42_unique_images=len(set(draws)),seed42_max_repeats=int(max(np.bincount(draws))),loss_weighting=False,targeted_augmentation=False)
    (out/'weighting.json').write_text(json.dumps(proposal,indent=2)+'\n')
    (out/'protocol.json').write_text(json.dumps(dict(seed=42,dataset_version=str(root.relative_to(ROOT)),parent_manifest_sha256=expected,class_mapping_sha256=sha256(ROOT/'configs/class_mapping.yaml'),splits=hashes,background_id=0,foreground_mapping={str(i+1):r['name'] for i,r in enumerate(mapping)},final_evaluation_policy='No parameter tuning or model evaluation until final comparison; already used in historical E0/E1/E2 model selection, not independent unseen test data'),indent=2)+'\n')
    print('Protocol prepared:500 calibration /1500 reserved evaluation;128/64 preflight; all14classes in both main splits.')


if __name__=='__main__':main()
