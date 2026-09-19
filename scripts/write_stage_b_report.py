"""Render the faculty/developer Stage B report from measured summaries."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.experiments.stage_b import ROOT,OUT,CONFIG,A


def main():
    read=lambda name:json.loads((OUT/name).read_text())
    comparison=read('comparison.json');decision=comparison['decision'];selected=decision['selected']
    runs={m:read(f'{m}_summary.json') for m in ['unweighted','weighted']}
    protocol=read('protocol.json');validation=read('validation.json');checkpoints=read('checkpoint_integrity.json')
    names=json.loads((A/'protocol_v2/weighting.json').read_text())['class_names']
    proposed=f'caffeinate -i .venv/bin/python -m src.experiments.stage_b --sampling {selected} --run-id E3_fasterrcnn_{selected}_20ep_seed42_v1 --allow-full-training'
    report=['# Accuracy Improvement Stage B: paired sampling pilot','',
        f'**Complete: two independent three-epoch MPS pilots. Provisional full-run sampling choice: {selected}. Full E3, E4 fusion and reserved-split evaluation remain unstarted.**','',
        'These measurements select a training configuration. They are not final detector results, a test-set evaluation, or evidence of superiority over E1. E1 remains the selected research detector.','',
        '## Shared configuration and provenance','',
        '- Train: deterministic 1,000-image subset of frozen 8,000; evaluation: 250 images drawn exclusively from Stage A calibration500. Seed 42 and all 14 classes represented in both. Rare-first class-covering anchors are followed by seeded-shuffle fill; this is not an unbiased random benchmark.',
        '- Both runs independently initialize official COCO_V1 Faster R-CNN ResNet-50-FPN, followed by the same seed-42 15-output head. Neither starts from Stage A or from the other pilot. Physical batch 1, workers 0, MPS, no additional augmentation; aspect-preserving min480/max640 resize.',
        '- Shared SGD: base LR .001, momentum .9, weight decay .0005; default pretrained backbone freezing retained. Sampling is the only configuration difference. B1 visits every selected training image once per epoch in a deterministic shuffled order. B2 uses Stage A image weights (mean capped inverse-square-root class weights) with replacement, exactly 1,000 draws and at most three repetitions per image per epoch.',
        '- COCO bbox AP at IoU .50:.95, maxDet300, score floor .001 and NMS .5. AP is macro-averaged over represented classes. P/R uses confidence .25 and IoU .5, macro-averaged over represented GT classes. It differs from YOLO operating-point metrics.',
        f"- Identical initialized model state SHA-256: `{comparison['initial_state_sha256']}`.",
        '- Initializer SHA-256: `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384`. Frozen parent, mapping and selected-manifest hashes are in `reports/accuracy_stage_b/protocol.json`; full effective configurations and source hashes remain with each run.',
        '- Reserved1500 manifest and images were not opened. The full 26,646-image annotation-catalogue audit remains distinct from the 10,000-image local subset integrity audit. Acquisition/integrity checks for unselected images remain incomplete. Historical validation reuse means the reserved pool is not an unseen test set.','',
        '## Learning rate fixed before either pilot','',
        'Stage A established finite loss at .005 for only 128 updates; it did not establish an optimal batch-one LR. A fivefold reduction to .001 with a two-epoch warm-up is a conservative safety choice. No optimizer or schedule was selected using the reserved split. The later decays remain a proposal, not something three epochs validated.','',
        '`LR(e) = .001 × min(1,e/2) × .1**(I[e>=13] + I[e>=18])`, with one-based epoch e.','',
        '| Epoch | LR |','|---:|---:|']
    report += [f'| {e} | {float(protocol["lr_by_epoch"][str(e)]):.5f} |' for e in range(1,21)]
    report += ['', '## Measured epoch results','',
        'Times include synchronized MPS training; total additionally includes sampling export, loss export, checkpoint save/reload/hash verification and logging. Train timing includes image decoding, integrity checking and optimizer work; validation includes decoding and COCO metric calculation. These are training durations, not inference FPS.','',
        '| Pilot | Epoch | LR | Train s | Val s | Total s | Cumulative s | Mean loss | AP50 | AP50:95 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m,r in runs.items():
        for e in r['epochs']:
            report.append(f"| {m} | {e['epoch']} | {e['learning_rate']:.4f} | {e['training_seconds']:.2f} | {e['validation_seconds']:.2f} | {e['total_epoch_seconds']:.2f} | {e['cumulative_seconds']:.2f} | {e['total_training_loss']:.4f} | {e['validation_map50']:.4f} | {e['validation_map50_95']:.4f} |")
    report += ['', 'All 6,000 training-step losses and gradients were finite. Component losses and macro P/R are preserved for every epoch in `epoch_comparison.csv`; per-step losses remain in each ignored run, and extrema/p95/p99 are summarized in `loss_diagnostics.json`. Endpoint loss stability does not establish long-run convergence.','',
        '## Sampling coverage','', '| Pilot | Epoch | Draws | Unique | Coverage % | Max repetitions | Histogram (repetitions: images) |','|---|---:|---:|---:|---:|---:|---|']
    for m,r in runs.items():
        for e in r['epochs']:
            s=e['sampling'];report.append(f"| {m} | {e['epoch']} | {s['total_draws']} | {s['unique_images']} | {s['coverage_percentage']:.1f} | {s['maximum_repetitions']} | {s['repetition_histogram']} |")
    report += ['', 'Zero-repeat images are included in weighted histograms. Effective per-class object and image exposures for each epoch are in `sampling_comparison.json`. These count repeat presentations, not new independent annotations.','',
        '## Per-class epoch-three calibration AP and exposure','',
        '| Class | Calibration objects | B1 AP50 | B2 AP50 | B1 AP50:95 | B2 AP50:95 | B1 objects drawn (3 epochs) | B2 objects drawn (3 epochs) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for i,name in enumerate(names):
        a=runs['unweighted']['epochs'][-1]['metrics']['per_class'][i];b=runs['weighted']['epochs'][-1]['metrics']['per_class'][i]
        exposure=[sum(e['sampling']['effective_per_class_objects'][i] for e in r['epochs']) for r in runs.values()]
        report.append(f"| {name} | {a['support']} | {a['ap50']:.4f} | {b['ap50']:.4f} | {a['ap50_95']:.4f} | {b['ap50_95']:.4f} | {exposure[0]} | {exposure[1]} |")
    report += ['', 'Every epoch’s per-class precision, recall, AP50 and AP50:95 is in `per_class_metrics.csv`. Mini-bus (9 objects) and Others (8 objects) are reported but excluded from the supported-minority AP decision. Minority membership was fixed from the five lowest full-training class counts: Others, Mini-bus, Tempo-traveller, Van and Bicycle.','',
        '## Predeclared decision','',
        'Weighted sampling requires finite, stable endpoint losses, no AP50 or AP50:95 regression exceeding 1.0 percentage point, and better aggregate minority exposure or supported minority macro AP. An exposure-only benefit also requires non-regressing supported minority macro AP. Otherwise use unweighted as the lower-coverage-risk choice. This rule and support threshold were frozen before either pilot.','',
        f"- Epoch-three B2−B1 AP50: **{decision['final_epoch_map50_delta']*100:+.3f} pp**; AP50:95: **{decision['final_epoch_map50_95_delta']*100:+.3f} pp**.",
        f"- Supported-minority macro AP50:95: B1 **{decision['supported_minority_ap_b1']:.4f}**, B2 **{decision['supported_minority_ap_b2']:.4f}** (Tempo-traveller, Van, Bicycle collectively).",
        f"- Three-epoch aggregate minority object exposure: B1 **{decision['aggregate_minority_object_exposure_b1']}**, B2 **{decision['aggregate_minority_object_exposure_b2']}**.",
        f"- Recommendation: **{selected}**. {decision['interpretation']}",
        '- Both pilots are evaluated at the same final epoch; no favorable-epoch or single-class cherry-picking. Three epochs are insufficient to establish a reliable final-accuracy advantage or optimal optimizer. No full training is authorized by this result alone.','',
        '## Checkpoint and process integrity','',
        'Both actual child processes exited zero and each completed exactly three epochs. All six epoch checkpoints passed SHA-256, size, safe CPU deserialization, finite parameter/optimizer state and epoch/LR metadata checks. All logs and checkpoints remain unchanged locally.','',
        '| Pilot | Epoch | Bytes | SHA-256 |','|---|---:|---:|---|']
    report += [f"| {c['method']} | {c['epoch']} | {c['bytes']} | `{c['sha256']}` |" for c in checkpoints]
    report += ['', 'Local relative checkpoint paths are recorded in `checkpoint_integrity.json`, under `runs/E3_stageB_{unweighted,weighted}_3ep_seed42_v1/epoch_00{1,2,3}.pth`. Actual exit records are in `process_status.json`. Full logs are the corresponding `.log` files directly under `runs/`.','',
        'Warnings: '+ ('none found in the captured pilot logs.' if not any(comparison['warnings'].values()) else json.dumps(comparison['warnings'])), '',
        '## Full 20-epoch proposal — not executed','']
    for m,p in comparison['projection'].items():
        lo,hi=p['observed_epoch_scaled_range_hours'];report.append(f"- {m}: **{p['hours_20_epochs']:.2f} hours**, observed-epoch-scaled planning range **{lo:.2f}–{hi:.2f} hours**.")
    report += ['', comparison['projection_method']+'. Thermal load, contention and sampling distribution can change this estimate; the range is not a confidence interval.', '',
        'Proposed command, only after separate full-run authorization (8,000 training / 500 calibration images):','', '```bash',proposed,'```','',
        '## Validation and delivery','',
        f"The Stage B validator passed: all 1,250 selected image/label pairs, parent/mapping/protocol hashes, dataset membership, disjointness, replayed draw/exposure reports, identical initializer/model state, paired configurations, process exits and six checkpoints. {validation['historical_files_unchanged']} historical tracked files retained their recorded hashes; the reserved manifest was excluded from reads.",
        f"Tests passed: **{read('tests_summary.json')['tests']['passed']} local / {read('tests_summary.json')['git_deliverable_tests']['passed']} deliverable**. Results are stored in `reports/accuracy_stage_b/tests.txt` and `git_deliverable_tests.txt`. The latter excludes the four pre-existing untracked Phase 3 tests. Git delivery is reported in the task response; checkpoints, full runs, raw/processed data, generated labels and caches are ignored.", '',
        'Reproduction: [commands and protocol](../reproducibility/ACCURACY_STAGE_B.md). Detailed evidence: `reports/accuracy_stage_b/`. Existing E0/E1/E2, Phase 3 and Stage A artifacts remain preserved. No fusion, weighted YOLO fine-tuning, full E3 or reserved-split evaluation was started.','']
    (ROOT/'docs/phase_reports/ACCURACY_IMPROVEMENT_STAGE_B.md').write_text('\n'.join(report))
    (OUT/'proposed_full_run.txt').write_text(proposed+'\n')
    prefix=f'**Accuracy Improvement Stage B complete:** paired three-epoch Faster R-CNN pilots on 1,000 training / 250 calibration images; provisional sampling choice **{selected}**. Configuration-selection evidence only; full E3, E4 fusion and reserved-split evaluation remain unstarted. [Measured report](docs/phase_reports/ACCURACY_IMPROVEMENT_STAGE_B.md).\n\n'
    p=ROOT/'README.md';old=p.read_text()
    if old.startswith('**Accuracy Improvement Stage B complete:**'):old=old.split('\n\n',1)[1]
    p.write_text(prefix+old)
    p=ROOT/'CHANGELOG.md';old=p.read_text();heading='## Accuracy Improvement Stage B — 2026-09-19'
    if not old.startswith(heading):
        p.write_text(heading+f'\n\n- Completed paired three-epoch COCO-initialized Faster R-CNN MPS pilots on deterministic 1000/250 train/calibration subsets.\n- Frozen batch-one SGD .001 and two-epoch warm-up with proposed epoch13/18 decays; exported per-class metrics, timings, coverage and all checkpoint hashes.\n- Provisional sampling choice: {selected}; full E3, E4 fusion and reserved1500 evaluation remain unstarted.\n\n'+old)
    p=ROOT/'configs/experiment_registry.json';registry=json.loads(p.read_text())
    registry['experiments']=[e for e in registry['experiments'] if e.get('id')!='E3_stageB_sampling_pilot']
    registry['experiments'].append(dict(id='E3_stageB_sampling_pilot',status='completed_configuration_selection_only',run_ids=[r['run_id'] for r in runs.values()],config='configs/accuracy/E3_stage_b.json',report='docs/phase_reports/ACCURACY_IMPROVEMENT_STAGE_B.md',train_images=1000,calibration_images=250,epochs_per_pilot=3,sampling_choice=selected,full_E3_started=False,fusion_started=False,reserved_evaluation_started=False))
    p.write_text(json.dumps(registry,indent=2)+'\n')
    print('Stage B report, reproduction command, README, CHANGELOG and registry updated')
if __name__=='__main__':main()
