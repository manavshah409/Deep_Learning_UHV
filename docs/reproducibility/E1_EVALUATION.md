# Reproduce E1 evaluation without retraining

Use the existing project virtual environment and frozen local dataset. Checkpoints are deliberately excluded from Git. E0 best SHA-256: `85b9e089ec3dfaa676e30a6191b7e5726f320d5bf85391f1675922b88f9778b3`; E1 best: `9f1045381024445b50e33539791da224b732d61781c73b347013477ebb077fab`. Never rerun the training pipeline to repair an evaluation.

The delivered evaluation IDs end in `_validation_seed42_v2`, benchmark IDs are `E0_common_protocol_v2` and `E1_common_protocol_v2`. Their existing directories are immutable. To reproduce, choose new unique IDs. From the repository root:

```bash
RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)
DATA=data/processed/uvh26_mv_yolo_v1/subsets/baseline_seed42_v2/dataset.yaml
E0=runs/yolov8n_uvh26_mv_baseline_seed42_v1/weights/best.pt
E1=runs/E1_yolov8s_uvh26_mv_640_seed42/weights/best.pt

.venv/bin/python -m src.evaluation.evaluate_baseline --weights "$E0" --data "$DATA" --name "E0_validation_$RUN_TAG" --device mps --batch 8
.venv/bin/python -m src.evaluation.evaluate_baseline --weights "$E1" --data "$DATA" --name "E1_validation_$RUN_TAG" --device mps --batch 8
.venv/bin/python -m src.evaluation.benchmark_inference --weights "$E0" --data "$DATA" --name "E0_timing_$RUN_TAG" --device mps --count 100 --warmup 10
.venv/bin/python -m src.evaluation.benchmark_inference --weights "$E1" --data "$DATA" --name "E1_timing_$RUN_TAG" --device mps --count 100 --warmup 10
```

Run each command only after the previous one exits successfully. Keep the machine on AC power, awake, and free of competing GPU jobs. These commands do not train. The evaluator fixes all remaining protocol settings in source and refuses existing IDs. It publishes `reports/evaluations/<ID>/COMPLETE.json` only after all metric/curve exports succeed. A run directory alone is not proof of completed export.

To inspect the delivered results without rerunning inference:

```bash
.venv/bin/python scripts/validate_e1_artifacts.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/e1_status.py
.venv/bin/python -m streamlit run app.py
```

The delivered comparison can be rebuilt from its measured inputs using `src.evaluation.compare_e1` and the four explicit `--e0-evaluation`, `--e1-evaluation`, `--e0-timing`, `--e1-timing` arguments. It refuses an existing comparison directory; preserve the delivered comparison and use an isolated checkout for a fresh rebuild. `scripts/run_e1_recovery.py` is the one-time post-evaluation supervisor and also refuses its existing status/output files.

Paired figures are local at `reports/predictions/E1_vs_E0_paired_v1/`, excluded from Git because they contain dataset imagery. The committed `reports/comparisons/E1_paired_summary.json` records reviewed findings. The faculty Phase 1 PDF/ZIP remain the historical E0 pack; the Phase 2 report and dashboard are the E1 deliverables.
