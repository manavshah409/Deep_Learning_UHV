# Stage F delivery progress — 2026-09-24

Work completed: formal rejection of E4, actual sequential-inference benchmark, reports/faculty summary/model overview/dashboard and delivery validation. No additional parameter search followed rejection.

Exact commands:

```bash
.venv/bin/python scripts/validate_stage_f.py --local
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q tests/test_fusion.py
.venv/bin/ruff check src/evaluation/fusion.py scripts/stage_f_fusion.py scripts/benchmark_stage_f.py scripts/validate_stage_f.py tests/test_fusion.py components/fusion.py components/e3.py
```

Inputs and hashes: Stage F protocol `7cddd248587f612a0f1e0852799e7dbead93c0202fe7bb8d2983d92ab3144f7a`; benchmark-only final amendment `b56302f44a8e2eb2a14b467bf9be9451bd9520a63afddf6ae1660fe0de6bea41`. `reports/accuracy_stage_f/integrity.json` lists checkpoint, calibration, mapping and prediction hashes. Reserved data were not accessed, enumerated or hashed.

Measurements: strongest WBF pooled OOF AP50:95 0.492431606 vs E1 0.482828904; macro F1 0.594704994 vs0.586175239. Neither P nor R reached0.70. Actual pipeline156.776ms mean/157.086ms median/162.078ms p95,6.379 still images/s.219 local tests (215 eligible plus4 existing untracked Phase3 cases) and15 fusion-specific tests passed; Ruff and dashboard checks passed;138 historical files unchanged.

Failures or deviations: no scientific process or integrity failure (CV and benchmark exit0). A lint-only unused-local rename after benchmark execution is recorded with exact reversible source-hash verification in `benchmark_lint_provenance.json`. One existing app import-order diagnostic remains unchanged. No extra search, new training or reserved evaluation occurred.

Remaining work: only Git delivery for Stage F. E5 data/label improvement and controlled fine-tuning remain proposals requiring a separately frozen protocol and authorization; do not start them in this task.

Git: base commit `9175ed5f86328166ee0c83f52f6fd58d082ce126`; the delivery commit is the commit introducing this note (`git log -1 -- docs/progress/2026-09-24_STAGE_F_DELIVERY.md`).
