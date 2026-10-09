# New data dashboard implementation progress

## Stage 0
Inspected dashboard/app.py, pages.py, services.py, common_metrics.py, existing tests, Git status and origin/master. Reuse frozen detector, hash verification, common COCO metrics and historical pages. Add isolated ingestion/audit, diagnostic analysis, immutable exports, CLI and new navigation. Existing Stage F/Phase 3 changes remain out of scope.

Boundary incident: the initial preservation snapshot recursively hashed files under reports, including potentially reserved manifest metadata. No reserved images were accessed and no reserved evaluation occurred. Subsequent preservation checks exclude reserved-named paths. This incident prevents claiming that reserved metadata was entirely unopened in this task.

## Plan
1. Validate memory-only image/YOLO/COCO uploads and provenance declarations.
2. Compute labeled diagnostics separately from unlabeled prediction summaries; export immutable bundles.
3. Connect eight faculty pages, historical fallback and explorer.
4. Test synthetic contracts, full suite, existing validators and startup; document and deliver eligible changes.

## Backend and UI implementation
Added dashboard/new_data/{ingest,analysis,runner,views}.py and scripts/evaluate_new_data.py; updated app navigation and synthetic AppTest contracts. Supports strict YOLO/COCO validation, memory-only unlabeled inference, common COCO evaluation, fixed matching/taxonomy, support-aware plots, bootstrap intervals, timing, immutable bundles and saved-record reconstruction.

Focused suite: 56 passed before final additions. Actual E1 CPU blank-image functionality smoke passed, including export and saved-result reconstruction; no real image was used, no accuracy claim. Early failures were test-path resolution, expected float rounding and monkeypatch target; fixed without weakening scientific checks. Ultralytics deprecated half argument replaced with explicit quantize32 after checking installed configuration.

Remaining: full suite, artifact validators, preservation, startup and Git delivery. Limitations are explicit in technical report, including no AP bootstrap, strata recall rather than AP, no unannotated environmental inference and no authenticated provenance for user-imported records.

## Final verification 2026-10-09
- Full suite: 275 passed in14.39s (`.venv/bin/python -m pytest -q`). Focused and populated AppTest pages are included.
- Ruff passed for dashboard/new_data, dashboard/app.py, offline CLI and dashboard tests. Git diff whitespace check passed.
- Existing read-only Stage D, Stage E and Stage F validators passed without --local; Stage F verified138 historical files and Stage D72. No historical report output changed in Git.
- Local-only Streamlit server started on127.0.0.1:8512 after sandbox socket-binding permission was granted. Browser showed the new eight-page navigation, dataset-status banner and correctly labelled historical fallback.
- Initial preservation snapshot included protocol_v1/final_evaluation_1500.json and protocol_v2/final_evaluation_1500.json bytes for hashing. This was an unintended metadata access; no reserved image was read or evaluated. Snapshot was temporary and is not delivered.
- README, README_DEMO, CHANGELOG, experiment registry, technical report and reproduction guide updated. Shared files are staged from HEAD plus only this task's additions, preserving unrelated working-tree changes.
- Deliberately excluded: existing Stage F/Phase3 files, checkpoints, reports/prediction archives, datasets, uploads, logs, caches and local smoke-run output. No model training, fine-tuning, fusion, video tracking or public deployment occurred.
