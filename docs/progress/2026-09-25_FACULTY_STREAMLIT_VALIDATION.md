# Faculty app completion evidence — 2026-09-25

- 24 focused dashboard tests passed; final full local suite 243 passed in 7.85 seconds. Full count includes 19 existing untracked Stage F/Phase 3 tests, which are not part of this commit.
- Ruff passed on new app/services/pages, package builder and tests.
- 840 pre-existing report files remain byte-identical. Stage F read-only validator passed; no scientific outputs changed.
- Actual E1 CPU inference passed on a generated blank image; no dataset image read.
- Local Streamlit startup at port 8511 passed; browser overview visually checked after correction to metric-card contrast. Screenshot evidence is in the task transcript.
- Portable ZIP extracted successfully; per-file hashes and Saved Evidence startup passed without weights/dataset. Final v2 differs only in the report’s corrected visual-check record.
- Commands: pytest -q; pytest tests/test_faculty_dashboard.py -q; ruff check dashboard tests/test_faculty_dashboard.py scripts/package_faculty_demo.py; scripts/package_faculty_demo.py --output deliverables/UVH26_Faculty_Streamlit_20260925_v2.zip; Streamlit startup.
- Package location: ignored deliverables/UVH26_Faculty_Streamlit_20260925_v2.zip.
- Package receipt: {"files": 28, "bytes": 547135, "sha256": "a271b158f3065ced44457ff950b5a1fe44c973e598424060a81054e390082e6a"}

Limitations: video UI explicitly deferred; no permitted photographic samples or annotated photographs bundled; actual image analytics require a user upload and local weights. No reserved evaluation, training, fusion or Phase 3 work. Existing Stage F and unrelated Phase 3 changes remain unstaged. Branch master; delivery commit is the commit introducing this note. Push outcome will be reported in the handover.
