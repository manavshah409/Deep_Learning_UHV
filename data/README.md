# UVH-26 data

Official source: https://huggingface.co/datasets/iisc-aim/UVH-26
Pinned revision: `59f82c57821e8a54dc40bc1f42e83909dbad0b70`.
Dataset license: CC BY 4.0, as stated by the official repository. Retain attribution and the original dataset card locally.

Raw files are immutable and ignored by Git. Download using:

```bash
hf download iisc-aim/UVH-26 --repo-type dataset \
  --revision 59f82c57821e8a54dc40bc1f42e83909dbad0b70 \
  --local-dir data/raw/UVH-26 --max-workers 8
```

Check storage first; the Hub lists approximately 90 GB. `interim/` holds local transfer logs and caches. `processed/` holds versioned labels, relative image symlinks and manifests. No complete image copy is created. `samples/` is for local diagnostic data.

Copy `configs/paths.example.yaml` to `configs/paths.local.yaml`. Paths resolve relative to the project root, independent of the shell working directory. Never add raw images, annotations, model weights or local path configurations to Git.

Citation: Akash Sharma et al., *Towards Image Annotations and Accurate Vision Models for Indian Traffic, Preliminary Dataset Release, UVH-26-v1.0*, Indian Institute of Science Technical Report, November 2025, https://doi.org/10.48550/arXiv.2511.02563.

## Completed Phase 1 subset

Frozen `uvh26_mv_yolo_v1/subsets/baseline_seed42_v2` contains 8,000 training images / 94,609 objects and 2,000 validation images / 24,342 objects. All selected images passed decode, dimensions, file/label hashes and split-leakage checks. The full 26,646-image annotation-catalogue audit is metadata-only; acquisition/integrity verification of unselected images remains incomplete.

Combined manifest SHA-256: `990054a93e300a90321db19b3d0bcd98a488a891cd4e2dbd88425f4eb592c2af`. Mapping SHA-256: `6fd0b458fed2cb174c924d4fa9ead86db0a292dc7c55fd0f4d4b3b9b20d08ea8`. Detailed split hashes, quarantine policy and frozen recovery addendum are linked from `docs/phase_reports/PHASE_1_BASELINE.md`. Raw files and labels must remain unchanged; source limitations are documented separately from model errors.
