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
