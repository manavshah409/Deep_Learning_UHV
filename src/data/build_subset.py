"""Seeded coverage-aware sampling inside each official split."""

import argparse
from collections import Counter
import json
import random
import yaml
from .common import paths, save_json, sha256


def select(rows, class_sets, n, seed):
    if n > len(rows) or n <= 0:
        raise ValueError("Invalid subset size")
    rng = random.Random(seed)
    candidates = sorted(rows, key=lambda r: r["image_id"])
    rng.shuffle(candidates)
    coverage = Counter(c for r in candidates for c in class_sets[r["image_id"]])
    chosen = []
    chosen_ids = set()
    uncovered = set(coverage)
    # Cover rare classes first, then seed-shuffled random fill. No alphabetical truncation.
    for cid in sorted(coverage, key=lambda c: (coverage[c], c)):
        if cid not in uncovered:
            continue
        row = next(
            r
            for r in candidates
            if cid in class_sets[r["image_id"]] and r["image_id"] not in chosen_ids
        )
        chosen.append(row)
        chosen_ids.add(row["image_id"])
        uncovered -= class_sets[row["image_id"]]
    if len(chosen) > n:
        raise ValueError("Subset cannot preserve coverage at requested size")
    chosen.extend(r for r in candidates if r["image_id"] not in chosen_ids)
    return sorted(chosen[:n], key=lambda r: r["image_id"])


def build(config, version, name, train, val, seed):
    cfg = paths(config)
    root = cfg["processed"] / version
    output = root / "subsets" / name
    if output.exists():
        raise ValueError("Immutable subset already exists; choose a new name")
    rows = json.loads((root / "manifest.json").read_text())
    selected = []
    distributions = []
    for split, n in [("train", train), ("val", val)]:
        population = [r for r in rows if r["split"] == split]
        counts = {
            r["image_id"]: Counter(
                int(line.split()[0])
                for line in (root / r["label"]).read_text().splitlines()
            )
            for r in population
        }
        chosen = select(population, {k: set(v) for k, v in counts.items()}, n, seed)
        selected += chosen
        full = sum(counts.values(), Counter())
        sub = sum((counts[r["image_id"]] for r in chosen), Counter())
        for cid in sorted(full):
            distributions.append(
                {
                    "split": split,
                    "class_id": cid,
                    "full_instances": full[cid],
                    "subset_instances": sub[cid],
                    "full_share": full[cid] / sum(full.values()),
                    "subset_share": sub[cid] / sum(sub.values()),
                }
            )
    output.mkdir(parents=True)
    save_json(output / "manifest.json", selected)
    # Relative entries starting ./ resolve against the text manifest location in Ultralytics.
    for split in ("train", "val"):
        (output / f"{split}.txt").write_text(
            "".join(f"./../../{r['image']}\n" for r in selected if r["split"] == split)
        )
    spec = yaml.safe_load((root / "dataset.yaml").read_text())
    spec.update(train=str(output / "train.txt"), val=str(output / "val.txt"))
    (output / "dataset.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    info = {
        "name": name,
        "seed": seed,
        "train": train,
        "val": val,
        "algorithm": "rare-class coverage followed by seeded random fill within official splits",
        "parent_manifest_sha256": sha256(root / "manifest.json"),
        "manifest_sha256": sha256(output / "manifest.json"),
    }
    save_json(output / "selection.json", info)
    save_json(cfg["reports"] / "tables" / f"{name}_selection.json", info)
    save_json(
        cfg["reports"] / "tables" / f"{name}_image_ids.json",
        [{"split": r["split"], "image_id": r["image_id"]} for r in selected],
    )
    import pandas as pd

    df = pd.DataFrame(distributions)
    df["share_difference_pp"] = 100 * (df.subset_share - df.full_share)
    df.to_csv(cfg["reports"] / "tables" / f"{name}_distribution.csv", index=False)
    print(output / "dataset.yaml")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paths.local.yaml")
    p.add_argument("--dataset-version", default="uvh26_mv_yolo_v1")
    p.add_argument("--name", required=True)
    p.add_argument("--train", type=int, default=8000)
    p.add_argument("--val", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    build(a.config, a.dataset_version, a.name, a.train, a.val, a.seed)


if __name__ == "__main__":
    main()
