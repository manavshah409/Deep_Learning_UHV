"""Inspect actual consensus JSON schemas before conversion."""

import argparse
from collections import Counter
from .common import paths, annotation_paths, load_coco, sha256, save_json, REVISION


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paths.local.yaml")
    a = p.parse_args()
    cfg = paths(a.config)
    result = {"revision": REVISION, "variants": {}}
    for variant in ("MV", "ST"):
        result["variants"][variant] = {}
        for split, path in annotation_paths(cfg["raw"], variant).items():
            if not path.exists():
                result["variants"][variant][split] = {"status": "missing"}
                continue
            d = load_coco(path)
            result["variants"][variant][split] = {
                "file": str(path.relative_to(cfg["raw"])),
                "sha256": sha256(path),
                "keys": list(d),
                "image_count": len(d["images"]),
                "object_count": len(d["annotations"]),
                "image_fields": dict(Counter(k for r in d["images"] for k in r)),
                "annotation_fields": dict(
                    Counter(k for r in d["annotations"] for k in r)
                ),
                "categories": d["categories"],
                "info": d.get("info"),
                "licenses": d.get("licenses"),
            }
    save_json(cfg["reports"] / "audit/schema.json", result)
    print(
        {
            v: {
                s: (r.get("image_count"), r.get("object_count"))
                for s, r in splits.items()
            }
            for v, splits in result["variants"].items()
        }
    )


if __name__ == "__main__":
    main()
