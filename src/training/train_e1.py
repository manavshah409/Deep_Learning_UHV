"""Phase 2 runner derived from frozen E0 code; E0 source is never edited."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import yaml
from src.data.common import ROOT, sha256, save_json
from src.training.loss_checks import assert_finite_losses


def record_checkpoint(trainer, info, report):
    """Mirror which epoch Ultralytics actually writes to best.pt, including ties."""
    if trainer.best_fitness == trainer.fitness:
        info["best_epoch"] = int(trainer.epoch) + 1
        info["best_fitness_unrounded"] = float(trainer.fitness)
    info["last_saved_epoch"] = int(trainer.epoch) + 1
    save_json(report, info)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--config", default="configs/baseline_yolov8n.yaml")
    p.add_argument("--name", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--device")
    p.add_argument("--batch", type=int)
    p.add_argument("--workers", type=int)
    a = p.parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data/interim/ultralytics"))
    import torch
    import ultralytics
    from ultralytics import YOLO

    cfg = yaml.safe_load(Path(a.config).read_text())
    model_name = cfg.pop("model")
    for key in ("device", "batch", "workers"):
        if getattr(a, key) is not None:
            cfg[key] = getattr(a, key)
    if cfg["device"] == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "Requested MPS unavailable; preserve requested device and diagnose before changing it"
        )
    if a.preflight and a.smoke:
        raise ValueError("Preflight and smoke are distinct modes")
    if a.preflight:
        cfg["epochs"] = 1
        if a.name.endswith("_preflight_v2"):
            cfg["fraction"] = (
                0.1  # recovery pipeline check only: 800 train, all 2000 val
            )
    if a.smoke:
        cfg.update(epochs=1, close_mosaic=0, workers=0)
    if a.name not in {
        "E1_yolov8s_uvh26_mv_640_seed42",
        "E1_yolov8s_uvh26_mv_640_seed42_preflight_v1",
        "E1_yolov8s_uvh26_mv_640_seed42_preflight_v2",
    }:
        raise ValueError(
            "This registered runner only permits E1 and its unique preflight"
        )
    if a.smoke:
        raise ValueError("E1 uses the full frozen subset, not a smoke subset")
    if a.preflight != a.name.endswith(("_preflight_v1", "_preflight_v2")):
        raise ValueError("Preflight flag must match registered run name")
    output = ROOT / "runs" / a.name
    if output.exists():
        raise ValueError("Run exists; choose a unique name")
    data = Path(a.data).resolve()
    manifest = data.parent / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("Dataset manifest required")
    if not a.smoke:
        review = ROOT / "reports/audit/visual_review.json"
        if not review.exists():
            raise ValueError("Manual visual annotation review required")
        evidence = json.loads(review.read_text())
        if evidence.get("status") not in (
            "passed",
            "passed_with_documented_source_limitations",
        ) or evidence.get("manifest_sha256") != sha256(manifest):
            raise ValueError("Visual review must match the exact training manifest")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    info = {
        "experiment_id": a.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "source_sha256": sha256(__file__),
        "source_tree": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in sorted((ROOT / "src").rglob("*.py"))
        },
        "requirements_sha256": sha256(ROOT / "requirements.txt"),
        "dataset_manifest_sha256": sha256(manifest),
        "class_mapping_sha256": sha256(ROOT / "configs/class_mapping.yaml"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "config": cfg,
        "model": model_name,
        "smoke": a.smoke,
        "preflight": a.preflight,
        "config_sha256": sha256(a.config),
        "split_manifest_sha256": {
            s: sha256(data.parent / f"{s}_manifest.json")
            for s in ("train", "val")
            if (data.parent / f"{s}_manifest.json").exists()
        },
        "pretrained_source": f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{Path(model_name).name}",
        "status": "running",
    }
    report = ROOT / "reports/tables" / f"{a.name}_provenance.json"
    save_json(report, info)
    start = time.perf_counter()
    try:
        model = YOLO(model_name)
        info["pretrained_sha256"] = sha256(model_name)

        def check_finite_epoch(trainer):
            if trainer.tloss is not None:
                assert_finite_losses(trainer.tloss)

        model.add_callback("on_train_epoch_end", check_finite_epoch)

        def startup(trainer):
            groups = [
                {k: v for k, v in g.items() if k != "params"}
                for g in trainer.optimizer.param_groups
            ]
            info["startup_optimizer"] = {
                "name": type(trainer.optimizer).__name__,
                "groups": groups,
                "accumulate": trainer.accumulate,
                "nbs": trainer.args.nbs,
                "effective_workers": trainer.args.workers,
                "training_images": len(trainer.train_loader.dataset),
                "validation_images": len(trainer.test_loader.dataset),
            }
            if not a.smoke:
                if type(trainer.optimizer).__name__ != "AdamW" or any(
                    abs(g["lr"] - cfg["lr0"]) > 1e-12 for g in groups
                ):
                    raise RuntimeError(
                        "Effective optimizer differs from frozen configuration"
                    )
            save_json(report, info)

        model.add_callback("on_pretrain_routine_end", startup)
        model.add_callback(
            "on_model_save", lambda trainer: record_checkpoint(trainer, info, report)
        )
        model.train(
            data=str(data),
            project=str(ROOT / "runs"),
            name=a.name,
            exist_ok=False,
            **cfg,
        )
        import pandas as pd

        results = pd.read_csv(output / "results.csv")
        results.columns = results.columns.str.strip()
        losscols = [c for c in results if "loss" in c]
        import numpy as np

        if not np.isfinite(results[losscols].to_numpy()).all():
            raise RuntimeError("Non-finite losses")
        best = output / "weights/best.pt"
        if not best.is_file():
            raise RuntimeError("No best checkpoint")
        info.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=time.perf_counter() - start,
            epochs_completed=len(results),
            early_stopped=len(results) < cfg["epochs"],
            weights={
                "path": str(best.relative_to(ROOT)),
                "bytes": best.stat().st_size,
                "sha256": sha256(best),
            },
            best_epoch_from_validation_map5095=int(
                results.loc[results["metrics/mAP50-95(B)"].idxmax(), "epoch"]
            ),
        )
        info["last_weights"] = {
            "path": str((output / "weights/last.pt").relative_to(ROOT)),
            "bytes": (output / "weights/last.pt").stat().st_size,
            "sha256": sha256(output / "weights/last.pt"),
        }
        info["effective_amp"] = bool(model.trainer.amp)
        info["effective_workers"] = int(model.trainer.args.workers)
        info["effective_batch"] = int(model.trainer.batch_size)
        info["optimizer_class"] = type(model.trainer.optimizer).__name__
        info["final_epoch_metrics"] = results.iloc[-1].to_dict()
        info["best_epoch_metrics"] = (
            results.loc[results["epoch"] == info["best_epoch"]].iloc[0].to_dict()
        )
        args = vars(model.trainer.args)
        info["augmentation"] = {
            k: args.get(k)
            for k in (
                "hsv_h",
                "hsv_s",
                "hsv_v",
                "degrees",
                "translate",
                "scale",
                "shear",
                "perspective",
                "flipud",
                "fliplr",
                "mosaic",
                "mixup",
                "cutmix",
                "copy_paste",
                "auto_augment",
                "erasing",
            )
        }
        info["optimizer_parameter_groups"] = [
            {k: v for k, v in group.items() if k != "params"}
            for group in model.trainer.optimizer.param_groups
        ]
        info["dataset_version"] = (
            json.loads((data.parent / "version.json").read_text())
            if (data.parent / "version.json").exists()
            else data.parent.name
        )
        info["mps_determinism_limitation"] = (
            "MPS scatter_reduce and index_put_with_accumulate warned of nondeterministic implementations during smoke run; seed is reproducible, exact numerical replay is not guaranteed."
        )
        import shutil

        shutil.copy2(
            output / "results.csv", ROOT / "reports/tables" / f"{a.name}_training.csv"
        )
        for figure in output.glob("*.png"):
            if "batch" not in figure.name:
                shutil.copy2(
                    figure, ROOT / "reports/figures" / f"{a.name}_{figure.name}"
                )
        # CSV-derived argmax is retained as a cross-check; best_epoch follows actual checkpoint saves.
        save_json(report, info)
        print(json.dumps(info, indent=2))
    except BaseException as e:
        info.update(
            status="failed",
            duration_seconds=time.perf_counter() - start,
            error=f"{type(e).__name__}: {e}",
        )
        save_json(report, info)
        raise


if __name__ == "__main__":
    main()
