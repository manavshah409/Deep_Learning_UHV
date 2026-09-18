"""Render measured E2 Gate A area AP and latency charts."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data.common import ROOT


def main():
    root = ROOT / "reports/comparisons/E2_gateA_v2"
    sizes = pd.read_csv(root / "size_ap.csv")
    timing = pd.read_csv(root / "latency.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(sizes))
    axes[0].bar(x - 0.18, 100 * sizes.ap50_95_640, 0.36, label="640")
    axes[0].bar(x + 0.18, 100 * sizes.ap50_95_960, 0.36, label="960")
    axes[0].set_xticks(x, sizes["size"])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("COCO-area AP50:95 (%)")
    axes[0].legend()
    axes[0].set_title("Frozen original-image areas; same checkpoint")
    stage = timing[timing.metric == "mean_ms"]
    x = np.arange(len(stage))
    axes[1].bar(x - 0.18, stage.res640, 0.36, label="640")
    axes[1].bar(x + 0.18, stage.res960, 0.36, label="960")
    axes[1].set_xticks(x, ["Preprocess", "Inference", "Postprocess", "File-to-result"])
    axes[1].set_ylabel("Mean synchronized latency (ms)")
    axes[1].legend()
    axes[1].set_title("Batch1 / 100 images; still-image timing only")
    fig.tight_layout()
    fig.savefig(root / "area_accuracy_latency.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
