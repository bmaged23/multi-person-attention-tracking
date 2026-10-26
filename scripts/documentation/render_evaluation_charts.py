"""Render scoped native latency and model-quality charts from saved measurements."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/evaluation/measurements"
OUT = ROOT / "docs/evaluation/charts"


def load(name):
    return json.loads((DATA / (name + ".json")).read_text())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    detector = [
        load(n)
        for n in [
            "detector-opencv-baseline",
            "detector-onnx-cpu",
            "detector-onnx-cuda-v2",
        ]
    ]
    cpu = load("workflow-original-cpu")
    gpu = load("workflow-cuda-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    values = [d["warm_median_seconds"] * 1000 for d in detector]
    bars = axes[0].bar(
        ["OpenCV CPU", "ONNX CPU", "ONNX CUDA"],
        values,
        color=["#7895bd", "#a7bdd9", "#20a887"],
    )
    axes[0].bar_label(bars, fmt="%.0f ms", padding=4)
    axes[0].set_ylim(0, max(values) * 1.22)
    axes[0].set_title("Person detection · 64 fixed images")
    axes[0].set_ylabel("Median warm detection latency (ms)")
    totals = [
        sum(c["performance"]["elapsed_seconds"] for c in d["clips"]) for d in [cpu, gpu]
    ]
    bars = axes[1].bar(
        ["Original CPU", "GPU-assisted"], totals, color=["#7895bd", "#20a887"]
    )
    axes[1].bar_label(bars, fmt="%.2f s", padding=4)
    axes[1].set_ylim(0, max(totals) * 1.22)
    axes[1].set_title("Complete workflow · 100 AMI frames")
    axes[1].set_ylabel("Seconds including annotation and encoding")
    fig.suptitle(
        "Measured native runtime comparison · GTX1660Ti · 8 CPU landmark workers",
        fontsize=12,
    )
    save(fig, "latency-comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, pair, title in zip(
        axes,
        [
            ("detector-opencv-baseline", "detector-cuda-confidence035"),
            ("holdout-baseline", "holdout-confidence035"),
        ],
        ["Exposed development · 64 images", "Reserved validation · 36 images"],
    ):
        a, b = [load(n) for n in pair]
        x = np.arange(3)
        ax.bar(
            x - 0.18,
            [a[k] * 100 for k in ["precision", "recall", "f1"]],
            0.36,
            label="Confidence0.15 (default)",
            color="#257baa",
        )
        ax.bar(
            x + 0.18,
            [b[k] * 100 for k in ["precision", "recall", "f1"]],
            0.36,
            label="Confidence0.35 (optional)",
            color="#e6a348",
        )
        ax.set_xticks(x, ["Precision", "Recall", "F1"])
        ax.set_ylim(0, 100)
        ax.set_title(title)
        ax.set_ylabel("Person-box metric at IoU0.5 (%)")
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Human COCO annotations · operating-point metrics, not COCO AP", fontsize=12
    )
    save(fig, "detection-quality.png")
    tracking = load("tracking-baseline")["results"]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(
        [r["case_id"] for r in tracking],
        [r["idf1"] * 100 for r in tracking],
        color=["#20a887"] * 3 + ["#e6a348"],
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set_ylim(0, 115)
    ax.set_ylabel("IDF1 (%)")
    ax.set_title("Synthetic perfect-box trajectories · crossing remains a limitation")
    save(fig, "tracking-stress.png")
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(2)
    for key, label, color in [
        ("person_detection", "Person detector", "#257baa"),
        ("landmarks", "Head/face landmarks", "#20a887"),
        ("scoring_tracking_reporting", "Scoring, tracking, reports", "#e6a348"),
        ("annotation_encoding", "Annotation/write", "#b37ec9"),
        ("final_encoding", "Final H.264 encode", "#dd7c76"),
        ("decode", "Input decode", "#a5a5a5"),
    ]:
        values = [
            sum(c["performance"]["stage_seconds"][key] for c in d["clips"])
            for d in [cpu, gpu]
        ]
        ax.barh(
            ["Original CPU", "GPU-assisted"],
            values,
            left=bottom,
            label=label,
            color=color,
        )
        bottom += values
    ax.set_xlabel("Seconds across two 50-frame AMI clips")
    ax.set_title("Measured stages · detector is the main original bottleneck")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "stage-breakdown.png")


if __name__ == "__main__":
    main()
