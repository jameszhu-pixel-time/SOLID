#!/usr/bin/env python3
"""Regenerate the result figures displayed in the repository README."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "assets" / "results"
FIGURES = ROOT / "assets" / "figures"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#30363d",
            "axes.labelcolor": "#24292f",
            "axes.titleweight": "bold",
            "font.size": 10,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def plot_main_results() -> None:
    rows = read_csv("checkpoint_125_175_pass_at_k.csv")
    rows = [
        row
        for row in rows
        if row["model"] == "Qwen3-4B"
        and row["checkpoint"] == "125"
        and row["method"] in {"TTRL", "SOLID"}
        and row["dataset"] in {"OptMATH", "MAMO", "InOR"}
    ]

    by_method_dataset = {(row["method"], row["dataset"]): row for row in rows}
    datasets = ["OptMATH", "MAMO", "InOR"]
    metrics = [
        ("majority_accuracy_pct", "maj@N"),
        ("pass_at_1_pct", "pass@1"),
        ("pass_at_2_pct", "pass@2"),
        ("pass_at_4_pct", "pass@4"),
    ]
    colors = {"TTRL": "#4C78A8", "SOLID": "#B279A2"}

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.4), sharex=True)
    x = range(len(datasets))
    width = 0.36
    for ax, (field, label) in zip(axes.flat, metrics):
        for offset, method in ((-width / 2, "TTRL"), (width / 2, "SOLID")):
            values = [float(by_method_dataset[(method, dataset)][field]) for dataset in datasets]
            bars = ax.bar(
                [index + offset for index in x],
                values,
                width,
                label=method,
                color=colors[method],
            )
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
        ax.set_title(label)
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(list(x), datasets)
        ax.grid(axis="y")
        ax.set_ylim(0, max(ax.get_ylim()[1], 65))

    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Qwen3-4B-Instruct: matched-budget results at step 125", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "main_results.png", dpi=220)
    plt.close(fig)


def plot_training_curve() -> None:
    rows = read_csv("ablation_solid_industry_or_eval.csv")
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row["dataset"] != "Industry_OR" or row["run"] not in {"TTRL", "SOLID"}:
            continue
        step = int(row["step"])
        if step <= 150:
            series[row["run"]].append((step, 100.0 * float(row["value"])))

    styles = {
        "TTRL": {"color": "#D55E00", "marker": "s", "linestyle": "--"},
        "SOLID": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    }
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    for method in ("TTRL", "SOLID"):
        values = sorted(series[method])
        ax.plot(
            [step for step, _ in values],
            [value for _, value in values],
            label=method,
            linewidth=2.2,
            markersize=4.8,
            markevery=2,
            **styles[method],
        )
    ax.set_title("IndustryOR validation during training")
    ax.set_xlabel("Training step")
    ax.set_ylabel("mean@4")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlim(0, 150)
    ax.grid()
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "training_curve.png", dpi=220)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_main_results()
    plot_training_curve()


if __name__ == "__main__":
    main()

