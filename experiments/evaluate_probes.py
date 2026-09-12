"""Step 5: linear probes on the cached fly optic lobe features versus raw hex pixels, with learning curves.

Ridge-regression classifiers on standardized features. The ridge strength is chosen by exact
leave-one-out accuracy on the training set, computed from one eigendecomposition per fit.
Fashion-MNIST is tested on test images 500-9,999 because the first 500 were used for calibration.

Writes data/processed/probe_results.csv and data/processed/probe_learning_curves.png.
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import extract_features
from fly_sim import probe
from fly_sim.connectome import CACHE_DIR

DATASETS = {  # name -> (first test index used for evaluation, panel title)
    "fashion-mnist": (500, "Fashion-MNIST · flash + jitter"),
    "moving-mnist": (0, "Moving MNIST · drifting digit"),
}
FEATURES = {"fly": "vpn", "pixels": "pixels"}
TRAIN_SIZES = [100, 300, 1000, 3000, 10000, 60000]
SEEDS = 3

RESULTS = CACHE_DIR / "probe_results.csv"
FIGURE = CACHE_DIR / "probe_learning_curves.png"

# Reference palette (dataviz skill), light mode: categorical slots 1-2, chrome and ink.
SERIES = {"fly": ("#2a78d6", "Fly optic lobe output (4,611 VPNs)"), "pixels": ("#eb6834", "Raw hex pixels (892 columns)")}
SURFACE, GRID, AXIS, MUTED, SECONDARY, PRIMARY = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#898781", "#52514e", "#0b0b0b"


def load(dataset, split):
    return extract_features.load_features("fly", dataset, split)


def ridge_probe(x, y, x_test, y_test):
    """Return test accuracy, leave-one-out accuracy and the chosen ridge scale."""
    fit = probe.fit(x, np.eye(10)[y], score=probe.accuracy(y))
    return (fit.predict(x_test).argmax(axis=1) == y_test).mean(), fit.loo_score, fit.scale


def run_probes():
    rows = []
    for dataset, (first_test, _) in DATASETS.items():
        train, test = load(dataset, "train"), load(dataset, "test")
        for feature, key in FEATURES.items():
            x_test = test[key][first_test:].astype(np.float64)
            y_test = test["labels"][first_test:]
            for size in TRAIN_SIZES:
                for seed in range(SEEDS if size < len(train["labels"]) else 1):
                    chosen = np.sort(np.random.default_rng(seed).choice(len(train["labels"]), size, replace=False))
                    start = time.perf_counter()
                    accuracy, loo, scale = ridge_probe(train[key][chosen].astype(np.float64), train["labels"][chosen], x_test, y_test)
                    rows.append(
                        {"dataset": dataset, "features": feature, "train_size": size, "seed": seed,
                         "test_accuracy": accuracy, "loo_accuracy": loo, "ridge_scale": scale, "test_images": len(y_test)}
                    )
                    print(f"{dataset:13s} {feature:6s} n={size:>6,} seed {seed}: test {accuracy:.1%} · loo {loo:.1%} · ridge {scale:.0e} · {time.perf_counter() - start:.0f}s", flush=True)
    return pd.DataFrame(rows)


def plot(results):
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10})
    summary = results.groupby(["dataset", "features", "train_size"])["test_accuracy"].agg(["mean", "std"]).fillna(0) * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)

    for ax, (dataset, (_, title)) in zip(axes, DATASETS.items()):
        ax.set_facecolor(SURFACE)
        ends = {}
        for feature, (color, label) in SERIES.items():
            s = summary.loc[(dataset, feature)]
            ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=color, alpha=0.1, linewidth=0)
            ax.plot(s.index, s["mean"], color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                    marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
            ends[feature] = s["mean"].iloc[-1]
        # Direct end labels only when they cannot collide; otherwise the legend and table carry the values.
        if abs(ends["fly"] - ends["pixels"]) > 2.5:
            for value in ends.values():
                ax.annotate(f"{value:.1f}%", (TRAIN_SIZES[-1], value), xytext=(8, 0), textcoords="offset points",
                            va="center", color=SECONDARY, fontsize=9)

        ax.set_xscale("log")
        ax.set_xticks(TRAIN_SIZES, [f"{n:,}" for n in TRAIN_SIZES])
        ax.minorticks_off()
        ax.set_xlim(TRAIN_SIZES[0] * 0.8, TRAIN_SIZES[-1] * 2.2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, length=0)
        ax.set_xlabel("Training images (log scale)", color=SECONDARY)
        ax.set_title(title, color=PRIMARY, loc="left", fontsize=11, fontweight="semibold")
    axes[0].set_ylabel("Test accuracy", color=SECONDARY)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, labelcolor=SECONDARY, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Linear probe accuracy", x=0.01, ha="left", color=PRIMARY, fontsize=13, fontweight="semibold")
    fig.text(0.01, 0.905, "Ridge classifier · mean ± std over 3 training subsets (1 at 60,000)", color=SECONDARY, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGURE, dpi=130, facecolor=SURFACE)


def main():
    results = run_probes()
    results.to_csv(RESULTS, index=False)
    table = results.pivot_table(index=["dataset", "train_size"], columns="features", values="test_accuracy", aggfunc=["mean", "std"]) * 100
    paired = results.pivot_table(index=["dataset", "train_size", "seed"], columns="features", values="test_accuracy")
    table[("fly − pixels", "mean")] = (paired["fly"] - paired["pixels"]).groupby(level=[0, 1]).mean() * 100
    print("\n" + table.round(1).to_string())
    plot(results)
    print(f"\nwrote {RESULTS} and {FIGURE}")


if __name__ == "__main__":
    main()
