"""Can the eye read the board at all? A probe that reconstructs each square from the features.

    uv run python experiments/checkers_readability.py

For every one of the 32 playable squares, a linear readout has to name its contents: empty, or a
man or king of either side. This is the precondition for everything else -- a value head cannot
weigh a position the network cannot see -- and it says where on the board the eye is sharpest.

Writes data/processed/checkers_readability.csv and checkers_readability.png.
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from checkers_play import CLASSES, PLAYERS, load_features, positions_table, square_contents
from fly_sim import checkers as ck
from fly_sim import palette, probe
from fly_sim.connectome import CACHE_DIR

TRAIN_SIZE = 10000

RESULTS = CACHE_DIR / "checkers_readability.csv"
FIGURE = CACHE_DIR / "checkers_readability.png"


def square_accuracy(prediction: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-square accuracy from a (rows, 32 * 5) score matrix."""
    chosen = prediction.reshape(len(prediction), ck.SQUARES, len(CLASSES)).argmax(axis=2)
    return (chosen == labels).mean(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    args = parser.parse_args()
    palette.style(plt)

    train_labels, test_labels = (square_contents(positions_table(split)) for split in ("train", "test"))
    rows, boards = [], {}
    for name, (network, key) in PLAYERS.items():
        start = time.perf_counter()
        train, test = load_features(network, "train"), load_features(network, "test")
        chosen = np.sort(np.random.default_rng(0).choice(len(train_labels), args.train_size, replace=False))
        y = np.eye(len(CLASSES))[train_labels[chosen]].reshape(args.train_size, -1)
        fit = probe.fit(train[key][chosen].astype(np.float64), y, score=lambda p: square_accuracy(p, train_labels[chosen]).mean())

        prediction = fit.predict(test[key].astype(np.float64))
        accuracy = square_accuracy(prediction, test_labels)
        boards[name] = accuracy
        guess = prediction.reshape(len(prediction), ck.SQUARES, len(CLASSES)).argmax(axis=2)
        rows.append({
            "readout": name,
            "accuracy": accuracy.mean(),
            "worst_square": accuracy.min(),
            "always_empty": (test_labels == 0).mean(),
            **{f"recall_{c.replace(' ', '_')}": float((guess[test_labels == i] == i).mean()) for i, c in enumerate(CLASSES)},
        })
        print(f"{name:8s} square accuracy {accuracy.mean():.1%} (worst square {accuracy.min():.1%}) · "
              f"ridge {fit.scale:.0e} · {time.perf_counter() - start:.0f}s", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS, index=False)
    print("\n" + results.round(3).to_string(index=False))
    print(f"\nguessing 'empty' everywhere would score {rows[0]['always_empty']:.1%}")
    plot(boards, results)
    print(f"wrote {RESULTS} and {FIGURE}")


def plot(boards, results):
    ramp = LinearSegmentedColormap.from_list("blue", palette.BLUE_RAMP)
    fig, axes = plt.subplots(1, len(boards), figsize=(3.05 * len(boards), 3.9), facecolor=palette.SURFACE)

    for ax, (name, accuracy) in zip(axes, boards.items()):
        grid = np.full((8, 8), np.nan)
        for square in range(ck.SQUARES):
            grid[ck.square_rowcol(square)] = accuracy[square]
        ax.imshow(grid, cmap=ramp, vmin=0.90, vmax=1.0)  # the spread is all in the last ten points
        for square in range(ck.SQUARES):
            row, col = ck.square_rowcol(square)
            ax.text(col, row, f"{accuracy[square] * 100:.0f}", ha="center", va="center", fontsize=7.5,
                    color=palette.SURFACE if accuracy[square] > 0.97 else palette.PRIMARY)
        ax.set_xticks([]), ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        overall = results.set_index("readout").loc[name, "accuracy"]
        ax.set_title(f"{palette.SERIES[name][1]}\n{overall:.1%} of squares read correctly", fontsize=9.5,
                     color=palette.PRIMARY, loc="left", pad=8)

    fig.suptitle("Reading the board off the features", x=0.012, ha="left", color=palette.PRIMARY, fontsize=13, fontweight="semibold")
    fig.text(0.012, 0.9, "Linear readout naming each playable square's contents: empty, man or king of either side. "
             f"Test positions; {results['always_empty'].iloc[0]:.0%} would be right by always saying \"empty\". "
             "The side to move sits at the bottom.", color=palette.SECONDARY, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(FIGURE, dpi=130, facecolor=palette.SURFACE)


if __name__ == "__main__":
    main()
