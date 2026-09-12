"""Train the connectome itself, and ask whether the fly's wiring is a better starting point.

    uv run python experiments/train_connectome.py

Every other experiment here freezes the network and fits a readout. This one trains the 6.1 million
synaptic weights as well, by backpropagation through time, and runs the same recipe from three
starting points -- the real connectome, a degree-preserving rewire of it, and a random network --
so the only thing that differs is where the weights began.

The objective is the one the frozen experiments got wrong. Ranking moves needs the differences
between sibling afterstates, which a value regression never optimizes, so the loss here is pairwise:
the teacher's best move must score above another legal move from the same position.

Writes data/processed/train_connectome.csv and train_connectome.png.
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from checkers_play import positions_table
from fly_sim import checkers as ck
from fly_sim import palette
from fly_sim.baselines import build
from fly_sim.connectome import CACHE_DIR
from fly_sim.optic_lobe import right_eye
from fly_sim.rate_model import CALIBRATED
from fly_sim.stimulus import column_positions, flash_jitter, render_board
from fly_sim.trainable import Adam, TrainableLobe, TrainingParams
from fly_sim.vision import BOARD

NETWORKS = ["fly", "rewired", "random"]
RESULTS = CACHE_DIR / "train_connectome.csv"
FIGURE = CACHE_DIR / "train_connectome.png"


def make_pairs(split: str, count: int, seed: int) -> list[tuple[ck.Position, ck.Position]]:
    """(afterstate of the teacher's move, afterstate of another legal move) from one position."""
    table = positions_table(split)
    table = table[table["best_from"] >= 0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(table))
    pairs = []
    for i in order:
        row = table.iloc[i]
        position = ck.Position(int(row.men), int(row.kings), int(row.opp_men), int(row.opp_kings))
        moves = ck.legal_moves(position)
        if len(moves) < 2:
            continue
        best = next((m for m in moves if m.path[0] == row.best_from and m.path[-1] == row.best_to), None)
        if best is None:
            continue
        others = [m for m in moves if m is not best]
        pairs.append((best.result, others[rng.integers(len(others))].result))
        if len(pairs) >= count:
            break
    return pairs


def make_choices(split: str, count: int, seed: int):
    """(every legal afterstate, index of the teacher's choice) -- the metric that matches play."""
    table = positions_table(split)
    table = table[table["best_from"] >= 0]
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.permutation(len(table)):
        row = table.iloc[i]
        position = ck.Position(int(row.men), int(row.kings), int(row.opp_men), int(row.opp_kings))
        moves = ck.legal_moves(position)
        if len(moves) < 2:
            continue
        best = next((k for k, m in enumerate(moves) if m.path[0] == row.best_from and m.path[-1] == row.best_to), None)
        if best is None:
            continue
        out.append(([m.result for m in moves], best))
        if len(out) >= count:
            break
    return out


class Luminance:
    """Boards rendered onto the eye's columns once, since the stimulus does not depend on the weights."""

    def __init__(self, xy):
        self.xy, self.cache = xy, {}

    def __call__(self, positions):
        missing = [p for p in dict.fromkeys(positions) if p not in self.cache]
        for position in missing:
            movie = flash_jitter(render_board(ck.to_array(position)), self.xy, BOARD,
                                 np.random.default_rng([int(v) for v in position]))
            self.cache[position] = movie[-1]  # the board is static, so the last frame is the board
        return np.stack([self.cache[p] for p in positions])


def pairwise_accuracy(net, luminance, pairs, batch=16):
    right = 0
    for start in range(0, len(pairs), batch):
        block = pairs[start : start + batch]
        boards = [p for pair in block for p in pair]
        scores, _ = net.scores(luminance(boards), keep=False)
        right += int((scores[0::2] > scores[1::2]).sum())
    return right / len(pairs)


def choice_accuracy(net, luminance, choices):
    right = 0
    for afterstates, best in choices:
        scores, _ = net.scores(luminance(afterstates), keep=False)
        right += int(np.argmax(scores) == best)
    return right / len(choices)


def train(network, mode, pairs, validation, choices, luminance, lobe, hexes, t, epochs, batch, seed):
    net = TrainableLobe(build(network, lobe, 0), hexes, CALIBRATED, t, seed=seed)
    trainable = mode == "weights"
    opt_w = Adam(net.values.shape, t.learning_rate) if trainable else None
    opt_r = Adam(net.readout.shape, t.readout_learning_rate)
    opt_b = Adam((), t.readout_learning_rate)

    initial = net.values.copy()
    history, seen, started = [], 0, time.perf_counter()
    for epoch in range(epochs):
        for start in range(0, len(pairs), batch):
            block = pairs[start : start + batch]
            boards = [p for pair in block for p in pair]
            net._refresh()
            scores, saved = net.scores(luminance(boards), keep=trainable)
            better, worse = scores[0::2], scores[1::2]
            margin = worse - better
            # softplus(margin): zero when the teacher's move already scores higher by a wide margin
            probability = 1 / (1 + np.exp(-margin))
            loss = np.mean(np.logaddexp(0, margin))
            dscore = np.zeros_like(scores)
            dscore[0::2] = -probability / len(block)
            dscore[1::2] = probability / len(block)

            grad_readout, grad_bias, grad_values = net.backward(dscore, saved, weights=trainable)
            net.readout += opt_r(grad_readout)
            net.readout_bias += opt_b(np.float32(grad_bias))
            if trainable:
                net.values += opt_w(grad_values)
                net.renormalize()
            seen += len(block)

        net._refresh()
        peak = float(net.scores(luminance([pair[0] for pair in validation[:16]]), keep=True)[1][0][-1].max())
        history.append({
            "peak_rate": peak,
            "network": network, "mode": mode, "epoch": epoch + 1, "pairs_seen": seen, "loss": float(loss),
            "val_pairwise": pairwise_accuracy(net, luminance, validation),
            "val_choice": choice_accuracy(net, luminance, choices),
            "minutes": (time.perf_counter() - started) / 60,
            "weight_change": float(np.abs(net.values - initial).mean() / np.abs(initial).mean()),
        })
        h = history[-1]
        print(f"  {network:8s} {mode:8s} epoch {epoch + 1}: loss {h['loss']:.4f} · "
              f"pairwise {h['val_pairwise']:.1%} · best-move {h['val_choice']:.1%} · {h['minutes']:.0f} min", flush=True)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--validation", type=int, default=600)
    parser.add_argument("--choices", type=int, default=200)
    args = parser.parse_args()

    lobe = right_eye()
    hexes = lobe.neurons.loc[lobe.inputs, ["hex1", "hex2"]].drop_duplicates().to_numpy()
    luminance = Luminance(column_positions(hexes[:, 0], hexes[:, 1]))
    t = TrainingParams(steps=args.steps)

    print("building the training pairs", flush=True)
    pairs = make_pairs("train", args.pairs, seed=0)
    validation = make_pairs("test", args.validation, seed=1)
    choices = make_choices("test", args.choices, seed=2)
    print(f"{len(pairs):,} training pairs · {len(validation):,} validation pairs · {len(choices)} full choices", flush=True)

    rows = []
    for network in NETWORKS:
        for mode in ("readout", "weights"):
            rows += train(network, mode, pairs, validation, choices, luminance, lobe, hexes,
                          t, args.epochs, args.batch, seed=0)
            pd.DataFrame(rows).to_csv(RESULTS, index=False)
    results = pd.DataFrame(rows)
    print("\n" + results.pivot_table(index=["network", "mode"], columns="epoch",
                                     values="val_choice").round(3).to_string())
    plot(results)
    print(f"\nwrote {RESULTS} and {FIGURE}")


def plot(results):
    palette.style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), facecolor=palette.SURFACE)
    for ax, key, title in [(axes[0], "val_pairwise", "Ranking the teacher's move above another"),
                           (axes[1], "val_choice", "Picking the teacher's move out of all of them")]:
        for network in NETWORKS:
            color = palette.SERIES[network][0]
            for mode, style, width in [("weights", "-", 2.0), ("readout", (0, (4, 3)), 1.5)]:
                series = results[(results["network"] == network) & (results["mode"] == mode)]
                if series.empty:
                    continue
                ax.plot(series["epoch"], series[key] * 100, color=color, linewidth=width, linestyle=style,
                        marker="o" if mode == "weights" else None, markersize=6,
                        markeredgecolor=palette.SURFACE, markeredgewidth=1.5,
                        label=f"{palette.SERIES[network][1]} — {'weights trained' if mode == 'weights' else 'readout only'}")
        ax.set_xlabel("Epoch", color=palette.SECONDARY)
        ax.set_title(title, fontsize=10.5, color=palette.PRIMARY, loc="left")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.set_xticks(sorted(results["epoch"].unique()))
        palette.clean(ax)
    axes[0].set_ylabel("Held-out accuracy", color=palette.SECONDARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=3, frameon=False, fontsize=8.5,
               labelcolor=palette.SECONDARY, bbox_to_anchor=(0.006, 0.865))
    fig.suptitle("Training the connectome itself", x=0.006, y=0.972, ha="left",
                 color=palette.PRIMARY, fontsize=13, fontweight="semibold")
    fig.text(0.006, 0.906, "Backpropagation through time over 6.1 million synapses, from three starting points. "
             "Solid: weights trained. Dashed: weights frozen, readout only.",
             color=palette.SECONDARY, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    fig.savefig(FIGURE, dpi=130, facecolor=palette.SURFACE)


if __name__ == "__main__":
    main()
