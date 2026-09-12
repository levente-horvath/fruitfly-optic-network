"""Sweep the rate model's global constants: stability, how deep responses reach, and discriminability.

Uses 30 Fashion-MNIST test images (3 per class), each shown twice with different jitter, so VPN
responses can be compared for the same image (reliability), the same class, and different classes.
Because the model is a ReLU network, scaling bias and input gain together only rescales all rates,
so the bias is fixed at 1 and only recurrent gain and input gain are swept.
"""

import time

import numpy as np
import pandas as pd

from fly_sim.optic_lobe import right_eye
from fly_sim.rate_model import RateModel, RateParams
from fly_sim.stimulus import Presentation, column_positions, flash_jitter, load_dataset

GAINS = [0.5, 1.0, 1.4, 1.6]
INPUT_GAINS = [0.3, 1.0, 3.0, 10.0, 30.0]
PER_CLASS = 3
BATCH = 32
GROUP_PATTERNS = {"lamina in": r"^L[1-3]$", "medulla": r"^(Mi|Tm\d|Dm|Pm)", "T4/T5": r"^T[45][a-d]"}


def correlations(a, b):
    """Correlation between every row of a and every row of b."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    return a @ b.T


def separation(features, labels):
    """Mean correlation between the two jitter repeats: same image, same class, different class."""
    n = len(labels)
    c = correlations(features[:n], features[n:])
    same_class = labels[:, None] == labels[None, :]
    other_image = ~np.eye(n, dtype=bool)
    return np.diag(c).mean(), c[same_class & other_image].mean(), c[~same_class].mean()


def main():
    lobe = right_eye()
    column_hex = lobe.neurons.loc[lobe.inputs, ["hex1", "hex2"]].drop_duplicates().to_numpy()
    xy = column_positions(column_hex[:, 0], column_hex[:, 1])
    model = RateModel(lobe, column_hex)

    images, labels = load_dataset("fashion-mnist", "test")
    chosen = np.concatenate([np.flatnonzero(labels == k)[:PER_CLASS] for k in range(10)])
    labels = labels[chosen]
    stim = Presentation()
    movies = np.stack(
        [flash_jitter(images[i], xy, stim, np.random.default_rng(seed)) for seed in (1, 2) for i in chosen]
    )
    window = (round(stim.blank / stim.dt), movies.shape[1])

    types = lobe.neurons["type"].fillna("")
    groups = {name: np.flatnonzero(types.str.match(pattern).to_numpy()) for name, pattern in GROUP_PATTERNS.items()}
    groups["VPN out"] = lobe.outputs

    pixels = movies[:, window[0] :].mean(axis=1)
    same, within, between = separation(pixels, labels)
    print(f"raw hex pixels: same image {same:.3f} · same class {within:.3f} · other class {between:.3f}\n")

    rows = []
    for gain in GAINS:
        for input_gain in INPUT_GAINS:
            p = RateParams(gain=gain, input_gain=input_gain)
            start = time.perf_counter()
            r0, steps = model.baseline(p)
            parts = [model.run(movies[i : i + BATCH], stim.dt, window, p) for i in range(0, len(movies), BATCH)]
            mean = np.concatenate([m for m, _ in parts])
            peak = max(pk.max() for _, pk in parts)
            delta = mean - r0
            same, within, between = separation(delta[:, lobe.outputs], labels)
            rows.append(
                {
                    "gain": gain,
                    "input": input_gain,
                    "settle steps": steps,
                    "baseline mean": r0.mean(),
                    "active at rest": (r0 > 1e-3).mean(),
                    "peak": peak,
                    **{f"|Δ| {name}": np.abs(delta[:, idx]).mean() for name, idx in groups.items()},
                    "VPN responding": (np.abs(delta[:, lobe.outputs]).mean(axis=0) > 0.01).mean(),
                    "same image": same,
                    "same class": within,
                    "other class": between,
                    "seconds": time.perf_counter() - start,
                }
            )
            print(f"gain {gain} · input {input_gain}: done in {rows[-1]['seconds']:.0f}s", flush=True)

    table = pd.DataFrame(rows)
    table["class gap"] = table["same class"] - table["other class"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n" + table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
