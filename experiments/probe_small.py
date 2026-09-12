"""Go/no-go check: a small linear probe on VPN responses versus raw hex pixels, on Fashion-MNIST.

    uv run python experiments/probe_small.py --gain 1.0 --input-gain 3.0
"""

import argparse
import time

import numpy as np

from fly_sim.optic_lobe import right_eye
from fly_sim.rate_model import RateModel, RateParams
from fly_sim.stimulus import Presentation, column_positions, flash_jitter, load_dataset

BATCH = 32
RIDGE_SCALES = [1e-3, 1e-2, 1e-1, 1.0, 10.0]


def ridge_accuracy(train_x, train_y, test_x, test_y):
    """Ridge-regression classifier on standardized features; ridge strength picked on a validation split."""
    mean, std = train_x.mean(axis=0), train_x.std(axis=0)
    std[std == 0] = 1

    def fit_predict(x, y, query, scale):
        x, query = (x - mean) / std, (query - mean) / std
        K = x @ x.T
        ridge = scale * np.trace(K) / len(K)
        alpha = np.linalg.solve(K + ridge * np.eye(len(K)), np.eye(10)[y])
        return (query @ x.T @ alpha).argmax(axis=1)

    split = int(0.8 * len(train_x))
    scores = [(fit_predict(train_x[:split], train_y[:split], train_x[split:], s) == train_y[split:]).mean() for s in RIDGE_SCALES]
    best = RIDGE_SCALES[int(np.argmax(scores))]
    return (fit_predict(train_x, train_y, test_x, best) == test_y).mean(), best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--input-gain", type=float, required=True)
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--test", type=int, default=500)
    args = parser.parse_args()

    lobe = right_eye()
    column_hex = lobe.neurons.loc[lobe.inputs, ["hex1", "hex2"]].drop_duplicates().to_numpy()
    xy = column_positions(column_hex[:, 0], column_hex[:, 1])
    model = RateModel(lobe, column_hex)
    p = RateParams(gain=args.gain, input_gain=args.input_gain)
    stim = Presentation()
    window = (round(stim.blank / stim.dt), round((stim.blank + stim.duration) / stim.dt))
    r0, _ = model.baseline(p)

    def features(images, seed):
        rng = np.random.default_rng(seed)
        vpn, pixels = [], []
        for i in range(0, len(images), BATCH):
            movies = np.stack([flash_jitter(image, xy, stim, rng) for image in images[i : i + BATCH]])
            mean, _ = model.run(movies, stim.dt, window, p)
            vpn.append(mean[:, lobe.outputs] - r0[lobe.outputs])
            pixels.append(movies[:, window[0] :].mean(axis=1))
        return np.concatenate(vpn).astype(np.float64), np.concatenate(pixels).astype(np.float64)

    start = time.perf_counter()
    train_images, train_labels = load_dataset("fashion-mnist", "train")
    test_images, test_labels = load_dataset("fashion-mnist", "test")
    train_vpn, train_pixels = features(train_images[: args.train], seed=0)
    test_vpn, test_pixels = features(test_images[: args.test], seed=1)
    elapsed = time.perf_counter() - start
    print(f"simulated {args.train + args.test} images in {elapsed:.0f}s ({elapsed / (args.train + args.test) * 1000:.0f} ms/image)")

    train_y, test_y = train_labels[: args.train], test_labels[: args.test]
    for name, train_x, test_x in [("raw hex pixels", train_pixels, test_pixels), ("VPN responses", train_vpn, test_vpn)]:
        accuracy, ridge = ridge_accuracy(train_x, train_y, test_x, test_y)
        print(f"{name:15s} {train_x.shape[1]:5d} features · test accuracy {accuracy:.1%} (ridge scale {ridge})")


if __name__ == "__main__":
    main()
