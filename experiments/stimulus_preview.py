"""Render example stimuli on the right eye's hex lattice, to check the image-to-column mapping.

Writes data/processed/stimulus_preview.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from fly_sim.connectome import CACHE_DIR
from fly_sim.optic_lobe import right_eye
from fly_sim.stimulus import Presentation, column_positions, flash_jitter, load_dataset, moving

OUTPUT = CACHE_DIR / "stimulus_preview.png"
FASHION_CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def draw_hexes(ax, xy, values):
    angles = np.deg2rad(90 + 60 * np.arange(6))  # pointy-top hexagons tile rows along x
    corners = np.column_stack([np.cos(angles), np.sin(angles)]) / np.sqrt(3)
    hexes = PolyCollection(xy[:, None, :] + corners, array=values, cmap="gray", clim=(0, 1), edgecolors="face")
    ax.add_collection(hexes)
    ax.set_xlim(xy[:, 0].min() - 1, xy[:, 0].max() + 1)
    ax.set_ylim(xy[:, 1].min() - 1, xy[:, 1].max() + 1)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_row(axes, image, title, frames, times, p):
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(title, fontsize=9)
    axes[0].axis("off")
    for ax, t in zip(axes[1:], times):
        draw_hexes(ax, xy, frames[round(t / p.dt)])
        ax.set_title(f"{t * 1000:.0f} ms", fontsize=9)


lobe = right_eye()
columns = lobe.neurons.loc[lobe.inputs, ["hex1", "hex2"]].drop_duplicates()
xy = column_positions(columns["hex1"].to_numpy(), columns["hex2"].to_numpy())
rng = np.random.default_rng(0)

fashion, fashion_labels = load_dataset("fashion-mnist", "test")
digits, digit_labels = load_dataset("mnist", "test")
flash = Presentation()
drift = Presentation(size=16.0)
times = [0.05, 0.1, 0.16, 0.26, 0.38]

fig, axes = plt.subplots(3, 6, figsize=(13, 6.8))
for row, i in zip(axes[:2], [0, 1]):
    frames = flash_jitter(fashion[i], xy, flash, rng)
    draw_row(row, fashion[i], f"Fashion-MNIST: {FASHION_CLASSES[fashion_labels[i]]}", frames, times, flash)
draw_row(axes[2], digits[0], f"Moving MNIST: {digit_labels[0]}", moving(digits[0], xy, drift, rng), times, drift)
fig.suptitle(f"Stimuli on the right eye ({len(xy)} columns) · flash + jitter (rows 1-2) and drifting digit (row 3)")
fig.tight_layout()
fig.savefig(OUTPUT, dpi=110)
print(f"wrote {OUTPUT}")
