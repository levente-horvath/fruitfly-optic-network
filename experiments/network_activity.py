"""Record the right-eye optic lobe responding to a few images and build an interactive 3D viewer.

Each neuron of our network is shown two ways: at its cell body inside the fly's whole central nervous
system, and in a layered view (eye column × synapses from the image). Neurons are colored by how their
firing rate changes as the image appears, next to a linear classifier reading the visual projection
neurons as the response builds up.

Run experiments/extract_features.py first; the classifiers are trained on its cached features.
Writes data/processed/network_activity.html.
"""

import base64
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from evaluate_probes import fit_ridge
from evaluate_probes import load as load_features
from extract_features import DATASETS, PARAMS, stimulus_seed
from fly_sim.connectome import ANNOTATIONS, CACHE_DIR, load
from fly_sim.optic_lobe import right_eye
from fly_sim.rate_model import RateModel
from fly_sim.stimulus import column_positions, load_dataset

TEMPLATE = Path(__file__).resolve().parents[1] / "viz" / "network_activity.html"
OUTPUT = CACHE_DIR / "network_activity.html"
VOXEL_UM = 0.008

FASHION_CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
CLASS_NAMES = {"fashion-mnist": FASHION_CLASSES, "moving-mnist": [str(d) for d in range(10)]}
EXAMPLES = [("fashion-mnist", 9), ("fashion-mnist", 1), ("fashion-mnist", 8), ("moving-mnist", 3)]  # (dataset, class)
FIRST_UNSEEN_TEST = {"fashion-mnist": 500, "moving-mnist": 0}  # Fashion-MNIST test 0-499 were used to calibrate
CLASSIFIER_TRAIN = 10_000

LAYER_SPACING = 12  # column spacings between depth planes
TYPE_PLANES = 8  # cell types in a depth are spread over this many thin sub-planes
LAYER_SCALE = 22.0  # world units per column spacing, so both views are about the same size
DEPTH_NAMES = ["Lamina inputs L1–L3", "1 synapse from the image", "2+ synapses from the image", "Outputs: visual projection neurons"]


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def synapse_depth(lobe) -> np.ndarray:
    """0 for the lamina inputs, 1 or 2 by shortest path from them, 3 for the output neurons."""
    A = (lobe.W != 0).astype(np.float32).tocsr()
    depth = np.full(len(lobe.neurons), 2, dtype=np.uint8)
    reached = np.zeros(len(lobe.neurons), dtype=bool)
    reached[lobe.inputs] = True
    depth[lobe.inputs] = 0
    frontier = reached.astype(np.float32)
    for hop in (1, 2):
        new = (A @ frontier > 0) & ~reached
        depth[new] = hop
        reached |= new
        frontier = new.astype(np.float32)
    depth[lobe.outputs] = 3
    return depth


def column_layout(lobe) -> tuple[np.ndarray, int]:
    """Eye-plane position of every neuron: its column if assigned, else the synapse-weighted mean of its partners'."""
    neurons = lobe.neurons
    has_hex = neurons["hex1"].notna().to_numpy()
    xy = np.zeros((len(neurons), 2))
    xy[has_hex] = column_positions(neurons.loc[has_hex, "hex1"].to_numpy(), neurons.loc[has_hex, "hex2"].to_numpy())
    A = abs(lobe.W)
    partners = (A + A.T).tocsc()
    known = has_hex.copy()
    while True:
        weights = partners[:, known]
        total = np.asarray(weights.sum(axis=1)).ravel()
        new = ~known & (total > 0)
        if not new.any():
            break
        xy[new] = ((weights @ xy[known]) / np.maximum(total, 1e-9)[:, None])[new]
        known |= new
    return xy, int(has_hex.sum())


def main():
    connectome = load()
    lobe = right_eye(connectome)
    neurons = lobe.neurons
    column_hex = neurons.loc[lobe.inputs, ["hex1", "hex2"]].drop_duplicates().to_numpy()
    eye_xy = column_positions(column_hex[:, 0], column_hex[:, 1])

    # Layered view: eye plane × depth, each cell type on its own thin sub-plane.
    depth = synapse_depth(lobe)
    xy, assigned = column_layout(lobe)
    type_plane = (np.unique(neurons["type"].fillna(""), return_inverse=True)[1] % TYPE_PLANES).astype(np.float64)
    z = depth * LAYER_SPACING + type_plane * (LAYER_SPACING * 0.5 / TYPE_PLANES)
    jitter = np.random.default_rng(0).uniform(-0.2, 0.2, size=xy.shape)
    layer = np.column_stack([xy + jitter, z.mean() - z]) * LAYER_SCALE  # lamina inputs nearest the camera

    # Brain view: cell body positions in the whole CNS, body axis vertical.
    somas = feather.read_table(ANNOTATIONS, columns=["bodyId", "somaLocation", "tosomaLocation"]).to_pandas()
    cns = connectome.neurons[["bodyId"]].merge(somas, on="bodyId", how="left")
    location = cns["somaLocation"].where(cns["somaLocation"].notna(), cns["tosomaLocation"])
    shown = location.notna().to_numpy()
    raw = np.stack(location[shown].to_numpy()).astype(np.float64) * VOXEL_UM
    raw -= (raw.min(axis=0) + raw.max(axis=0)) / 2
    brain_all = np.full((len(cns), 3), np.nan)
    brain_all[shown] = np.column_stack([raw[:, 0], -raw[:, 2], raw[:, 1]])
    cns_index = neurons["cns_index"].to_numpy()
    in_network = np.zeros(len(cns), dtype=bool)
    in_network[cns_index] = True

    model = RateModel(lobe, column_hex)
    resting = model.baseline(PARAMS)[0]
    fits, recorded = {}, []
    for dataset, label in EXAMPLES:
        source, make_movie, stim = DATASETS[dataset]
        if dataset not in fits:
            train = load_features(dataset, "train")
            chosen = np.sort(np.random.default_rng(0).choice(len(train["labels"]), CLASSIFIER_TRAIN, replace=False))
            fits[dataset] = fit_ridge(train["vpn"][chosen].astype(np.float64), train["labels"][chosen])
            print(f"{dataset} classifier: leave-one-out accuracy {fits[dataset].loo_accuracy:.1%} on {CLASSIFIER_TRAIN:,} images")

        images, labels = load_dataset(source, "test")
        start = FIRST_UNSEEN_TEST[dataset]
        index = int(start + np.flatnonzero(labels[start:] == label)[0])
        movie = make_movie(images[index], eye_xy, stim, np.random.default_rng(stimulus_seed(dataset, "test", index)))
        delta = model.trajectory(movie[None], stim.dt, PARAMS)[0] - resting

        blank = round(stim.blank / stim.dt)
        running = np.cumsum(delta[blank:, lobe.outputs], axis=0) / np.arange(1, len(delta) - blank + 1)[:, None]
        scores = np.zeros((len(delta), 10), dtype=np.float32)
        scores[blank:] = fits[dataset].scores(running.astype(np.float64))
        recorded.append((dataset, label, index, images[index], movie, delta, scores, blank, stim.dt))
        name = CLASS_NAMES[dataset][label]
        print(f"  {dataset} test #{index} ({name}): predicted {CLASS_NAMES[dataset][scores[-1].argmax()]} at {len(delta) * stim.dt * 1000:.0f} ms")

    # Compress rate changes to int8 on a signed square-root scale, so small changes stay visible.
    scale = float(np.percentile(np.abs(np.concatenate([r[5].ravel() for r in recorded])), 99.9))
    payload = {
        "network": {
            "count": len(neurons),
            "layer": b64(layer.astype(np.float32)),
            "brain": b64(brain_all[cns_index].astype(np.float32)),
            "type": b64((np.unique(neurons["type"].fillna(""), return_inverse=True)[1]).astype(np.uint16)),
            "typeNames": list(np.unique(neurons["type"].fillna(""))),
            "superclass": list(neurons["superclass"]),
            "depth": b64(depth),
            "depthNames": DEPTH_NAMES,
            "bodyIds": b64(neurons["bodyId"].to_numpy(np.float64)),
        },
        "context": {"brain": b64(brain_all[shown & ~in_network].astype(np.float32))},
        "columns": b64(eye_xy.astype(np.float32)),
        "deltaScale": scale,
        "classNames": CLASS_NAMES,
        "examples": [
            {
                "name": CLASS_NAMES[dataset][label],
                "dataset": dataset,
                "label": label,
                "testIndex": index,
                "frames": len(delta),
                "blankFrames": blank,
                "frameMs": dt * 1000,
                "image": b64(np.clip(image * 255, 0, 255).round().astype(np.uint8)),
                "stimulus": b64(np.clip(movie * 255, 0, 255).round().astype(np.uint8)),
                "activity": b64((np.sign(delta) * np.sqrt(np.minimum(np.abs(delta) / scale, 1)) * 127).round().astype(np.int8)),
                "scores": b64(scores),
            }
            for dataset, label, index, image, movie, delta, scores, blank, dt in recorded
        ],
    }
    OUTPUT.write_text(TEMPLATE.read_text().replace("__PAYLOAD__", json.dumps(payload)))

    counts = np.bincount(depth, minlength=4)
    print(f"network: {len(neurons):,} neurons · depth planes {dict(zip(DEPTH_NAMES, counts.tolist()))}")
    print(f"layer view: {assigned:,} neurons with an eye column, the rest placed by synaptic partners")
    print(f"brain view: {int(np.isfinite(brain_all[cns_index, 0]).sum()):,} network neurons with a cell body position, "
          f"{int((shown & ~in_network).sum()):,} other CNS neurons as context")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
