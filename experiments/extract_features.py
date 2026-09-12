"""Run every stimulus through a frozen network and cache the visual projection neuron responses.

    uv run python experiments/extract_features.py --dataset fashion-mnist --workers 4
    uv run python experiments/extract_features.py --dataset checkers --network rewired

Features are written in chunks to data/features/<network>/<dataset>/<split>/, so an interrupted run
resumes where it stopped. Each stimulus is seeded by (dataset, split, index), so the baseline
networks see movies identical to the fly's.
"""

import argparse
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from multiprocessing import get_context

import numpy as np
import pandas as pd

from fly_sim.baselines import NETWORKS
from fly_sim.connectome import DATA_DIR
from fly_sim.rate_model import CALIBRATED
from fly_sim import checkers as ck
from fly_sim.stimulus import CHECKERS_POSITIONS, PIECE_V2, Presentation, flash_jitter, load_dataset, moving, render_board
from fly_sim.vision import BOARD, BOARD_V2, Eye

FEATURES_DIR = DATA_DIR / "features"
# The column luminances depend only on the stimulus, not on the network, so they are written once
# and shared. Visual projection rates are stored as float16: the probes standardize every feature
# before fitting, and half precision moves a value head's test R2 by 3e-5.
PIXELS_DIR = FEATURES_DIR / "pixels"
BATCH = 32


@dataclass(frozen=True)
class Dataset:
    """A source of stimuli: `load` gives a renderer of stimulus i and the target for each i."""

    load: Callable[[str], tuple[Callable[[int], np.ndarray], np.ndarray]]
    presentation: Presentation
    make_movie: Callable = flash_jitter


def images_from(name: str):
    def load(split):
        images, labels = load_dataset(name, split)
        return images.__getitem__, labels

    return load


def boards(split: str, piece=None):
    table = pd.read_parquet(CHECKERS_POSITIONS).query("split == @split").reset_index(drop=True)
    positions = [ck.Position(*row) for row in table[["men", "kings", "opp_men", "opp_kings"]].itertuples(index=False, name=None)]
    return (lambda i: render_board(ck.to_array(positions[i]), piece=piece)), table["value"].to_numpy()


# The order fixes the stimulus seeds, so new datasets are appended rather than inserted.
DATASETS = {
    "fashion-mnist": Dataset(images_from("fashion-mnist"), Presentation()),
    "moving-mnist": Dataset(images_from("mnist"), Presentation(size=16.0), moving),
    "checkers": Dataset(boards, BOARD),
    "checkers-v2": Dataset(lambda split: boards(split, PIECE_V2), BOARD_V2),
}
SPLITS = ["train", "test"]

_worker = {}


def stimulus_seed(dataset: str, split: str, index: int) -> list[int]:
    return [list(DATASETS).index(dataset), SPLITS.index(split), index]


def init_worker(dataset, split, network, seed):
    render, targets = DATASETS[dataset].load(split)
    _worker.update(dataset=dataset, split=split, render=render, targets=targets, eye=Eye(network, seed))


def extract_chunk(task):
    start, stop, path = task
    w = _worker
    spec = DATASETS[w["dataset"]]

    vpn, pixels = [], []
    for batch_start in range(start, stop, BATCH):
        indices = range(batch_start, min(batch_start + BATCH, stop))
        seeds = [stimulus_seed(w["dataset"], w["split"], i) for i in indices]
        rates, luminance = w["eye"].look([w["render"](i) for i in indices], seeds, spec.presentation, spec.make_movie)
        vpn.append(rates)
        pixels.append(luminance)

    indices = np.arange(start, stop)
    write(path, vpn=np.concatenate(vpn).astype(np.float16), labels=w["targets"][start:stop], indices=indices)
    shared = PIXELS_DIR / w["dataset"] / w["split"] / path.name
    if not shared.exists():
        shared.parent.mkdir(parents=True, exist_ok=True)
        write(shared, pixels=np.concatenate(pixels).astype(np.float32), indices=indices)
    return path


def write(path, **arrays):
    """Write atomically, so an interrupted run never leaves a half-written chunk behind."""
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def _read(directory, key):
    """Concatenate a directory of chunks into one array in stimulus order, ignoring any overlap."""
    files = sorted(directory.glob("chunk_*.npz"))
    if not files:
        return None, None
    parts = [np.load(f) for f in files]
    indices = np.concatenate([p["indices"] for p in parts])
    values = np.concatenate([p[key] for p in parts])
    order = np.argsort(indices, kind="stable")
    unique, first = np.unique(indices[order], return_index=True)
    return values[order][first], unique


def load_features(network: str, dataset: str, split: str) -> dict:
    """Every stimulus's features, in stimulus order: vpn, pixels and the target.

    Older caches keep a copy of the pixels inside each network's own chunks; newer ones share one
    copy across networks, so both layouts are accepted.
    """
    directory = FEATURES_DIR / network / dataset / split
    vpn, indices = _read(directory, "vpn")
    if vpn is None:
        raise FileNotFoundError(f"no cached features in {directory}")
    labels = _read(directory, "labels")[0]
    pixels = _read(PIXELS_DIR / dataset / split, "pixels")[0]
    if pixels is None or len(pixels) != len(vpn):
        pixels = _read(directory, "pixels")[0]
    return {"vpn": vpn, "pixels": pixels, "labels": labels, "indices": indices}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--network", choices=list(NETWORKS), default="fly")
    parser.add_argument("--seed", type=int, default=0, help="which draw of the baseline network")
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--limit", type=int, help="only the first N stimuli, written to a separate directory")
    args = parser.parse_args()

    # The sparse simulation is single-threaded; keep numpy from oversubscribing cores across workers.
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[var] = "1"

    spec = DATASETS[args.dataset]
    folder = args.network if args.network == "fly" or args.seed == 0 else f"{args.network}{args.seed}"
    for split in SPLITS if args.split == "all" else [args.split]:
        count = len(spec.load(split)[1])
        if args.limit:
            count = min(count, args.limit)
        out = FEATURES_DIR / folder / args.dataset / (f"{split}-first{args.limit}" if args.limit else split)
        out.mkdir(parents=True, exist_ok=True)
        (out / "params.json").write_text(
            json.dumps(
                {"rate": asdict(CALIBRATED), "presentation": asdict(spec.presentation),
                 "network": args.network, "seed": args.seed, "stimuli": count},
                indent=2,
            )
        )

        chunks = [(s, min(s + args.chunk, count), out / f"chunk_{s:06d}_{min(s + args.chunk, count):06d}.npz") for s in range(0, count, args.chunk)]
        todo = [c for c in chunks if not c[2].exists()]
        print(f"{folder}/{args.dataset}/{split}: {count:,} stimuli, {len(chunks) - len(todo)}/{len(chunks)} chunks already done", flush=True)
        if not todo:
            continue

        start, done = time.perf_counter(), 0
        with get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=(args.dataset, split, args.network, args.seed)) as pool:
            for k, task in enumerate(pool.imap_unordered(extract_chunk, todo), start=1):
                chunk = next(c for c in todo if c[2] == task)
                done += chunk[1] - chunk[0]
                elapsed = time.perf_counter() - start
                remaining = sum(c[1] - c[0] for c in todo) - done
                print(
                    f"  {k}/{len(todo)} chunks · {done:,} stimuli · {elapsed / 60:.1f} min · "
                    f"{elapsed / done * 1000:.0f} ms each · ~{remaining * elapsed / done / 60:.0f} min left",
                    flush=True,
                )


if __name__ == "__main__":
    main()
