"""Load the male CNS connectome as a signed sparse weight matrix over traced neurons."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import scipy.sparse as sp

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = DATA_DIR / "processed"

ANNOTATIONS = DATA_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = DATA_DIR / "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = DATA_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

ANNOTATION_COLUMNS = [
    "bodyId",
    "status",
    "type",
    "instance",
    "group",
    "superclass",
    "class",
    "somaSide",
    "somaNeuromere",
    "flywireType",
]

# Sign given to every outgoing synapse of a neuron, keyed by its consensus neurotransmitter.
# GABA and glutamate inhibit, as in Shiu et al. 2024; histamine inhibits too. Modulators have
# no simple sign and get +1 as a placeholder. Unclear or missing predictions get 0, which
# silences that neuron's output rather than guessing.
NT_SIGN = {
    "acetylcholine": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
    "dopamine": 1,
    "octopamine": 1,
    "serotonin": 1,
    "unclear": 0,
}


@dataclass
class Connectome:
    neurons: pd.DataFrame  # row i describes matrix index i
    W: sp.csc_matrix  # W[post, pre] = sign(pre) * synapse count

    def indices(self, **where) -> np.ndarray:
        """Matrix indices of neurons matching every column=value filter, e.g. type="LPLC2"."""
        mask = np.ones(len(self.neurons), dtype=bool)
        for column, value in where.items():
            mask &= (self.neurons[column] == value).to_numpy()
        return np.flatnonzero(mask)


def load(rebuild: bool = False) -> Connectome:
    """Load the cached connectome, building it from the raw tables on first use."""
    neurons_path, weights_path = CACHE_DIR / "neurons.parquet", CACHE_DIR / "W.npz"
    if not rebuild and neurons_path.exists() and weights_path.exists():
        return Connectome(
            pd.read_parquet(neurons_path), sp.load_npz(weights_path).tocsc()
        )

    connectome = build()
    CACHE_DIR.mkdir(exist_ok=True)
    connectome.neurons.to_parquet(neurons_path)
    sp.save_npz(weights_path, connectome.W)
    return connectome


def build() -> Connectome:
    annotations = feather.read_table(
        ANNOTATIONS, columns=ANNOTATION_COLUMNS
    ).to_pandas()
    neurons = (
        annotations[annotations["status"] == "Traced"]
        .drop(columns="status")
        .sort_values("bodyId")
        .reset_index(drop=True)
    )
    nt = feather.read_table(
        NEUROTRANSMITTERS, columns=["body", "consensus_nt"]
    ).to_pandas()
    neurons = neurons.merge(
        nt.rename(columns={"body": "bodyId"}), on="bodyId", how="left"
    )
    neurons["sign"] = neurons["consensus_nt"].map(NT_SIGN).fillna(0).astype(np.int8)

    body_ids = neurons["bodyId"].to_numpy()
    traced = pa.array(body_ids)
    pre, post, count = [], [], []
    total_edges = total_synapses = 0

    # Read batch by batch so the full 152M-row table never sits in memory at once.
    with pa.memory_map(str(WEIGHTS)) as source:
        reader = pa.ipc.open_file(source)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            total_edges += batch.num_rows
            total_synapses += pc.sum(batch.column("weight")).as_py()
            keep = pc.and_(
                pc.is_in(batch.column("body_pre"), value_set=traced),
                pc.is_in(batch.column("body_post"), value_set=traced),
            )
            batch = batch.filter(keep)
            pre.append(np.searchsorted(body_ids, batch.column("body_pre").to_numpy()))
            post.append(np.searchsorted(body_ids, batch.column("body_post").to_numpy()))
            count.append(batch.column("weight").to_numpy())

    pre, post, count = (np.concatenate(parts) for parts in (pre, post, count))
    n = len(neurons)
    sign = neurons["sign"].to_numpy()
    W = sp.csc_matrix((sign[pre] * count, (post, pre)), shape=(n, n), dtype=np.float32)
    W.eliminate_zeros()

    print(f"traced neurons: {n:,}")
    print(
        f"edges kept:     {len(count):,} of {total_edges:,} ({len(count) / total_edges:.1%})"
    )
    print(
        f"synapses kept:  {count.sum():,} of {total_synapses:,} ({count.sum() / total_synapses:.1%})"
    )
    print(f"nonzero W:      {W.nnz:,} (density {W.nnz / n**2:.2e})")
    print("output sign:   ", neurons["sign"].value_counts().sort_index().to_dict())
    return Connectome(neurons, W)
