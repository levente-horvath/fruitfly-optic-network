"""The right-eye optic lobe as a standalone network.

Images enter at the lamina (L1-L3) and features are read from the visual projection neurons.
Photoreceptors, central-brain feedback and connections from the left eye are left out.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import scipy.sparse as sp

from fly_sim.connectome import ANNOTATIONS, CACHE_DIR, Connectome, load

# Photoreceptors (ol_sensory) are excluded because images enter at the lamina, and R1-R6 are only
# ~13% traced. Centrifugal neurons are excluded because they carry central-brain feedback.
NETWORK_SUPERCLASSES = ["ol_intrinsic", "visual_projection"]
# Cutting the right eye out needs the whole 165k-neuron connectome in memory, about 0.9 GB. The
# result is 50 MB, so it is cached: workers that only want the subnetwork never pay for the rest.
# Delete these two files to rebuild.
CACHED_W = CACHE_DIR / "right_eye_W.npz"
CACHED_NEURONS = CACHE_DIR / "right_eye_neurons.parquet"
INPUT_TYPES = ["L1", "L2", "L3"]
OUTPUT_SUPERCLASS = "visual_projection"


@dataclass
class OpticLobe:
    neurons: pd.DataFrame  # row i describes matrix index i; cns_index maps back to the full connectome
    W: sp.csc_matrix  # W[post, pre] = sign(pre) * synapse count, within this eye only
    inputs: np.ndarray  # lamina neurons that receive the image
    outputs: np.ndarray  # visual projection neurons that are read out


def right_eye(connectome: Connectome | None = None, cache: bool = True) -> OpticLobe:
    supplied = connectome is not None
    if cache and not supplied and CACHED_W.exists() and CACHED_NEURONS.exists():
        return _assemble(pd.read_parquet(CACHED_NEURONS), sp.load_npz(CACHED_W).tocsc())

    connectome = connectome or load()
    neurons, eye = assign_eyes(connectome)
    keep = neurons["superclass"].isin(NETWORK_SUPERCLASSES).to_numpy() & (eye == "R")
    index = np.flatnonzero(keep)

    sub = (
        neurons.iloc[index]
        .rename(columns={"assignedOlHex1": "hex1", "assignedOlHex2": "hex2"})
        .assign(cns_index=index)
        .reset_index(drop=True)
    )
    W = connectome.W.tocsr()[index].tocsc()[:, index]
    if cache and not supplied:
        CACHED_W.parent.mkdir(parents=True, exist_ok=True)
        sp.save_npz(CACHED_W, W.tocoo())
        sub.to_parquet(CACHED_NEURONS, index=False)
    return _assemble(sub, W)


def _assemble(neurons: pd.DataFrame, W: sp.csc_matrix) -> OpticLobe:
    """Attach the input and output sets, which follow from the cell types."""
    inputs = np.flatnonzero(neurons["type"].isin(INPUT_TYPES).to_numpy())
    outputs = np.flatnonzero((neurons["superclass"] == OUTPUT_SUPERCLASS).to_numpy())
    return OpticLobe(neurons, W, inputs, outputs)


def assign_eyes(connectome: Connectome, passes: int = 2) -> tuple[pd.DataFrame, np.ndarray]:
    """Return all neurons with annotations attached, and each optic lobe neuron's eye ("R", "L" or "").

    Annotated sides follow the soma, but some types (CT1, Li39, MeVC1, ...) have their soma near
    the midline and arborize entirely in the opposite optic lobe. So each neuron is assigned to the
    eye that supplies most of its input synapses, or failing that, receives most of its output.
    """
    annotations = feather.read_table(
        ANNOTATIONS, columns=["bodyId", "rootSide", "assignedOlHex1", "assignedOlHex2"]
    ).to_pandas()
    neurons = connectome.neurons.merge(annotations, on="bodyId", how="left")
    suffix = neurons["instance"].str.extract(r"_([LR])$")[0]
    annotated = suffix.fillna(neurons["somaSide"]).fillna(neurons["rootSide"]).to_numpy(dtype=object)
    neurons["annotated_side"] = annotated

    optic = neurons["superclass"].fillna("").str.match(r"^(ol_|visual)").to_numpy()
    A = abs(connectome.W).tocsr()  # rows receive, columns send
    side = np.select([annotated == "R", annotated == "L"], [1, -1], 0) * optic
    for _ in range(passes):
        right, left = (side == 1).astype(np.float32), (side == -1).astype(np.float32)
        by_input = np.sign(A @ right - A @ left)
        by_output = np.sign(A.T @ right - A.T @ left)
        side = np.where(by_input != 0, by_input, np.where(by_output != 0, by_output, side)) * optic
    eye = np.select([side == 1, side == -1], ["R", "L"], "")
    return neurons, eye


def main():
    connectome = load()
    lobe = right_eye(connectome)
    n, W = lobe.neurons, lobe.W
    print(f"right-eye network: {len(n):,} neurons · {W.nnz:,} edges · {abs(W).sum():,.0f} synapses")
    print("  by superclass:", n["superclass"].value_counts().to_dict())
    print(f"  excitatory/inhibitory edges: {(W.data > 0).sum():,} / {(W.data < 0).sum():,}")

    all_neurons, eye = assign_eyes(connectome)
    moved_in = n[n["annotated_side"] != "R"]
    optic_right_label = (all_neurons["annotated_side"] == "R").to_numpy() & (eye == "L")
    print("\neye assignment vs annotated side:")
    print(f"  annotated L/unsided but innervates the right eye: {len(moved_in)}", moved_in["type"].value_counts().to_dict())
    print(f"  annotated R but innervates the left eye: {optic_right_label.sum()}", all_neurons.loc[optic_right_label, "type"].value_counts().to_dict())

    print("\ninputs:")
    for cell_type, group in n.iloc[lobe.inputs].groupby("type"):
        columns = group[["hex1", "hex2"]].dropna().drop_duplicates()
        print(f"  {cell_type}: {len(group)} neurons in {len(columns)} columns, {group['hex1'].isna().sum()} without a column")

    outputs = n.iloc[lobe.outputs]
    families = outputs["type"].str.extract(r"^([A-Za-z]+)")[0].value_counts()
    print(f"\noutputs: {len(outputs):,} visual projection neurons in {outputs['type'].nunique()} types")
    print("  families:", families.head(10).to_dict())

    # Which input synapses the cuts removed, by the kind of neuron that sent them.
    index = n["cns_index"].to_numpy()
    superclass = all_neurons["superclass"].fillna("")
    in_network = np.zeros(len(all_neurons), dtype=bool)
    in_network[index] = True
    sender = np.select(
        [in_network, superclass == "ol_sensory", superclass == "visual_centrifugal", eye == "L", eye == "R"],
        ["kept (right-eye network)", "photoreceptors", "centrifugal (brain feedback)", "left eye", "other right optic lobe"],
        "central brain / nerve cord",
    )
    incoming = abs(connectome.W).tocsr()[index]
    total = incoming.sum()
    print(f"\ninput synapses onto the network: {total:,.0f}")
    for category in pd.unique(sender):
        synapses = (incoming @ (sender == category).astype(np.float32)).sum()
        print(f"  {category:30s} {synapses:>12,.0f}  ({synapses / total:.1%})")

    kept_in = abs(W).sum(axis=1).A1
    total_in = incoming.sum(axis=1).A1
    lost = pd.Series(1 - kept_in / np.maximum(total_in, 1)).groupby(n["type"]).agg(["mean", "size"])
    print("\ntypes losing the most input (>= 20 neurons):")
    print(lost[lost["size"] >= 20].sort_values("mean", ascending=False).head(10).round(2).to_string())

    no_input = (kept_in == 0) & ~np.isin(np.arange(len(n)), lobe.inputs)
    no_output = (abs(W).sum(axis=0).A1 == 0) & ~np.isin(np.arange(len(n)), lobe.outputs)
    print(f"\nneurons with no inputs left (excluding L1-L3): {no_input.sum()}", n.loc[no_input, "type"].value_counts().head(8).to_dict())
    print(f"neurons with no outputs left (excluding VPNs): {no_output.sum()}", n.loc[no_output, "type"].value_counts().head(8).to_dict())


if __name__ == "__main__":
    main()
