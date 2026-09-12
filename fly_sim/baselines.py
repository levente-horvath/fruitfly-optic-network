"""Null networks: the same neurons and the same input and output sets, but the wiring destroyed.

A large linear readout on fixed features can do well on almost any expansion, so a fly-network
score only means something next to these. Both nulls keep every neuron's identity, its column
assignment and its output synapse budget, and change only who it talks to.
"""

import dataclasses

import numpy as np
import scipy.sparse as sp

from fly_sim.optic_lobe import OpticLobe


def _presynaptic_signs(W: sp.csc_matrix) -> np.ndarray:
    """Each neuron's neurotransmitter sign, read back off the columns of W (0 if it has no outputs)."""
    csc = W.tocsc()
    signs = np.zeros(W.shape[0], dtype=np.float32)
    for pre in range(W.shape[0]):
        data = csc.data[csc.indptr[pre] : csc.indptr[pre + 1]]
        if data.size:
            signs[pre] = np.sign(data[0])
    return signs


def rewired(lobe: OpticLobe, seed: int = 0) -> OpticLobe:
    """Degree-preserving rewire: every edge keeps its sender and weight, and gets a random receiver.

    This is the directed configuration model. Each neuron keeps its exact number and strength of
    output synapses and its exact number of input edges; only the wiring diagram is randomized.
    """
    coo = lobe.W.tocoo()
    rng = np.random.default_rng(seed)
    order = rng.permutation(coo.nnz)  # the (sender, weight) pairs move together to new receivers
    pre, data = coo.col[order], coo.data[order]
    keep = pre != coo.row  # drop the self-loops the shuffle creates (~0.9% of edges)
    W = sp.coo_matrix((data[keep], (coo.row[keep], pre[keep])), shape=lobe.W.shape).tocsc()
    return dataclasses.replace(lobe, W=W)


def random_network(lobe: OpticLobe, seed: int = 0) -> OpticLobe:
    """Erdos-Renyi network of the same size and density, with the real weights and per-neuron signs."""
    coo = lobe.W.tocoo()
    n, nnz = lobe.W.shape[0], coo.nnz
    rng = np.random.default_rng(seed)
    signs = _presynaptic_signs(lobe.W)

    row, col = rng.integers(0, n, size=nnz), rng.integers(0, n, size=nnz)
    keep = row != col
    magnitude = rng.permutation(np.abs(coo.data))[: keep.sum()]
    W = sp.coo_matrix((magnitude * signs[col[keep]], (row[keep], col[keep])), shape=lobe.W.shape).tocsc()
    return dataclasses.replace(lobe, W=W)


NETWORKS = {"fly": lambda lobe, seed: lobe, "rewired": rewired, "random": random_network}


def build(name: str, lobe: OpticLobe, seed: int = 0) -> OpticLobe:
    return NETWORKS[name](lobe, seed)
