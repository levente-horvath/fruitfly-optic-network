"""Leaky integrate-and-fire simulation of a whole connectome, following Shiu et al. 2024."""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class LIFParams:
    """Defaults from github.com/philshiu/Drosophila_brain_model. Units: mV and seconds."""

    v_0: float = -52.0  # resting potential
    v_rst: float = -52.0  # reset potential
    v_th: float = -45.0  # spike threshold
    t_mbr: float = 0.020  # membrane time constant
    tau: float = 0.005  # synaptic time constant
    t_rfc: float = 0.0022  # refractory period
    t_dly: float = 0.0018  # synaptic delay
    w_syn: float = 0.275  # mV per synapse
    f_poi: float = 250.0  # strength of one Poisson input spike, in synapses
    dt: float = 0.0001


def simulate(
    W: sp.csc_matrix,
    stimulus: np.ndarray,
    rate: float = 150.0,
    duration: float = 1.0,
    params: LIFParams = LIFParams(),
    seed: int = 0,
) -> np.ndarray:
    """Drive `stimulus` neurons with Poisson input at `rate` Hz; return every neuron's spike count."""
    p = params
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    weights = W.data.astype(np.float64) * p.w_syn
    stimulus = np.asarray(stimulus)

    v = np.full(n, p.v_0)
    g = np.zeros(n)
    refractory = np.zeros(n, dtype=np.int32)
    counts = np.zeros(n, dtype=np.int64)

    delay_steps = round(p.t_dly / p.dt)
    refractory_steps = round(p.t_rfc / p.dt)
    in_flight = [np.empty(0, dtype=np.int64)] * delay_steps  # ring buffer of spikes awaiting delivery

    for step in range(round(duration / p.dt)):
        # Step order mirrors Brian2's schedule: integrate, threshold, synapses, reset.
        active = refractory == 0
        dv = (p.v_0 - v + g) * (p.dt / p.t_mbr)
        dg = -g * (p.dt / p.tau)
        v += dv * active
        g += dg * active

        spiked = np.flatnonzero((v > p.v_th) & active)

        slot = step % delay_steps
        if in_flight[slot].size:
            g += _propagate(W.indptr, W.indices, weights, in_flight[slot], n)
        in_flight[slot] = spiked
        v[stimulus[rng.random(stimulus.size) < rate * p.dt]] += p.f_poi * p.w_syn

        v[spiked] = p.v_rst
        g[spiked] = 0.0
        counts[spiked] += 1
        np.subtract(refractory, 1, out=refractory, where=refractory > 0)
        refractory[spiked] = refractory_steps

    return counts


def _propagate(indptr, indices, weights, sources, n):
    """Sum the outgoing weight columns of `sources` into a dense input vector."""
    starts = indptr[sources]
    lengths = indptr[sources + 1] - starts
    offsets = np.repeat(starts - np.cumsum(lengths) + lengths, lengths) + np.arange(lengths.sum())
    return np.bincount(indices[offsets], weights=weights[offsets], minlength=n)
