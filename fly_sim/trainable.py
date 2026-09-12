"""Backpropagation through time on the optic lobe, so the connectome itself can be trained.

Everywhere else in this repository the network is frozen and only a readout is fitted. Here the
6.1 million synaptic weights are trainable too, which turns the question from "is the fly's wiring
a useful fixed feature extractor" into "is it a useful starting point". The wiring diagram -- which
neuron contacts which -- is never changed; only the strength of each existing connection moves.

PyTorch's sparse autograd cannot do this: its CSR backward needs MKL, which Apple silicon lacks,
and its COO backward costs 4.2 s per board against 0.13 s for the gradient written out by hand.
So the gradient is written out by hand.

    r[t+1] = (1 - c) r[t] + c relu(g W r[t] + b + I[t]),   c = dt / tau

    dL/dr[t]  += (1 - c) dL/dr[t+1] + g W' (c dL/dr[t+1] . mask[t])
    dL/dW[e]  += g sum_b (c dL/dr[t+1] . mask[t])[row[e], b] r[t][col[e], b]
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from fly_sim.optic_lobe import OpticLobe
from fly_sim.rate_model import SUSTAINED_INPUTS, TRANSIENT_INPUTS, RateParams

GRADIENT_CHUNK = 2_000_000  # edges per block when accumulating the weight gradient


@dataclass(frozen=True)
class TrainingParams:
    steps: int = 16  # timesteps the board is shown for; 16 x 5 ms = 80 ms
    window: int = 6  # the last `window` steps are averaged into the features
    learning_rate: float = 2e-5  # Adam steps are ~lr, and a typical weight is ~0.008
    readout_learning_rate: float = 3e-3  # the readout starts from nothing, so it moves faster


class TrainableLobe:
    """One network, its weights trainable, scoring a board with a linear readout on the VPNs."""

    def __init__(self, lobe: OpticLobe, column_hex: np.ndarray, p: RateParams, t: TrainingParams, seed: int = 0):
        W = lobe.W.tocsr().astype(np.float32)
        self.scale = np.maximum(np.asarray(abs(W).sum(axis=1)).ravel(), 1).astype(np.float32)
        self.values = (W.data / np.repeat(self.scale, np.diff(W.indptr))).astype(np.float32)
        self.pattern = sp.csr_matrix((self.values.copy(), W.indices, W.indptr), shape=W.shape)

        self.row = np.repeat(np.arange(W.shape[0]), np.diff(W.indptr)).astype(np.int32)
        self.col = W.indices.astype(np.int32)
        # The backward pass needs W', kept as its own matrix whose data is refreshed from `values`.
        self.perm = np.argsort(self.col, kind="stable")
        counts = np.bincount(self.col, minlength=W.shape[0])
        self.transpose = sp.csr_matrix(
            (self.values[self.perm].copy(), self.row[self.perm], np.concatenate([[0], np.cumsum(counts)])),
            shape=W.shape,
        )

        self.n, self.p, self.t = W.shape[0], p, t
        self.outputs = lobe.outputs
        column_of = {(h1, h2): i for i, (h1, h2) in enumerate(column_hex)}
        neurons = lobe.neurons

        def inputs(types):
            index = np.flatnonzero(neurons["type"].isin(types).to_numpy())
            hexes = neurons.loc[index, ["hex1", "hex2"]].itertuples(index=False)
            return index, np.array([column_of[(h1, h2)] for h1, h2 in hexes])

        self.transient, self.transient_columns = inputs(TRANSIENT_INPUTS)
        self.sustained, self.sustained_columns = inputs(SUSTAINED_INPUTS)

        rng = np.random.default_rng(seed)
        self.readout = (rng.normal(size=len(self.outputs)) * 0.001).astype(np.float32)
        self.readout_bias = np.float32(0.0)
        self.start = self._resting()
        self.rest = self.start[self.outputs][:, None]  # features are measured against rest, as elsewhere

    def _resting(self, iterations: int = 400) -> np.ndarray:
        """Steady state with no stimulus, used as the initial condition for every board."""
        r = np.full(self.n, self.p.bias, dtype=np.float32)
        c = self.p.dt / self.p.tau
        for _ in range(iterations):
            r = (1 - c) * r + c * np.maximum(self.p.gain * (self.pattern @ r) + self.p.bias, 0)
        return r

    def _refresh(self):
        self.pattern.data[:] = self.values
        self.transpose.data[:] = self.values[self.perm]

    def renormalize(self):
        """Hold every neuron's total input budget at 1, as the frozen model assumes.

        Training moves the relative strength of a neuron's inputs, not how much drive it receives
        in total. Without this the gain of 1.4 -- calibrated against row-normalized weights, with
        linear instability near 1.67 -- is no longer meaningful and the rates can run away.
        """
        total = np.add.reduceat(np.abs(self.values), self.pattern.indptr[:-1])
        self.values /= np.maximum(np.repeat(total, np.diff(self.pattern.indptr)), 1e-6)

    def scores(self, luminance: np.ndarray, keep: bool = True):
        """Score each board. luminance is (batch, columns); returns scores and the saved trajectory."""
        batch = len(luminance)
        c, g = self.p.dt / self.p.tau, self.p.gain
        r = np.repeat(self.start[:, None], batch, axis=1)
        adapted = np.zeros_like(luminance)
        trail, masks = [], []

        features = np.zeros((len(self.outputs), batch), dtype=np.float32)
        for step in range(self.t.steps):
            adapted += (self.p.dt / self.p.adapt_tau) * (luminance - adapted)
            z = g * (self.pattern @ r) + self.p.bias
            z[self.transient] -= self.p.input_gain * (luminance - adapted)[:, self.transient_columns].T
            z[self.sustained] -= self.p.input_gain * luminance[:, self.sustained_columns].T
            mask = z > 0
            if keep:
                trail.append(r.copy())
                masks.append(mask)
            r = (1 - c) * r + c * np.where(mask, z, 0)
            if step >= self.t.steps - self.t.window:
                features += r[self.outputs]
        features = features / self.t.window - self.rest
        return self.readout @ features + self.readout_bias, (trail, masks, features)

    def backward(self, dscore: np.ndarray, saved, weights: bool = True):
        """Gradients of the loss with respect to the readout and, optionally, every synapse."""
        trail, masks, features = saved
        c, g = self.p.dt / self.p.tau, self.p.gain
        grad_readout = features @ dscore
        grad_bias = dscore.sum()
        if not weights:
            return grad_readout, grad_bias, None  # nothing else depends on the trajectory

        dr = np.zeros((self.n, len(dscore)), dtype=np.float32)
        contribution = np.outer(self.readout, dscore).astype(np.float32) / self.t.window
        grad_values = np.zeros_like(self.values) if weights else None

        for step in reversed(range(self.t.steps)):
            if step >= self.t.steps - self.t.window:
                dr[self.outputs] += contribution
            dz = np.where(masks[step], c * dr, 0)
            if weights:
                self._accumulate(grad_values, dz, trail[step], g)
            dr = (1 - c) * dr + g * (self.transpose @ dz)
        return grad_readout, grad_bias, grad_values

    def _accumulate(self, grad_values, dz, r, g):
        """grad[e] += g * sum_b dz[row[e], b] * r[col[e], b], in blocks to bound the temporaries."""
        for start in range(0, len(self.values), GRADIENT_CHUNK):
            block = slice(start, start + GRADIENT_CHUNK)
            grad_values[block] += g * np.einsum(
                "eb,eb->e", dz[self.row[block]], r[self.col[block]], optimize=False
            )


class Adam:
    def __init__(self, shape, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.lr, self.b1, self.b2, self.eps, self.step = learning_rate, beta1, beta2, eps, 0

    def __call__(self, gradient):
        self.step += 1
        self.m = self.b1 * self.m + (1 - self.b1) * gradient
        self.v = self.b2 * self.v + (1 - self.b2) * gradient**2
        m = self.m / (1 - self.b1**self.step)
        v = self.v / (1 - self.b2**self.step)
        return (-self.lr * m / (np.sqrt(v) + self.eps)).astype(np.float32)
