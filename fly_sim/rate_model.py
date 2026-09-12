"""Frozen firing-rate dynamics on the right-eye optic lobe, driven by luminance movies."""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from fly_sim.optic_lobe import OpticLobe

# Columnar lamina inputs: L1 and L2 respond transiently to contrast, L3 more tonically.
TRANSIENT_INPUTS = ["L1", "L2"]
SUSTAINED_INPUTS = ["L3"]


@dataclass(frozen=True)
class RateParams:
    gain: float = 1.0  # recurrent gain on the row-normalized weights; linear instability above ~1.67
    input_gain: float = 1.0  # stimulus drive, relative to the tonic bias
    bias: float = 1.0  # tonic drive to every neuron; all rates scale with it
    tau: float = 0.02  # neuron time constant, seconds
    adapt_tau: float = 0.05  # adaptation time constant of the transient inputs, seconds
    dt: float = 0.005  # integration step, seconds


CALIBRATED = RateParams(gain=1.4, input_gain=30.0)  # chosen by experiments/calibrate_rate_model.py


class RateModel:
    """tau dr/dt = -r + relu(gain * W_hat r + bias + input), each row of W_hat normalized to sum |w| = 1.

    Normalizing each neuron's input keeps the relative strength of its synapses but removes the
    ~80-fold spread in total input synapses across neurons, so one gain suits all of them.
    Photoreceptors inhibit the lamina, so light drives L1-L3 negatively.
    """

    def __init__(self, lobe: OpticLobe, column_hex: np.ndarray):
        W = lobe.W.tocsr().astype(np.float64)
        total = np.asarray(abs(W).sum(axis=1)).ravel()
        self.W = (sp.diags(1 / np.maximum(total, 1)) @ W).tocsr().astype(np.float32)
        self._baselines = {}

        column_of = {(h1, h2): i for i, (h1, h2) in enumerate(column_hex)}
        neurons = lobe.neurons

        def inputs(types):
            index = np.flatnonzero(neurons["type"].isin(types).to_numpy())
            hexes = neurons.loc[index, ["hex1", "hex2"]].itertuples(index=False)
            return index, np.array([column_of[(h1, h2)] for h1, h2 in hexes])

        self.transient, self.transient_columns = inputs(TRANSIENT_INPUTS)
        self.sustained, self.sustained_columns = inputs(SUSTAINED_INPUTS)

    def baseline(self, p: RateParams, tol: float = 1e-5, max_steps: int = 4000) -> tuple[np.ndarray, int]:
        """Steady-state rates with no stimulus, and the number of steps it took to settle."""
        if p not in self._baselines:
            r = np.full(self.W.shape[0], p.bias, dtype=np.float32)
            for steps in range(1, max_steps + 1):
                step = (p.dt / p.tau) * (np.maximum(p.gain * (self.W @ r) + p.bias, 0) - r)
                r += step
                if np.abs(step).max() < tol * p.bias:
                    break
            self._baselines[p] = (r, steps)
        return self._baselines[p]

    def run(
        self,
        movies: np.ndarray,
        frame_dt: float,
        window: tuple[int, int],
        p: RateParams,
        background: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate movies shaped (batch, frames, columns), starting from the baseline state.

        Returns each neuron's mean rate over frames [window[0], window[1]) and its peak rate over
        the whole movie, both shaped (batch, neurons).
        """
        resting = self.baseline(p)[0]
        total = np.zeros((len(resting), len(movies)), dtype=np.float32)
        peak = np.repeat(resting[:, None], len(movies), axis=1)
        count = 0
        for f, r in self._steps(movies, frame_dt, p, background):
            np.maximum(peak, r, out=peak)
            if window[0] <= f < window[1]:
                total += r
                count += 1
        return (total / count).T, peak.T

    def trajectory(self, movies: np.ndarray, frame_dt: float, p: RateParams, background: float = 0.0) -> np.ndarray:
        """Every neuron's rate at the end of each frame, shaped (batch, frames, neurons)."""
        batch, frames, _ = movies.shape
        rates = np.empty((batch, frames, self.W.shape[0]), dtype=np.float32)
        substeps = round(frame_dt / p.dt)
        for step, (f, r) in enumerate(self._steps(movies, frame_dt, p, background), start=1):
            if step % substeps == 0:
                rates[:, f] = r.T
        return rates

    def _steps(self, movies, frame_dt, p, background):
        """Integrate from the baseline state, yielding (frame, rates shaped (neurons, batch)) after each step."""
        batch, frames, columns = movies.shape
        r = np.repeat(self.baseline(p)[0][:, None], batch, axis=1)  # (neurons, batch) for fast W @ r
        adapted = np.full((batch, columns), background, dtype=np.float32)
        for f in range(frames):
            frame = movies[:, f]
            for _ in range(round(frame_dt / p.dt)):
                adapted += (p.dt / p.adapt_tau) * (frame - adapted)
                drive = p.gain * (self.W @ r) + p.bias
                drive[self.transient] -= p.input_gain * (frame - adapted)[:, self.transient_columns].T
                drive[self.sustained] -= p.input_gain * (frame - background)[:, self.sustained_columns].T
                np.maximum(drive, 0, out=drive)
                r += (p.dt / p.tau) * (drive - r)
                yield f, r
