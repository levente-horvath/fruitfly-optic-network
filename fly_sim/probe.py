"""Ridge readout with the ridge strength chosen by exact leave-one-out on the training set.

The same fit serves classification (one-hot targets, scored by accuracy) and value regression
(one column, scored by mean squared error), so probes on the fly features and on every baseline
go through one implementation.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

SCALES = np.logspace(-6, 4, 21)


@dataclass
class Ridge:
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray  # (features, targets), applied to standardized features
    target_mean: np.ndarray
    loo_score: float
    scale: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std) @ self.coef + self.target_mean


def accuracy(labels: np.ndarray) -> Callable[[np.ndarray], float]:
    return lambda prediction: float((prediction.argmax(axis=1) == labels).mean())


def negative_mse(prediction: np.ndarray) -> float:
    return -float((prediction**2).mean())  # the prediction passed in is already the error


def fit(x: np.ndarray, targets: np.ndarray, score: Callable[[np.ndarray], float] | None = None) -> Ridge:
    """Fit standardized-feature ridge regression to `targets`, picking the scale that scores best."""
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1
    x = (x - mean) / std
    target_mean = targets.mean(axis=0)
    t = targets - target_mean
    score = score or (lambda prediction: -float(((prediction - targets) ** 2).mean()))

    # Thin SVD x = U S V^T, via whichever Gram matrix is smaller.
    n, d = x.shape
    s2, basis = np.linalg.eigh(x @ x.T if n <= d else x.T @ x)
    keep = s2 > 1e-9 * s2.max()
    s2, basis = s2[keep], basis[:, keep]
    U, V = (basis, (x.T @ basis) / np.sqrt(s2)) if n <= d else ((x @ basis) / np.sqrt(s2), basis)
    del x

    # Leave-one-out: loo_i = t_i - residual_i / (1 - leverage_i). The leverage includes 1/n for the
    # intercept; without it the held-out row is exactly minus the sum of the other centered rows,
    # and a model that fits them predicts it perfectly. With fewer rows than features both residual
    # and 1 - leverage are near zero, so compute them from ridge / (s2 + ridge) and the parts
    # outside U's span, instead of subtracting nearly equal numbers.
    Ut = U.T @ t
    outside_t = t - U @ Ut
    outside_leverage = 1 - np.einsum("ij,ij->i", U, U) - 1 / n
    best = (-np.inf, SCALES[0])
    for scale in SCALES:
        kept_out = (scale * s2.mean()) / (s2 + scale * s2.mean())
        residual = U @ (kept_out[:, None] * Ut) + outside_t
        one_minus_leverage = np.einsum("ij,ij,j->i", U, U, kept_out) + outside_leverage
        loo = t - residual / one_minus_leverage[:, None]
        value = score(loo + target_mean)
        if value > best[0]:
            best = (value, scale)

    loo_score, scale = best
    coef = V @ ((np.sqrt(s2) / (s2 + scale * s2.mean()))[:, None] * Ut)
    return Ridge(mean, std, coef, target_mean, loo_score, scale)
