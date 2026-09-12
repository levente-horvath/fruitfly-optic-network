"""Does stimulating loom-detecting LPLC2 neurons drive the giant-fiber escape neuron (DNp01)?

No training: the network is the male CNS connectome with Shiu et al. 2024 parameters. As a
control, the same number of randomly chosen visual projection neurons is stimulated instead.
Firing rates for both conditions are saved for experiments/activity_cloud.py.
"""

import time

import numpy as np
import pandas as pd

from fly_sim.connectome import CACHE_DIR, load
from fly_sim.lif import simulate

N_TRIALS = 3
DURATION = 1.0  # seconds per trial
RATES_PATH = CACHE_DIR / "lplc2_giant_fiber_rates.parquet"


def run(connectome, stimulus, label):
    start = time.perf_counter()
    counts = sum(
        simulate(connectome.W, stimulus, duration=DURATION, seed=trial) for trial in range(N_TRIALS)
    )
    rates = counts / (N_TRIALS * DURATION)
    neurons = connectome.neurons.assign(rate=rates)

    print(f"\n== {label}: {len(stimulus)} neurons stimulated ({time.perf_counter() - start:.0f}s)")
    print("giant fiber:")
    print(neurons.loc[neurons["type"] == "DNp01", ["instance", "rate"]].to_string(index=False))

    downstream = neurons.drop(index=stimulus)
    print(f"downstream neurons firing > 1 Hz: {(downstream['rate'] > 1).sum():,}")
    top = (
        downstream[downstream["rate"] > 0]
        .groupby("type")["rate"]
        .agg(mean_hz="mean", neurons="count")
        .sort_values("mean_hz", ascending=False)
        .head(15)
    )
    print("most active downstream types:")
    print(top.round(1).to_string())
    return rates


def main():
    connectome = load()
    lplc2 = connectome.indices(type="LPLC2")
    candidates = np.setdiff1d(connectome.indices(superclass="visual_projection"), lplc2)
    control = np.random.default_rng(0).choice(candidates, size=len(lplc2), replace=False)

    results = pd.DataFrame({"bodyId": connectome.neurons["bodyId"]})
    for key, stimulus, label in [
        ("lplc2", lplc2, "LPLC2 (loom detectors)"),
        ("control", control, "Control: random visual projection neurons"),
    ]:
        results[f"{key}_rate"] = run(connectome, stimulus, label)
        results[f"{key}_stimulated"] = np.isin(np.arange(len(results)), stimulus)
    results.to_parquet(RATES_PATH)
    print(f"\nsaved rates to {RATES_PATH}")


if __name__ == "__main__":
    main()
