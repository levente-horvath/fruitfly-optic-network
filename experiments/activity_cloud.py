"""Render every neuron's soma as an interactive 3D point cloud colored by simulated activity.

Run experiments/lplc2_giant_fiber.py first; it saves the firing rates shown here. Writes a
self-contained HTML file to data/processed/activity_cloud.html.
"""

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from fly_sim.connectome import ANNOTATIONS, CACHE_DIR, load
from lplc2_giant_fiber import DURATION, N_TRIALS, RATES_PATH

TEMPLATE = Path(__file__).resolve().parents[1] / "viz" / "activity_cloud.html"
OUTPUT = CACHE_DIR / "activity_cloud.html"
VOXEL_UM = 0.008  # MaleCNS coordinates are 8 nm voxels

CONDITIONS = [("lplc2", "LPLC2"), ("control", "Control")]


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def codes(values: pd.Series, dtype) -> tuple[np.ndarray, list[str]]:
    """Integer-encode a string column; 0 means missing."""
    categorical = pd.Categorical(values)
    return (categorical.codes + 1).astype(dtype), [""] + list(categorical.categories)


def main():
    connectome = load()
    rates = pd.read_parquet(RATES_PATH)
    somas = feather.read_table(ANNOTATIONS, columns=["bodyId", "somaLocation", "tosomaLocation"])
    neurons = connectome.neurons.merge(somas.to_pandas(), on="bodyId", how="left")
    assert (neurons["bodyId"].to_numpy() == rates["bodyId"].to_numpy()).all()

    location = neurons["somaLocation"].where(neurons["somaLocation"].notna(), neurons["tosomaLocation"])
    shown = location.notna().to_numpy()
    xyz = np.stack(location[shown].to_numpy()).astype(np.float64) * VOXEL_UM
    x, y, z = (xyz - (xyz.min(0) + xyz.max(0)) / 2).T
    positions = np.column_stack([x, -z, y]).astype(np.float32)  # body axis vertical: brain on top

    shown_neurons = neurons[shown]
    superclass, superclass_names = codes(shown_neurons["superclass"], np.uint8)
    cell_type, type_names = codes(shown_neurons["type"], np.uint16)
    side = shown_neurons["somaSide"].map({"L": 1, "R": 2, "M": 3}).fillna(0).astype(np.uint8)

    payload = {
        "count": int(shown.sum()),
        "total": len(neurons),
        "subtitle": f"Untrained LIF model · mean of {N_TRIALS} × {DURATION:g} s trials",
        "positions": b64(positions),
        "bodyIds": b64(shown_neurons["bodyId"].to_numpy(np.float64)),
        "superclass": b64(superclass),
        "superclassNames": superclass_names,
        "type": b64(cell_type),
        "typeNames": type_names,
        "side": b64(side.to_numpy()),
        "conditions": [
            {
                "name": name,
                "rates": b64(rates.loc[shown, f"{key}_rate"].to_numpy(np.float32)),
                "stimulated": b64(rates.loc[shown, f"{key}_stimulated"].to_numpy(np.uint8)),
            }
            for key, name in CONDITIONS
        ],
    }
    OUTPUT.write_text(TEMPLATE.read_text().replace("__PAYLOAD__", json.dumps(payload)))
    print(f"{shown.sum():,} of {len(neurons):,} neurons have a soma position")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
