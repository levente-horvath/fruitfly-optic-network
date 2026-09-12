"""Find a board encoding the eye can read almost perfectly.

    uv run python experiments/encoding_search.py

Perception is the ceiling on play. At 99.4% per square only 0.994^32 = 82% of boards are read
without a single error, and one misread piece is worth about a man -- which swamps every positional
judgement the evaluation is trying to make. This tries candidate encodings and measures the one
thing that matters: how often every square is right.

Writes data/processed/encoding_search.csv.
"""

import argparse
import time

import numpy as np
import pandas as pd

from checkers_play import CLASSES, one_hot, positions_table, square_contents
from fly_sim import checkers as ck
from fly_sim import probe
from fly_sim.connectome import CACHE_DIR
from fly_sim.stimulus import Presentation, render_board
from fly_sim.vision import Eye

RESULTS = CACHE_DIR / "encoding_search.csv"

# name -> (pieces {board value: (radius in squares, luminance)}, presentation)
# The current encoding leans on one luminance step and one size step; the alternatives separate the
# four classes further, or give the board more of the eye.
ENCODINGS = {
    "current": ({1: (0.34, 0.95), 2: (0.46, 0.95), -1: (0.34, 0.45), -2: (0.46, 0.45)}, Presentation(size=22.0, jitter=0.5)),
    "wider size contrast": ({1: (0.28, 0.95), 2: (0.50, 0.95), -1: (0.28, 0.45), -2: (0.50, 0.45)}, Presentation(size=22.0, jitter=0.5)),
    "four luminances": ({1: (0.40, 1.00), 2: (0.40, 0.72), -1: (0.40, 0.44), -2: (0.40, 0.20)}, Presentation(size=22.0, jitter=0.5)),
    "four luminances + size": ({1: (0.30, 1.00), 2: (0.50, 0.75), -1: (0.30, 0.42), -2: (0.50, 0.20)}, Presentation(size=22.0, jitter=0.5)),
    "bigger board": ({1: (0.30, 1.00), 2: (0.50, 0.75), -1: (0.30, 0.42), -2: (0.50, 0.20)}, Presentation(size=26.0, jitter=0.5)),
    "longer look": ({1: (0.30, 1.00), 2: (0.50, 0.75), -1: (0.30, 0.42), -2: (0.50, 0.20)}, Presentation(size=22.0, jitter=0.5, duration=0.6)),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boards", type=int, default=3000)
    parser.add_argument("--train", type=int, default=2200)
    args = parser.parse_args()

    table = positions_table("train").iloc[: args.boards]
    positions = [ck.Position(int(r.men), int(r.kings), int(r.opp_men), int(r.opp_kings))
                 for r in table.itertuples()]
    contents = square_contents(table)
    targets = one_hot(contents)
    eye = Eye("fly")
    eye.model

    rows = []
    for name, (pieces, presentation) in ENCODINGS.items():
        started = time.perf_counter()
        images = [render_board(ck.to_array(p), piece=pieces) for p in positions]
        features = []
        for start in range(0, len(images), 64):
            block = slice(start, start + 64)
            seeds = [[int(v) for v in p] for p in positions[block]]
            features.append(eye.look(images[block], seeds, presentation)[0])
        features = np.concatenate(features).astype(np.float64)

        cut = args.train
        fit = probe.fit(features[:cut], targets[:cut],
                        score=lambda p: (p.reshape(len(p), 32, 5).argmax(2) == contents[:cut]).mean())
        guess = fit.predict(features[cut:]).reshape(-1, 32, 5).argmax(2)
        truth = contents[cut:]
        per_square = (guess == truth).mean()
        rows.append({
            "encoding": name,
            "square_accuracy": per_square,
            "perfect_boards": float((guess == truth).all(axis=1).mean()),
            "worst_class": min(float((guess[truth == i] == i).mean()) for i in range(len(CLASSES))),
            "minutes": (time.perf_counter() - started) / 60,
        })
        print(f"{name:24s} squares {per_square:.3%} · whole boards right {rows[-1]['perfect_boards']:.1%} · "
              f"worst class {rows[-1]['worst_class']:.1%} · {rows[-1]['minutes']:.1f} min", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS, index=False)

    best = max(rows, key=lambda r: r["perfect_boards"])
    print(f"\nbest: {best['encoding']} — {best['perfect_boards']:.1%} of boards read with no errors "
          f"(current: {rows[0]['perfect_boards']:.1%})")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
