"""Build the labeled checkers position dataset the value head is trained on.

    uv run python experiments/checkers_positions.py --games 900 --workers 4

Positions come from games between players of mixed strength, from random to a depth-4 search, so
the set covers both sensible and ragged positions. Every unique position is then labeled with the
value of a deeper alpha-beta search, seen from the side to move. Train and test are split by game
and positions shared between them are removed from test.

Writes data/datasets/checkers/positions.parquet.
"""

import argparse
import os
import random
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

from fly_sim import checkers as ck
from fly_sim.stimulus import CHECKERS_POSITIONS as POSITIONS
LABEL_DEPTH = 6  # the teacher whose values the readout has to reproduce
TEST_FRACTION = 0.15
# (search depth, chance of playing a uniformly random move instead); depth 0 is a random player.
STYLES = [(0, 1.0), (1, 0.3), (2, 0.3), (4, 0.2), (4, 0.05)]


def policy(depth: int, noise: float, rng: random.Random):
    def choose(position, moves):
        if depth == 0 or rng.random() < noise:
            return rng.choice(moves)
        return ck.best_move(position, depth, rng)[0]

    return choose


def generate(games: int, seed: int) -> pd.DataFrame:
    """Play the games and return every position with the game it came from."""
    rows = []
    for game in range(games):
        rng = random.Random(f"{seed}-{game}")
        players = [policy(*rng.choice(STYLES), rng) for _ in range(2)]
        result = ck.play_game(players)
        for ply, position in enumerate(result.positions):
            rows.append((*position, game, ply))
        if (game + 1) % 50 == 0:
            print(f"  {game + 1}/{games} games · {len(rows):,} positions", flush=True)
    return pd.DataFrame(rows, columns=["men", "kings", "opp_men", "opp_kings", "game", "ply"])


def label(row) -> tuple[int, int, int]:
    """Teacher value of the position and the from/to squares of its best move."""
    position = ck.Position(*row)
    move, value = ck.best_move(position, LABEL_DEPTH)
    return value, (move.path[0] if move else -1), (move.path[-1] if move else -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=900)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    POSITIONS.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    positions = generate(args.games, args.seed)
    test_games = set(random.Random(args.seed).sample(range(args.games), round(args.games * TEST_FRACTION)))
    positions["split"] = ["test" if g in test_games else "train" for g in positions["game"]]

    # Dedupe globally, keeping the training copy of any position both splits reached, so a test
    # position is one no training game ever stood in.
    keys = ["men", "kings", "opp_men", "opp_kings"]
    from_test = positions["split"].eq("test").to_numpy()
    order = np.lexsort((positions["ply"].to_numpy(), positions["game"].to_numpy(), from_test))
    unique = positions.iloc[order].drop_duplicates(keys).reset_index(drop=True)
    counts = unique["split"].value_counts()
    print(
        f"{len(positions):,} positions from {args.games} games -> {len(unique):,} unique "
        f"({counts.get('train', 0):,} train, {counts.get('test', 0):,} test, "
        f"{from_test.sum() - counts.get('test', 0):,} test positions already seen in training games) "
        f"· {time.perf_counter() - start:.0f}s",
        flush=True,
    )

    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[var] = "1"
    start, labels = time.perf_counter(), []
    with get_context("spawn").Pool(args.workers) as pool:
        for i, out in enumerate(pool.imap(label, unique[keys].itertuples(index=False, name=None), chunksize=64), start=1):
            labels.append(out)
            if i % 5000 == 0:
                elapsed = time.perf_counter() - start
                print(f"  labeled {i:,}/{len(unique):,} · {elapsed / 60:.1f} min · ~{(len(unique) - i) * elapsed / i / 60:.0f} min left", flush=True)
    unique[["value", "best_from", "best_to"]] = labels

    unique.to_parquet(POSITIONS, index=False)
    print(f"\nwrote {POSITIONS}")
    print(unique.groupby("split")["value"].describe().round(1).to_string())


if __name__ == "__main__":
    main()
