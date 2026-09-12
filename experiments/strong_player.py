"""A player built to win rather than to be a controlled experiment: search plus the fly's eye.

    uv run python experiments/strong_player.py --depths 0 1 --positions 200

Everything in checkers_play.py is shaped by the comparison it has to support -- a linear readout on
frozen features, scoring one afterstate at a time. This one drops those constraints and keeps only
the fly: the eye still does the seeing, but a search decides the move, and the leaves of the whole
tree are scored in a single batch.
"""

import argparse
import random
import time

import numpy as np
import pandas as pd

from checkers_play import eye_encoder, load_features, play_matches, search_player, squash
from fly_sim import checkers as ck
from fly_sim import probe, search
from fly_sim.connectome import CACHE_DIR

RESULTS = CACHE_DIR / "strong_player.csv"
HEAD = CACHE_DIR / "checkers_value_head.npz"


def value_head() -> probe.Ridge:
    if HEAD.exists():
        d = np.load(HEAD)
        return probe.Ridge(d["mean"], d["std"], d["coef"], d["target_mean"], float(d["loo"]), float(d["scale"]))
    train = load_features("fly", "train")
    fit = probe.fit(train["vpn"].astype(np.float64), squash(train["labels"])[:, None])
    np.savez(HEAD, mean=fit.mean, std=fit.std, coef=fit.coef, target_mean=fit.target_mean,
             loo=fit.loo_score, scale=fit.scale)
    return fit


class SearchPlayer:
    """Scores a whole search tree through the eye, batching every leaf of every game together."""

    def __init__(self, encode, head, depth: int = 1, cache_limit: int = 40000):
        self.encode, self.head, self.depth = encode, head, depth
        self.cache: dict[ck.Position, float] = {}
        self.cache_limit = cache_limit
        self.boards = 0

    def evaluate(self, positions: list[ck.Position]) -> np.ndarray:
        missing = [p for p in dict.fromkeys(positions) if p not in self.cache]
        if len(self.cache) > self.cache_limit:
            self.cache.clear()
            missing = list(dict.fromkeys(positions))
        for start in range(0, len(missing), 256):
            block = missing[start : start + 256]
            scores = self.head.predict(self.encode(block).astype(np.float64)).ravel()
            self.cache.update(zip(block, scores))
            self.boards += len(block)
        return np.array([self.cache[p] for p in positions])

    def __call__(self, batch):
        """One batched pass for every game in the batch: expand all trees, score all leaves once."""
        trees = [[search.expand(move.result, self.depth) for move in moves] for _, moves in batch]
        wanted: list[ck.Position] = []
        for game in trees:
            for tree in game:
                search.leaves(tree, wanted)
        unique = list(dict.fromkeys(wanted))
        values = dict(zip(unique, self.evaluate(unique))) if unique else {}
        chosen = []
        for (_, moves), game in zip(batch, trees):
            scores = [-search.fold(tree, values) for tree in game]
            chosen.append(moves[int(np.argmax(scores))])
        return chosen


def agreement(player: SearchPlayer, positions, batch: int = 8) -> float:
    """How often the searched choice matches the depth-6 teacher's, on the standing 200 positions."""
    hits = 0
    for start in range(0, len(positions), batch):
        block = positions[start : start + batch]
        chosen = player([(p, ck.legal_moves(p)) for p, _ in block])
        hits += sum(move.path[0] == best[0] and move.path[-1] == best[1] for move, (_, best) in zip(chosen, block))
    return hits / len(positions)


def benchmark_positions(count: int):
    """The same held-out positions every measurement in this project has used."""
    from checkers_play import positions_table

    table = positions_table("test")
    table = table[table["best_from"] >= 0]
    rng = np.random.default_rng(2)
    out = []
    for i in rng.permutation(len(table)):
        row = table.iloc[i]
        position = ck.Position(int(row.men), int(row.kings), int(row.opp_men), int(row.opp_kings))
        if len(ck.legal_moves(position)) > 1:
            out.append((position, (int(row.best_from), int(row.best_to))))
        if len(out) >= count:
            break
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--positions", type=int, default=200)
    parser.add_argument("--games", type=int, default=0, help="also play this many games per opponent")
    args = parser.parse_args()

    head, encode = value_head(), eye_encoder("fly", "vpn")
    positions = benchmark_positions(args.positions)
    rows = []
    for depth in args.depths:
        player = SearchPlayer(encode, head, depth)
        started = time.perf_counter()
        score = agreement(player, positions)
        row = {"depth": depth, "agreement": score, "boards": player.boards,
               "minutes": (time.perf_counter() - started) / 60}
        print(f"depth {depth}: teacher move agreement {score:.1%} · {player.boards:,} boards · "
              f"{row['minutes']:.1f} min", flush=True)

        for name, opponent in ([("random", 0), ("greedy (depth 1)", 1)] if args.games else []):
            player.boards = 0
            winners = play_matches([player, search_player(opponent, random.Random(5))], args.games)
            wins, draws = sum(w == 0 for w in winners), sum(w is None for w in winners)
            row[f"vs_{name}"] = (wins + draws / 2) / args.games
            print(f"  vs {name}: {wins}W {draws}D {args.games - wins - draws}L = "
                  f"{row[f'vs_{name}']:.1%}", flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(RESULTS, index=False)
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
