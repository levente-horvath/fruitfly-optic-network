"""Train the readout on positions the network actually reaches, not the ones the teacher reaches.

    uv run python experiments/self_play.py --rounds 5 --games 60

The frozen experiments trained on positions drawn from teacher-quality games and then asked the
network to evaluate positions its own play produced -- a distribution it never saw. Each round here
plays self-play games with the current network, has the depth-6 teacher label the positions those
games actually visited, adds them to the pool and refits. The data is self-play; the labels stay
with the teacher, which is a far stronger signal than bootstrapping a weak network off itself.

Writes data/processed/self_play.csv.
"""

import argparse
import os
import random
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

from checkers_play import ValuePlayer, eye_encoder, load_features, play_matches, search_player, squash
from checkers_positions import LABEL_DEPTH
from fly_sim import checkers as ck
from fly_sim import probe
from fly_sim.connectome import CACHE_DIR

RESULTS = CACHE_DIR / "self_play.csv"


def epsilon_greedy(player: ValuePlayer, epsilon: float, rng: random.Random):
    def choose(batch):
        picked = player(batch)
        return [rng.choice(moves) if rng.random() < epsilon else move
                for (_, moves), move in zip(batch, picked)]
    return choose


def self_play_positions(player, games: int, epsilon: float, seed: int) -> list[ck.Position]:
    """Play the network against itself and return every position the games passed through."""
    rng = random.Random(seed)
    mover = epsilon_greedy(player, epsilon, rng)
    boards = [ck.START] * games
    alive, quiet, seen = list(range(games)), [0] * games, []

    for _ in range(160):
        if not alive:
            break
        batch, playing = [], []
        for game in alive:
            moves = ck.legal_moves(boards[game])
            if moves:
                batch.append((boards[game], moves))
                playing.append(game)
                seen.append(boards[game])
        if not batch:
            break
        for game, move in zip(playing, mover(batch)):
            quiet[game] = 0 if move.captures or move.promotes else quiet[game] + 1
            boards[game] = move.result
        alive = [g for g in playing if quiet[g] < 60]
    return seen


def label(position_tuple):
    position = ck.Position(*position_tuple)
    move, _ = ck.best_move(position, LABEL_DEPTH)
    return (move.path[0], move.path[-1]) if move else (-1, -1)


def make_pairs(positions, labels, rng):
    """(afterstate of the teacher's move, afterstate of another legal move)."""
    pairs = []
    for position, (best_from, best_to) in zip(positions, labels):
        if best_from < 0:
            continue
        moves = ck.legal_moves(position)
        if len(moves) < 2:
            continue
        best = next((m for m in moves if m.path[0] == best_from and m.path[-1] == best_to), None)
        if best is None:
            continue
        others = [m for m in moves if m is not best]
        pairs.append((best.result, others[rng.randrange(len(others))].result))
    return pairs


def fit_ranker(features_best, features_alt, epochs=40, batch=64, lr=3e-3):
    """Logistic ranking: the teacher's afterstate must outscore the alternative."""
    from fly_sim.trainable import Adam

    w = np.zeros(features_best.shape[1])
    opt = Adam(w.shape, lr)
    for epoch in range(epochs):
        order = np.random.default_rng(epoch).permutation(len(features_best))
        for s in range(0, len(order), batch):
            i = order[s : s + batch]
            difference = features_alt[i] - features_best[i]
            w = w + opt(difference.T @ (1 / (1 + np.exp(-(difference @ w)))) / len(i))
    return w


class Ranker:
    """A plain linear score on the cached feature set, shaped like probe.Ridge for ValuePlayer."""

    def __init__(self, weights):
        self.weights = weights

    def predict(self, x):
        return -(x @ self.weights)[:, None]  # ValuePlayer negates, and higher must mean better


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--network", default="fly")
    args = parser.parse_args()
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[var] = "1"

    encode = eye_encoder(args.network, "vpn")
    train = load_features(args.network, "train")

    head = probe.fit(train["vpn"].astype(np.float64), squash(train["labels"])[:, None])
    player = ValuePlayer(encode, head)
    pool_best, pool_alt, rows = [], [], []

    for round_index in range(1, args.rounds + 1):
        started = time.perf_counter()
        positions = self_play_positions(player, args.games, args.epsilon, seed=round_index)
        unique = list(dict.fromkeys(positions))
        print(f"round {round_index}: {len(positions):,} positions from {args.games} self-play games, "
              f"{len(unique):,} unique · {(time.perf_counter() - started) / 60:.1f} min", flush=True)

        with get_context("spawn").Pool(args.workers) as pool:
            labels = pool.map(label, [tuple(p) for p in unique], chunksize=32)
        pairs = make_pairs(unique, labels, random.Random(round_index))
        print(f"  labelled, {len(pairs):,} usable pairs · {(time.perf_counter() - started) / 60:.1f} min", flush=True)

        pool_best += [a for a, _ in pairs]
        pool_alt += [b for _, b in pairs]
        features_best = player.features(pool_best).astype(np.float64)
        features_alt = player.features(pool_alt).astype(np.float64)
        weights = fit_ranker(features_best, features_alt)
        player = ValuePlayer(encode, Ranker(weights))
        player.cache = {}

        scores = {}
        for name, depth in [("random", 0), ("greedy (depth 1)", 1)]:
            w = play_matches([player, search_player(depth, random.Random(99))], 40)
            scores[name] = (sum(x == 0 for x in w) + sum(x is None for x in w) / 2) / 40
        rows.append({"round": round_index, "pool_pairs": len(pool_best),
                     "vs_random": scores["random"], "vs_greedy": scores["greedy (depth 1)"],
                     "minutes": (time.perf_counter() - started) / 60})
        print(f"  pool {len(pool_best):,} pairs · vs random {scores['random']:.1%} · "
              f"vs greedy {scores['greedy (depth 1)']:.1%} · {rows[-1]['minutes']:.1f} min", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS, index=False)

    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
