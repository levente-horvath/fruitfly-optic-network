"""Read the board off the eye, then judge the board. Two stages, each trained on what suits it.

    uv run python experiments/strong_readout.py --dataset checkers-v2

The one-stage readouts in this project ask a single linear map to see the board and evaluate it at
once, from one scalar label per position. That is a bad trade: per-square supervision fits a decoder
to 99.7% from a few thousand positions, while a scalar value target could not even find the
composition that decoder implies.

So this splits them. The decoder is fitted on simulated features with dense per-square targets. The
evaluator is fitted on *true* boards with a ranking loss -- no simulation at all, so it can have as
many training pairs as we care to generate -- and is then applied to what the decoder recovers.
Because it will meet decoding errors at play time, it is also trained on boards corrupted at the
decoder's own measured error rate.

Writes data/processed/strong_readout.csv.
"""

import argparse
import random
import time

import numpy as np
import pandas as pd

import extract_features
from checkers_play import CLASSES, one_hot, positions_table, square_contents
from fly_sim import checkers as ck
from fly_sim import probe
from fly_sim.connectome import CACHE_DIR
from fly_sim.trainable import Adam

RESULTS = CACHE_DIR / "strong_readout.csv"
DECODER = CACHE_DIR / "strong_decoder.npz"
EVALUATOR = CACHE_DIR / "strong_evaluator.npz"


def fit_decoder(features: np.ndarray, contents: np.ndarray) -> probe.Ridge:
    """Features -> the contents of all 32 squares, 160 dense targets per position."""
    return probe.fit(features, one_hot(contents),
                     score=lambda p: (p.reshape(len(p), 32, 5).argmax(2) == contents).mean())


def confusion(decoded: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """P(decoder says j | the square really is i), so the evaluator can train on realistic errors."""
    guess = decoded.reshape(len(decoded), 32, len(CLASSES)).argmax(2)
    table = np.zeros((len(CLASSES), len(CLASSES)))
    for i in range(len(CLASSES)):
        mask = truth == i
        table[i] = (np.bincount(guess[mask], minlength=len(CLASSES)) / mask.sum()) if mask.sum() else np.eye(len(CLASSES))[i]
    return table / table.sum(axis=1, keepdims=True)  # np.random.choice rejects rows off 1 by 1e-9


def board_pairs(split: str, count: int, seed: int):
    """(teacher's afterstate, another legal afterstate) as square contents -- boards only, no eye."""
    table = positions_table(split)
    table = table[table["best_from"] >= 0]
    rng = random.Random(seed)
    best_boards, other_boards = [], []
    order = list(range(len(table)))
    rng.shuffle(order)
    for i in order:
        row = table.iloc[i]
        position = ck.Position(int(row.men), int(row.kings), int(row.opp_men), int(row.opp_kings))
        moves = ck.legal_moves(position)
        if len(moves) < 2:
            continue
        best = next((m for m in moves if m.path[0] == row.best_from and m.path[-1] == row.best_to), None)
        if best is None:
            continue
        others = [m for m in moves if m is not best]
        best_boards.append(best.result)
        other_boards.append(others[rng.randrange(len(others))].result)
        if len(best_boards) >= count:
            break
    return [contents_of(best_boards), contents_of(other_boards)]


def contents_of(positions):
    return square_contents(pd.DataFrame(positions, columns=["men", "kings", "opp_men", "opp_kings"]))


def corrupt(contents: np.ndarray, table: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply the decoder's own confusion to a true board."""
    out = contents.copy()
    for i in range(len(CLASSES)):
        mask = contents == i
        if mask.sum():
            out[mask] = rng.choice(len(CLASSES), size=int(mask.sum()), p=table[i])
    return out


def fit_evaluator(best, other, table, epochs=30, batch=64, lr=3e-3, seed=0):
    """Linear ranker on the board, trained on boards corrupted the way the decoder corrupts them."""
    rng = np.random.default_rng(seed)
    w = np.zeros(32 * len(CLASSES))
    opt = Adam(w.shape, lr)
    for epoch in range(epochs):
        noisy_best = one_hot(corrupt(best, table, rng)) if table is not None else one_hot(best)
        noisy_other = one_hot(corrupt(other, table, rng)) if table is not None else one_hot(other)
        order = rng.permutation(len(best))
        for s in range(0, len(order), batch):
            i = order[s : s + batch]
            difference = noisy_other[i] - noisy_best[i]
            w = w + opt(difference.T @ (1 / (1 + np.exp(-(difference @ w)))) / len(i))
    return w


def composed_agreement(decoder, weights, dataset: str, count: int = 200) -> float:
    """Decode each legal afterstate off the eye, judge the decoded board, compare with the teacher."""
    from fly_sim.stimulus import PIECE_V2, render_board
    from fly_sim.vision import BOARD, BOARD_V2, Eye
    from strong_player import benchmark_positions

    piece = PIECE_V2 if dataset.endswith("v2") else None
    presentation = BOARD_V2 if dataset.endswith("v2") else BOARD
    eye = Eye("fly")
    eye.model
    hits = 0
    for position, best in benchmark_positions(count):
        moves = ck.legal_moves(position)
        afterstates = [m.result for m in moves]
        images = [render_board(ck.to_array(a), piece=piece) for a in afterstates]
        seeds = [[int(v) for v in a] for a in afterstates]
        features = eye.look(images, seeds, presentation)[0].astype(np.float64)
        # The evaluator was trained on afterstates, so its score already means "good for the side
        # that just moved". No negation here -- that convention belongs to the value head, and
        # applying it anyway made this pick the worst move (5% against 16.8% for guessing).
        scores = decoder.predict(features) @ weights
        chosen = moves[int(np.argmax(scores))]
        hits += chosen.path[0] == best[0] and chosen.path[-1] == best[1]
    return hits / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="checkers-v2")
    parser.add_argument("--pairs", type=int, default=20000)
    parser.add_argument("--decoder-positions", type=int, default=20000)
    parser.add_argument("--agreement", type=int, default=200, help="benchmark positions to play out")
    args = parser.parse_args()

    started = time.perf_counter()
    train = extract_features.load_features("fly", args.dataset, "train")
    test = extract_features.load_features("fly", args.dataset, "test")
    train_contents = square_contents(positions_table("train"))[: len(train["vpn"])]
    test_contents = square_contents(positions_table("test"))[: len(test["vpn"])]

    n = min(args.decoder_positions, len(train["vpn"]))
    decoder = fit_decoder(train["vpn"][:n].astype(np.float64), train_contents[:n])
    decoded = decoder.predict(test["vpn"].astype(np.float64))
    squares = float((decoded.reshape(-1, 32, len(CLASSES)).argmax(2) == test_contents).mean())
    perfect = float((decoded.reshape(-1, 32, len(CLASSES)).argmax(2) == test_contents).all(axis=1).mean())
    table = confusion(decoded, test_contents)
    print(f"decoder on {args.dataset}: {squares:.3%} of squares, {perfect:.1%} of boards with no error "
          f"· {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    np.savez(DECODER, mean=decoder.mean, std=decoder.std, coef=decoder.coef,
             target_mean=decoder.target_mean, confusion=table)

    best, other = board_pairs("train", args.pairs, seed=0)
    rows = []
    for name, noise in [("clean boards", None), ("boards corrupted like the decoder", table)]:
        weights = fit_evaluator(best, other, noise)
        clean_rank = float(((one_hot(best) @ weights) > (one_hot(other) @ weights)).mean())
        oracle = float(np.mean([  # the same evaluator on true boards: isolates decoding loss
            np.argmax(one_hot(contents_of([m.result for m in ck.legal_moves(p)])) @ weights)
            == next(k for k, m in enumerate(ck.legal_moves(p))
                    if m.path[0] == b[0] and m.path[-1] == b[1])
            for p, b in __import__("strong_player").benchmark_positions(args.agreement)]))
        agreement = composed_agreement(decoder, weights, args.dataset, args.agreement) if args.agreement else float("nan")
        rows.append({"dataset": args.dataset, "evaluator": name, "decoder_squares": squares,
                     "decoder_perfect": perfect, "rank_on_true_boards": clean_rank,
                     "agreement_true_board": oracle, "teacher_agreement": agreement})
        print(f"  evaluator trained on {name:34s}: ranks true boards {clean_rank:.1%} · "
              f"picks teacher's move {agreement:.1%} through the eye, {oracle:.1%} given the true board",
              flush=True)
        np.savez(EVALUATOR if noise is not None else str(EVALUATOR).replace(".npz", "_clean.npz"),
                 weights=weights)
    pd.DataFrame(rows).to_csv(RESULTS, index=False)
    print(f"\nwrote {RESULTS}, {DECODER} and {EVALUATOR}")


if __name__ == "__main__":
    main()
