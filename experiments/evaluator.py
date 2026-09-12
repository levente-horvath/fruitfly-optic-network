"""Train the board evaluator, which is now the thing holding the player back.

    uv run python experiments/evaluator.py

Perception costs 2.5 points; the evaluator costs 9, plus whatever search would add if it were
calibrated. So this trains on *true* boards -- no simulation anywhere -- which makes an experiment
take seconds, and only the final model is composed with the decoder.

Two objectives at once, because the player needs both and neither alone gives both:
  * value regression against the teacher, which makes scores comparable ACROSS positions, the
    property minimax needs and a ranking loss never provides (sign agreement 49.8%, chance);
  * pairwise ranking among the afterstates of one position, which is what choosing a move needs.

Everything is scored for the side to move at that board, which is the convention fly_sim.search
expects, so a move is chosen by taking the afterstate the opponent likes least.

Writes data/processed/evaluator.csv and evaluator_best.npz.
"""

import argparse
import time

import numpy as np
import pandas as pd

from checkers_play import one_hot, positions_table, square_contents, squash
from fly_sim import checkers as ck, search
from fly_sim.connectome import CACHE_DIR
from fly_sim.trainable import Adam
from strong_player import benchmark_positions

RESULTS = CACHE_DIR / "evaluator.csv"
BEST = CACHE_DIR / "evaluator_best.npz"


def boards_of(positions) -> np.ndarray:
    return one_hot(square_contents(pd.DataFrame(positions, columns=["men", "kings", "opp_men", "opp_kings"])))


def training_data(split: str, pairs: int, seed: int = 0):
    """Positions with teacher values, and (teacher's afterstate, another afterstate) pairs."""
    table = positions_table(split)
    values = squash(table["value"].to_numpy())
    boards = one_hot(square_contents(table))

    rng = np.random.default_rng(seed)
    chosen, alternative = [], []
    labelled = table[table["best_from"] >= 0]
    for i in rng.permutation(len(labelled))[: pairs * 3]:
        row = labelled.iloc[i]
        position = ck.Position(int(row.men), int(row.kings), int(row.opp_men), int(row.opp_kings))
        moves = ck.legal_moves(position)
        if len(moves) < 2:
            continue
        best = next((m for m in moves if m.path[0] == row.best_from and m.path[-1] == row.best_to), None)
        if best is None:
            continue
        others = [m for m in moves if m is not best]
        chosen.append(best.result)
        alternative.append(others[rng.integers(len(others))].result)
        if len(chosen) >= pairs:
            break
    return boards, values, boards_of(chosen), boards_of(alternative)


class MLP:
    """160 -> hidden -> 1. Plain numpy; the model is far too small to need anything else."""

    def __init__(self, inputs: int, hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.hidden = hidden
        if hidden:
            self.W1 = rng.normal(size=(inputs, hidden)) * np.sqrt(2 / inputs)
            self.b1 = np.zeros(hidden)
            self.W2 = rng.normal(size=hidden) * np.sqrt(1 / hidden)
        else:
            self.W2 = np.zeros(inputs)
        self.b2 = np.zeros(())
        self.opt = [Adam(np.shape(p), 3e-3) for p in self.parameters()]

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2] if self.hidden else [self.W2, self.b2]

    def forward(self, x):
        if not self.hidden:
            return x @ self.W2 + self.b2, None
        h = np.maximum(x @ self.W1 + self.b1, 0)
        return h @ self.W2 + self.b2, h

    def backward(self, x, h, dout):
        """Gradients for one batch; dout is dLoss/dOutput per row."""
        if not self.hidden:
            return [x.T @ dout, np.array(dout.sum())]
        dW2 = h.T @ dout
        dh = np.outer(dout, self.W2) * (h > 0)
        return [x.T @ dh, dh.sum(axis=0), dW2, np.array(dout.sum())]

    def step(self, grads):
        updates = [opt(g) for opt, g in zip(self.opt, grads)]
        if self.hidden:
            self.W1 += updates[0]; self.b1 += updates[1]; self.W2 += updates[2]; self.b2 = self.b2 + updates[3]
        else:
            self.W2 += updates[0]; self.b2 = self.b2 + updates[1]

    def __call__(self, x):
        return self.forward(x)[0]


def train(model, boards, values, chosen, alternative, weight_rank, epochs, batch, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(boards))
        pair_order = rng.permutation(len(chosen))
        for k, s in enumerate(range(0, len(order), batch)):
            i = order[s : s + batch]
            out, h = model.forward(boards[i])
            dout = 2 * (out - values[i]) / len(i)                      # value regression
            grads = model.backward(boards[i], h, dout)

            if weight_rank:
                j = pair_order[(k * batch) % len(pair_order) :][:batch]
                if len(j):
                    # the teacher's afterstate should be the one the OPPONENT likes least
                    best_out, best_h = model.forward(chosen[j])
                    alt_out, alt_h = model.forward(alternative[j])
                    p = 1 / (1 + np.exp(-(best_out - alt_out)))
                    scale = weight_rank / len(j)
                    for g, add in zip(grads, model.backward(chosen[j], best_h, p * scale)):
                        g += add
                    for g, add in zip(grads, model.backward(alternative[j], alt_h, -p * scale)):
                        g += add
            model.step(grads)
    return model


def assess(model, positions, depths=(0, 1)):
    """Teacher agreement at each search depth, and whether scores mean anything across positions."""
    evaluate = lambda ps: model(boards_of(ps))  # noqa: E731  (value to the side to move)
    out = {}
    for depth in depths:
        hits = 0
        for position, best in positions:
            moves, scores = search.move_scores(position, evaluate, depth)
            chosen = moves[int(np.argmax(scores))]
            hits += chosen.path[0] == best[0] and chosen.path[-1] == best[1]
        out[f"depth_{depth}"] = hits / len(positions)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()

    boards, values, chosen, alternative = training_data("train", args.pairs)
    test = positions_table("test")
    test_boards, test_values = one_hot(square_contents(test)), squash(test["value"].to_numpy())
    positions = benchmark_positions(200)
    print(f"{len(boards):,} valued positions · {len(chosen):,} ranking pairs", flush=True)

    rows = []
    for hidden in (0, 256):
        for weight_rank in (0.0, 1.0):
            started = time.perf_counter()
            model = MLP(boards.shape[1], hidden)
            train(model, boards, values, chosen, alternative, weight_rank, args.epochs, args.batch)
            prediction = model(test_boards)
            r2 = 1 - float(((prediction - test_values) ** 2).mean() / test_values.var())
            sign = float((np.sign(prediction) == np.sign(test_values))[test_values != 0].mean())
            row = {"hidden": hidden, "rank_weight": weight_rank, "value_r2": r2,
                   "sign_across_positions": sign, **assess(model, positions),
                   "minutes": (time.perf_counter() - started) / 60}
            rows.append(row)
            print(f"hidden {hidden:3d} · rank weight {weight_rank:.1f}: R2 {r2:.3f} · "
                  f"sign {sign:.1%} · depth0 {row['depth_0']:.1%} · depth1 {row['depth_1']:.1%} · "
                  f"{row['minutes']:.1f} min", flush=True)
            pd.DataFrame(rows).to_csv(RESULTS, index=False)

    best = max(rows, key=lambda r: max(r["depth_0"], r["depth_1"]))
    print(f"\nbest: hidden {best['hidden']}, rank weight {best['rank_weight']} — "
          f"depth0 {best['depth_0']:.1%}, depth1 {best['depth_1']:.1%}")
    print("(hand-written evaluator: depth0 50.0%, depth1 55.0%)")


if __name__ == "__main__":
    main()


def listwise_data(split: str, limit: int, seed: int = 0):
    """Every legal afterstate of each position, flattened, with the teacher's choice marked.

    A pair teaches one comparison per position; the full list teaches all of them, which is the
    same simulation cost (none -- these are true boards) for roughly seven times the signal.
    """
    table = positions_table(split)
    table = table[table["best_from"] >= 0]
    rng = np.random.default_rng(seed)
    flat, owner, target, offset = [], [], [], 0
    for i in rng.permutation(len(table)):
        row = table.iloc[i]
        position = ck.Position(int(row.men), int(row.kings), int(row.opp_men), int(row.opp_kings))
        moves = ck.legal_moves(position)
        if len(moves) < 2:
            continue
        best = next((k for k, m in enumerate(moves) if m.path[0] == row.best_from and m.path[-1] == row.best_to), None)
        if best is None:
            continue
        flat += [m.result for m in moves]
        owner += [len(target)] * len(moves)
        target.append(offset + best)
        offset += len(moves)
        if len(target) >= limit:
            break
    return boards_of(flat), np.array(owner), np.array(target)


def train_listwise(model, boards, values, flat, owner, target, weight, epochs, batch=256, seed=0):
    """Value regression plus a softmax over each position's whole move list."""
    rng = np.random.default_rng(seed)
    groups = len(target)
    starts = np.searchsorted(owner, np.arange(groups))
    ends = np.append(starts[1:], len(owner))
    for _ in range(epochs):
        order = rng.permutation(len(boards))
        picks = rng.permutation(groups)
        for k, s in enumerate(range(0, len(order), batch)):
            i = order[s : s + batch]
            out, h = model.forward(boards[i])
            grads = model.backward(boards[i], h, 2 * (out - values[i]) / len(i))

            block = picks[(k * 32) % groups :][:32]
            if len(block):
                rows = np.concatenate([np.arange(starts[g], ends[g]) for g in block])
                out2, h2 = model.forward(flat[rows])
                score = -out2  # v is the opponent's value, so our preference is its negative
                dout = np.zeros(len(rows))
                at = 0
                for g in block:
                    n = ends[g] - starts[g]
                    exp = np.exp(score[at : at + n] - score[at : at + n].max())
                    p = exp / exp.sum()
                    p[target[g] - starts[g]] -= 1
                    dout[at : at + n] = -p * weight / len(block)  # chain through score = -out
                    at += n
                for g, add in zip(grads, model.backward(flat[rows], h2, dout)):
                    g += add
            model.step(grads)
    return model
