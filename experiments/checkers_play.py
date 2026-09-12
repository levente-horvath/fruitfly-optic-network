"""Train a value head on the cached features and let it play checkers.

    uv run python experiments/checkers_play.py --workers 4

The head is a ridge readout that scores a position for the side to move; to move, the player has
the engine list its legal moves, renders every resulting board, looks at it, and takes the move
whose afterstate the head likes least for the opponent. Nothing but the readout is trained, and
nothing but the move list comes from the engine, so the fly network and the null networks are
compared on exactly the same footing.

Writes data/processed/checkers_results.csv, checkers_matches.csv and checkers_results.png.
"""

import argparse
import itertools
import os
import random
import time
from multiprocessing import get_context

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import extract_features
from fly_sim import checkers as ck
from fly_sim import palette, probe
from fly_sim.connectome import CACHE_DIR
from fly_sim.stimulus import CHECKERS_POSITIONS, render_board
from fly_sim.vision import Eye

# name -> (network whose features to use, which feature set)
PLAYERS = {
    "fly": ("fly", "vpn"),
    "rewired": ("rewired", "vpn"),
    "random": ("random", "vpn"),
    "pixels": ("fly", "pixels"),  # the movies are seeded identically, so every network stores the same pixels
}
# Not a readout of the eye but the ceiling for the ones that are: the same ridge value head given
# the board itself, one-hot per square. Nothing that looks at a picture can beat what a linear
# function of the exact position can do.
REFERENCE = "board"
CLASSES = ["empty", "own man", "own king", "opponent man", "opponent king"]
VALUE_SCALE = 300.0  # a three-man advantage is most of the way to +-1 after the squash
TRAIN_SIZES = [300, 1000, 3000, 10000, 30000]
OPPONENTS = {"random": 0, "greedy (depth 1)": 1, "teacher (depth 2)": 2}
MATCH_GAMES = 40
AGREEMENT_POSITIONS = 1500
CACHE_LIMIT = 8000  # positions held as features per player, ~150 MB of visual projection rates

RESULTS = CACHE_DIR / "checkers_results.csv"
MATCHES = CACHE_DIR / "checkers_matches.csv"
FIGURE = CACHE_DIR / "checkers_results.png"


def load_features(network: str, split: str) -> dict:
    return extract_features.load_features(network, "checkers", split)


def square_contents(table: pd.DataFrame) -> np.ndarray:
    """Class of each playable square for every position in the table, shaped (positions, 32)."""
    contents = np.zeros((len(table), ck.SQUARES), dtype=np.int64)
    for value, column in enumerate(["men", "kings", "opp_men", "opp_kings"], start=1):
        masks = table[column].to_numpy(dtype=np.uint64)[:, None] >> np.arange(ck.SQUARES, dtype=np.uint64)
        contents[(masks & 1).astype(bool)] = value
    return contents


def one_hot(contents: np.ndarray) -> np.ndarray:
    return np.eye(len(CLASSES))[contents].reshape(len(contents), -1)


def positions_table(split: str) -> pd.DataFrame:
    """The positions behind the cached features, in the same order."""
    return pd.read_parquet(CHECKERS_POSITIONS).query("split == @split").reset_index(drop=True)


def squash(value: np.ndarray) -> np.ndarray:
    return np.tanh(np.asarray(value, dtype=np.float64) / VALUE_SCALE)


def board_seed(position: ck.Position) -> list[int]:
    """Jitter seeded by the position itself, so a position always looks the same and can be cached."""
    return [int(x) for x in position]


class ValuePlayer:
    """Scores every afterstate with the ridge head and plays the one the opponent will like least."""

    def __init__(self, encode, head: probe.Ridge):
        self.encode, self.head = encode, head
        self.cache: dict[ck.Position, np.ndarray] = {}

    def features(self, positions: list[ck.Position]) -> np.ndarray:
        if len(self.cache) > CACHE_LIMIT:
            self.cache.clear()  # before choosing what is missing, or the survivors are lost
        missing = [p for p in dict.fromkeys(positions) if p not in self.cache]
        for start in range(0, len(missing), 64):
            batch = missing[start : start + 64]
            self.cache.update(zip(batch, self.encode(batch)))
        return np.stack([self.cache[p] for p in positions])

    def values(self, positions: list[ck.Position]) -> np.ndarray:
        return self.head.predict(self.features(positions).astype(np.float64)).ravel()

    def __call__(self, batch: list[tuple[ck.Position, list[ck.Move]]]) -> list[ck.Move]:
        """Pick a move, resolving forced captures before asking the network what it thinks.

        Captures are compulsory in checkers, so a static score for an afterstate is close to
        meaningless when the opponent has a jump waiting. Evaluating the afterstate directly costs
        a perfect evaluator 45% against a random opponent versus 96% with this in place.
        """
        trees = [[quiescent_leaves(move.result) for move in moves] for _, moves in batch]
        leaves = [leaf for tree in trees for branch in tree for leaf, _ in branch]
        values = dict(zip(leaves, self.values(leaves))) if leaves else {}
        chosen = []
        for (_, moves), tree in zip(batch, trees):
            # each branch is (leaf, sign): sign is +1 when the leaf is scored for the same side
            own = [max(sign * values[leaf] for leaf, sign in branch) for branch in tree]
            chosen.append(moves[int(np.argmax([-v for v in own]))])
        return chosen


def quiescent_leaves(position: ck.Position, depth: int = 6) -> list[tuple[ck.Position, int]]:
    """Positions where the forced-capture sequence has run out, and whose side each is scored for.

    Returns (leaf, sign) pairs: the value of `position` to its own side to move is
    max over the pairs of sign * value(leaf), which mirrors the negamax the engine does.
    """
    moves = ck.legal_moves(position)
    if depth == 0 or not moves or not moves[0].captures:
        return [(position, 1)]
    out = []
    for move in moves:
        out += [(leaf, -sign) for leaf, sign in quiescent_leaves(move.result, depth - 1)]
    return out


def eye_encoder(network: str, key: str):
    """Renders each position, shows it to the eye and returns the chosen feature set."""
    eye = Eye(network)

    def encode(positions):
        images, seeds = [render_board(ck.to_array(p)) for p in positions], [board_seed(p) for p in positions]
        return eye.columns(images, seeds) if key == "pixels" else eye.look(images, seeds)[0]

    return encode


def board_encoder():
    """The reference readout: the exact position, one-hot per square, with no eye in the loop."""
    return lambda positions: one_hot(square_contents(pd.DataFrame(positions, columns=["men", "kings", "opp_men", "opp_kings"])))


def search_player(depth: int, rng: random.Random):
    """depth 0 plays uniformly at random; deeper plays the alpha-beta teacher."""

    def choose(batch):
        return [rng.choice(moves) if depth == 0 else ck.best_move(position, depth, rng)[0] for position, moves in batch]

    return choose


def play_matches(players, games: int, seed: int = 0) -> list[int | None]:
    """Run `games` games in lockstep so the network player scores a whole ply of boards at once.

    players[0] starts the even-numbered games and players[1] the odd ones. Returns the winner's
    index in `players` for each game, or None for a draw.
    """
    boards = [ck.START] * games
    first = [game % 2 for game in range(games)]  # which player moves first in each game
    winners: list[int | None] = [None] * games
    active = list(range(games))
    quiet = [0] * games

    for ply in range(200):
        if not active:
            break
        to_move = [(game, (first[game] + ply) % 2) for game in active]
        still, options = [], {}
        for game, side in to_move:
            moves = ck.legal_moves(boards[game])
            if not moves:
                winners[game] = 1 - side  # the side to move is trapped and loses
            else:
                still.append((game, side))
                options[game] = moves
        active = [game for game, _ in still]

        for side in (0, 1):
            batch = [(boards[game], options[game]) for game, s in still if s == side]
            if not batch:
                continue
            games_here = [game for game, s in still if s == side]
            for game, move in zip(games_here, players[side](batch)):
                quiet[game] = 0 if move.captures or move.promotes else quiet[game] + 1
                boards[game] = move.result
        active = [game for game in active if quiet[game] < 60]
    return winners


def move_agreement(player: ValuePlayer, table: pd.DataFrame) -> float:
    """How often the head's move matches the depth-6 teacher's, over positions with a real choice."""
    choices = []
    for row in table.itertuples():
        position = ck.Position(row.men, row.kings, row.opp_men, row.opp_kings)
        moves = ck.legal_moves(position)
        if len(moves) > 1:
            choices.append((position, moves, row.best_from, row.best_to))

    hits = 0
    for start in range(0, len(choices), 24):
        batch = choices[start : start + 24]
        for move, (_, _, best_from, best_to) in zip(player([(p, m) for p, m, _, _ in batch]), batch):
            hits += move.path[0] == best_from and move.path[-1] == best_to
    return hits / max(len(choices), 1)


SHORT = {"fly": "Fly", "pixels": "Pixels", "rewired": "Rewire", "random": "Random", REFERENCE: "Board"}


def plot(curves: pd.DataFrame, played: pd.DataFrame):
    palette.style(plt)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.2), facecolor=palette.SURFACE,
                             gridspec_kw={"width_ratios": [1.1, 0.85, 1.35]})

    # Learning curves. The three networks land on top of each other, so only the two lines that
    # separate get an end label; the tie is stated once instead of three times.
    reference = curves[curves["player"] == REFERENCE].sort_values("train_size")
    axes[0].plot(reference["train_size"], reference["r2"], color=palette.MUTED, linewidth=1.6,
                 linestyle=(0, (4, 3)), zorder=2, label="The board itself (ceiling)")
    for name, (color, label) in palette.SERIES.items():
        series = curves[curves["player"] == name].sort_values("train_size")
        axes[0].plot(series["train_size"], series["r2"], color=color, linewidth=2, marker="o", markersize=7,
                     markeredgecolor=palette.SURFACE, markeredgewidth=2, solid_capstyle="round", label=label, zorder=3)
    top = curves["train_size"].max()
    for name, text, offset in [(REFERENCE, "0.85  board itself", 0), ("pixels", "0.46  pixels", 0)]:
        value = curves.query("player == @name and train_size == @top")["r2"].iloc[0]
        axes[0].annotate(text, (top, value + offset), xytext=(9, 0), textcoords="offset points",
                         va="center", color=palette.SECONDARY, fontsize=8.5)
    tie = curves.query("train_size == @top and player in ['fly', 'rewired', 'random']")["r2"]
    axes[0].annotate(f"{tie.min():.2f}-{tie.max():.2f}\nall three networks", (top, tie.mean()),
                     xytext=(9, 0), textcoords="offset points", va="center", color=palette.SECONDARY, fontsize=8.5)

    ticks = [300, 1000, 3000, 10000, top]
    axes[0].set_xscale("log")
    axes[0].set_xticks(ticks, ["300", "1k", "3k", "10k", "40k"])
    axes[0].minorticks_off()
    axes[0].set_xlim(240, top * 4.6)
    axes[0].set_ylim(0, 0.95)
    axes[0].set_xlabel("Training positions (log scale)", color=palette.SECONDARY)
    axes[0].set_ylabel("Test $R^2$ against the teacher's value", color=palette.SECONDARY)
    axes[0].set_title("Reproducing the teacher's evaluation", fontsize=10.5, color=palette.PRIMARY, loc="left")
    palette.clean(axes[0])

    agreement = played[played["kind"] == "agreement"].set_index("player")["agreement"]
    names = [*palette.SERIES, REFERENCE]
    colors = [*(color for color, _ in palette.SERIES.values()), palette.GRID]
    axes[1].bar(range(len(names)), [agreement[n] * 100 for n in names], width=0.62, color=colors)
    for x, name in enumerate(names):
        axes[1].text(x, agreement[name] * 100 + 0.7, f"{agreement[name]:.0%}", ha="center",
                     color=palette.SECONDARY, fontsize=9)
    axes[1].set_xticks(range(len(names)), [SHORT[n] for n in names], color=palette.SECONDARY, fontsize=9)
    axes[1].set_ylim(0, 33)
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    axes[1].set_ylabel("Moves matching the teacher", color=palette.SECONDARY)
    axes[1].set_title("Agreement with the depth-6 teacher", fontsize=10.5, color=palette.PRIMARY, loc="left")
    palette.clean(axes[1])

    matches = played[played["kind"] == "match"].copy()
    matches["score"] = (matches["wins"] + matches["draws"] / 2) / matches["games"] * 100
    opponents, width = list(OPPONENTS), 0.17
    series = {**{n: c for n, (c, _) in palette.SERIES.items()}, REFERENCE: palette.GRID}
    for slot, (name, color) in enumerate(series.items()):
        scores = [matches.query("player == @name and opponent == @o")["score"].iloc[0] for o in opponents]
        offsets = np.arange(len(opponents)) + (slot - 2) * width
        axes[2].bar(offsets, scores, width=width * 0.88, color=color)
        for x, score in zip(offsets, scores):
            axes[2].text(x, score + 1.5, f"{score:.0f}", ha="center", color=palette.SECONDARY, fontsize=8)
    axes[2].axhline(50, color=palette.AXIS, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    axes[2].text(len(opponents) - 0.52, 52, "even", color=palette.MUTED, fontsize=8, ha="right")
    axes[2].set_xticks(range(len(opponents)), opponents, color=palette.SECONDARY)
    axes[2].set_ylim(0, 100)
    axes[2].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    games = int(matches["games"].iloc[0])
    axes[2].set_ylabel(f"Match score, {games} games (draw = \u00bd)", color=palette.SECONDARY)
    axes[2].set_title("Playing strength against fixed opponents", fontsize=10.5, color=palette.PRIMARY, loc="left")
    palette.clean(axes[2])

    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 2, 3, 4, 0]  # the four readouts first, the ceiling last
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left", ncol=5, frameon=False,
               labelcolor=palette.SECONDARY, fontsize=9, bbox_to_anchor=(0.006, 0.875), handletextpad=0.6, columnspacing=1.6)
    fig.suptitle("A frozen fly optic lobe playing checkers", x=0.006, y=0.975, ha="left",
                 color=palette.PRIMARY, fontsize=13, fontweight="semibold")
    fig.text(0.006, 0.917, "Every readout is a ridge regression on frozen features, trained only to copy an alpha-beta "
             "teacher's position values. Moves come from scoring each legal afterstate.",
             color=palette.SECONDARY, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(FIGURE, dpi=130, facecolor=palette.SURFACE)


_player_cache: tuple[str, ValuePlayer] | None = None


def _player(name: str, head: probe.Ridge) -> ValuePlayer:
    """Hold one player at a time: a worker that kept an Eye per network would carry three copies
    of the network and its rate model, which is what tips this machine into swap."""
    global _player_cache
    if _player_cache is None or _player_cache[0] != name:
        encode = board_encoder() if name == REFERENCE else eye_encoder(*PLAYERS[name])
        _player_cache = (name, ValuePlayer(encode, head))
    return _player_cache[1]


def run_task(task):
    """One unit of evaluation work: a move-agreement check or one match against a fixed opponent."""
    kind, name, head, payload = task
    start = time.perf_counter()
    player = _player(name, head)
    if kind == "agreement":
        outcome = {"agreement": move_agreement(player, payload)}
    else:
        opponent, depth, games = payload
        winners = play_matches([player, search_player(depth, random.Random(f"{name} vs {opponent}"))], games)
        outcome = {
            "opponent": opponent,
            "wins": sum(w == 0 for w in winners),
            "draws": sum(w is None for w in winners),
            "losses": sum(w == 1 for w in winners),
            "games": games,
        }
    return {"kind": kind, "player": name, **outcome, "minutes": (time.perf_counter() - start) / 60}


def feature_sets():
    """Yield (name, train features, train targets, test features, test targets), one readout at a time.

    A network's features are 40,000 x 4,611 in double precision, so they are loaded and dropped one
    at a time rather than all held at once.
    """
    for name, (network, key) in PLAYERS.items():
        train, test = load_features(network, "train"), load_features(network, "test")
        yield (name, train[key].astype(np.float64), squash(train["labels"])[:, None],
               test[key].astype(np.float64), squash(test["labels"])[:, None])
    tables = {split: positions_table(split) for split in ("train", "test")}
    yield (REFERENCE,
           one_hot(square_contents(tables["train"])), squash(tables["train"]["value"].to_numpy())[:, None],
           one_hot(square_contents(tables["test"])), squash(tables["test"]["value"].to_numpy())[:, None])


def fit_heads(seed: int = 0):
    """Fit the value head for every readout at a range of training-set sizes."""
    rows, heads = [], {}
    for name, x, y, x_test, y_test in feature_sets():
        for size in [*TRAIN_SIZES, len(x)]:
            start = time.perf_counter()
            chosen = np.sort(np.random.default_rng(seed).choice(len(x), size, replace=False)) if size < len(x) else slice(None)
            head = probe.fit(x[chosen], y[chosen])
            prediction = head.predict(x_test)
            rows.append({
                "player": name,
                "train_size": size,
                "features": x.shape[1],
                "r2": 1 - float(((prediction - y_test) ** 2).mean() / y_test.var()),
                "sign_accuracy": float((np.sign(prediction.ravel()) == np.sign(y_test.ravel()))[y_test.ravel() != 0].mean()),
                "ridge_scale": head.scale,
            })
            print(f"{name:8s} n={size:>6,}: test R2 {rows[-1]['r2']:.3f} · sign {rows[-1]['sign_accuracy']:.1%} · "
                  f"ridge {head.scale:.0e} · {time.perf_counter() - start:.0f}s", flush=True)
            if size == len(x):
                heads[name] = head
        del x, y, x_test, y_test  # drop this readout's features before the next one is loaded
    return pd.DataFrame(rows), heads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--games", type=int, default=MATCH_GAMES)
    parser.add_argument("--skip-play", action="store_true", help="only fit the heads and score the values")
    parser.add_argument("--plot-only", action="store_true", help="redraw the figure from the saved results")
    args = parser.parse_args()
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[var] = "1"

    if args.plot_only:
        plot(pd.read_csv(RESULTS), pd.read_csv(MATCHES))
        print(f"wrote {FIGURE}")
        return

    curves, heads = fit_heads()
    curves.to_csv(RESULTS, index=False)
    if args.skip_play:
        return

    test_table = positions_table("test")
    sample = test_table.sample(min(AGREEMENT_POSITIONS, len(test_table)), random_state=0)
    readouts = [*PLAYERS, REFERENCE]
    tasks = [("agreement", name, heads[name], sample) for name in readouts]
    tasks += [
        ("match", name, heads[name], (opponent, depth, args.games))
        for name, (opponent, depth) in itertools.product(readouts, OPPONENTS.items())
    ]

    start, results = time.perf_counter(), []
    with get_context("spawn").Pool(args.workers) as pool:
        for done, row in enumerate(pool.imap_unordered(run_task, tasks), start=1):
            results.append(row)
            label = row.get("opponent", "teacher move agreement")
            score = row["agreement"] if row["kind"] == "agreement" else (row["wins"] + row["draws"] / 2) / row["games"]
            print(f"  [{done}/{len(tasks)}] {row['player']:8s} vs {label:22s} {score:.1%} · {row['minutes']:.1f} min · "
                  f"{(time.perf_counter() - start) / 60:.0f} min elapsed", flush=True)

    played = pd.DataFrame(results)
    played.to_csv(MATCHES, index=False)
    curves = curves.merge(played[played["kind"] == "agreement"][["player", "agreement"]], on="player", how="left")
    curves.to_csv(RESULTS, index=False)

    summary = played[played["kind"] == "match"].copy()
    summary["score"] = (summary["wins"] + summary["draws"] / 2) / summary["games"]
    print("\n" + summary.pivot_table(index="player", columns="opponent", values="score").mul(100).round(1).to_string())
    plot(curves, played)
    print(f"\nwrote {RESULTS}, {MATCHES} and {FIGURE}")


if __name__ == "__main__":
    main()
