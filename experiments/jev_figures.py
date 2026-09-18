"""Figures for the Jev README, from what experiments/jev_checkers.py recorded.

    uv run python experiments/jev_figures.py

docs/jev_scores.png  match scores against the engines, next to the fly's; and Jev vs the fly
docs/jev_games.png   pieces left on each side through the two recorded games
docs/jev_habits.png  how often a move hands the opponent a capture; which listed option Jev takes
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fly_sim import checkers as ck
from fly_sim import palette
from fly_sim.connectome import CACHE_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_checkers import move_label  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs"
JEV, HINTS, FLY = "#eb6834", "#1baf7a", "#2a78d6"
DRAW = palette.AXIS


def gives_capture(move: ck.Move) -> bool:
    replies = ck.legal_moves(move.result)
    return bool(replies) and bool(replies[0].captures)


def label_bar(ax, x, height, text, **kw):
    ax.annotate(text, (x, height), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                fontsize=9, color=palette.SECONDARY, **kw)


def scores_figure(results: pd.DataFrame):
    fly = pd.read_csv(CACHE_DIR / "strong_player_matches.csv")["score"].to_numpy() * 100
    plain = results[~results.hints].set_index("opponent")
    hinted = results[results.hints].set_index("opponent")
    opponents = ["random", "depth1", "depth2"]
    names = ["random mover", "1-move lookahead", "2-move lookahead"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=palette.SURFACE,
                                      gridspec_kw={"width_ratios": [1.5, 1]})
    width = 0.26
    x = np.arange(3)
    for offset, values, color, label in [
        (-width, plain.loc[opponents, "score"].to_numpy() * 100, JEV, "Jev"),
        (0, hinted.loc[opponents, "score"].to_numpy() * 100, HINTS, "Jev + capture hints"),
        (width, fly, FLY, "The fly"),
    ]:
        # 2px surface gap between neighbouring bars
        left.bar(x + offset, values, width - 0.02, color=color, label=label, edgecolor=palette.SURFACE, linewidth=1)
        for xi, v in zip(x + offset, values):
            label_bar(left, xi, v, f"{v:.0f}")
    palette.clean(left)
    left.set_xticks(x, names, color=palette.SECONDARY)
    left.set_ylim(0, 112)
    left.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    left.set_title("Score against engines (win 1, draw ½), 20–40 games each", loc="left", fontsize=11,
                   color=palette.PRIMARY)
    left.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=palette.SECONDARY)

    row = plain.loc["fly"]
    parts = [("Jev wins", row.wins, JEV), ("draws", row.draws, DRAW), ("fly wins", row.losses, FLY)]
    start = 0
    for name, n, color in parts:
        right.barh(0, n, left=start, height=0.5, color=color, edgecolor=palette.SURFACE, linewidth=2)
        if n:
            right.text(start + n / 2, 0.42, f"{name}\n{int(n)}", ha="center", va="bottom", fontsize=9,
                       color=palette.SECONDARY)
        start += n
    right.set_xlim(0, row.games)
    right.set_ylim(-0.6, 0.9)
    right.set_yticks([])
    right.set_xticks([0, 5, 10, 15, 20])
    palette.clean(right, spines=("top", "right", "left"))
    right.grid(False)
    right.set_title(f"Jev vs the fly, {int(row.games)} games", loc="left", fontsize=11, color=palette.PRIMARY)
    right.text(0, -0.55, "draw = 60 moves in a row with no capture or crowning", fontsize=8.5, color=palette.MUTED)
    fig.tight_layout()
    fig.savefig(DOCS / "jev_scores.png", dpi=160, facecolor=palette.SURFACE)


def habits_figure(calls: list[dict]):
    """Blunder rate against what a random pick and a small engine would do on the same positions."""
    rows = []
    for call in calls:
        position = ck.Position(*call["position"])
        by_label = {move_label(m): m for m in ck.legal_moves(position)}
        chosen = by_label[call["choice"]]
        options = [by_label[label] for label in call["options"]]
        rows.append({
            "hints": call["hints"], "position": position,
            "jev": gives_capture(chosen),
            "random": np.mean([gives_capture(m) for m in options]),
            "index": call["options"].index(call["choice"]), "n": len(options),
        })
    table = pd.DataFrame(rows)
    sample = table[~table.hints].sample(min(600, (~table.hints).sum()), random_state=0)
    engine = np.mean([gives_capture(ck.best_move(p, 2)[0]) for p in sample["position"]])

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.4), facecolor=palette.SURFACE)
    bars = [
        ("random pick", table[~table.hints]["random"].mean(), palette.MUTED),
        ("Jev", table[~table.hints]["jev"].mean(), JEV),
        ("Jev + hints", table[table.hints]["jev"].mean(), HINTS),
        ("2-move engine", engine, palette.SECONDARY),
    ]
    for i, (name, value, color) in enumerate(bars):
        left.bar(i, value * 100, 0.6, color=color, edgecolor=palette.SURFACE, linewidth=1)
        label_bar(left, i, value * 100, f"{value:.0%}")
    palette.clean(left)
    left.set_xticks(range(len(bars)), [b[0] for b in bars], color=palette.SECONDARY)
    left.set_ylim(0, max(b[1] for b in bars) * 100 * 1.12)
    left.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    left.set_title("Moves that let the opponent capture straight away", loc="left", fontsize=11,
                   color=palette.PRIMARY)

    plain = table[~table.hints]
    ks = np.arange(6)
    picked = [(plain["index"] == k).mean() * 100 for k in ks]
    chance = [np.mean(np.where(plain["n"] > k, 1 / plain["n"], 0)) * 100 for k in ks]
    right.bar(ks - 0.18, picked, 0.34, color=JEV, label="Jev picks it", edgecolor=palette.SURFACE, linewidth=1)
    right.bar(ks + 0.18, chance, 0.34, color=palette.AXIS, label="picking at random", edgecolor=palette.SURFACE,
              linewidth=1)
    for k in ks:
        label_bar(right, k - 0.18, picked[k], f"{picked[k]:.0f}")
    palette.clean(right)
    right.set_xticks(ks, [f"{k + 1}{'st' if k == 0 else 'nd' if k == 1 else 'rd' if k == 2 else 'th'}"
                          for k in ks], color=palette.SECONDARY)
    right.set_ylim(0, max(picked) * 1.15)
    right.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    right.set_title("Which listed option Jev takes (moves listed by starting square)", loc="left", fontsize=11,
                    color=palette.PRIMARY)
    right.legend(frameon=False, fontsize=9, labelcolor=palette.SECONDARY)
    fig.tight_layout()
    fig.savefig(DOCS / "jev_habits.png", dpi=160, facecolor=palette.SURFACE)
    return {name: value for name, value, _ in bars}, picked, chance


def game_figure(paths: list[Path]):
    """Pieces left on each side, move by move, in the recorded replay games."""
    fig, axes = plt.subplots(1, len(paths), figsize=(12.5, 3.6), facecolor=palette.SURFACE, sharey=True)
    for ax, path in zip(np.atleast_1d(axes), paths):
        game = json.loads(path.read_text())
        boards = [p["board"] for p in game["plies"]] + [game["final"]]
        plies = np.arange(len(boards))
        for codes, color, name in [((1, 2), JEV, "Jev"), ((3, 4), FLY, "Fly")]:
            counts = [sum(c in codes for c in b) for b in boards]
            ax.step(plies, counts, where="post", color=color, linewidth=2)
            ax.annotate(f"{name} {counts[-1]}", (plies[-1], counts[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=palette.SECONDARY)
        palette.clean(ax)
        ax.set_xlim(0, len(boards) + 14)
        ax.set_ylim(0, 13)
        ax.set_xlabel("move", color=palette.MUTED, fontsize=9)
        first = "Jev" if game["order"][0] == "jev" else "the fly"
        result = {"jev": "Jev wins", "fly": "the fly wins"}.get(game["winner"], "draw")
        ax.set_title(f"{first} moves first — {result} after {len(game['plies'])} moves", loc="left", fontsize=11,
                     color=palette.PRIMARY)
    np.atleast_1d(axes)[0].set_ylabel("pieces left", color=palette.MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(DOCS / "jev_games.png", dpi=160, facecolor=palette.SURFACE)


def main():
    palette.style(plt)
    game_figure([CACHE_DIR / "jev_replay_a.json", CACHE_DIR / "jev_replay_b.json"])
    results = pd.read_csv(CACHE_DIR / "jev_checkers.csv")
    calls = [json.loads(line) for line in open(CACHE_DIR / "jev_checkers_calls.jsonl")]
    scores_figure(results)
    blunders, picked, chance = habits_figure(calls)
    print({k: round(v, 3) for k, v in blunders.items()})
    print("picked", np.round(picked, 1), "chance", np.round(chance, 1))
    print(f"{len(calls):,} Jev calls analysed; wrote docs/jev_scores.png, docs/jev_habits.png")


if __name__ == "__main__":
    main()
