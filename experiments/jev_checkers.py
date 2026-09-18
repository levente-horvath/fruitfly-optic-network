"""Jev plays checkers: first against the engines, then against the fly.

    uv sync --extra jev                                      # TYPESAFE_API_KEY in .env or the environment
    uv run python experiments/jev_checkers.py estimate --games 20
    uv run python experiments/jev_checkers.py --max-calls 5000 run --games 20 --opponents random depth1 depth2 fly
    uv run python experiments/jev_checkers.py run --offline  # no API calls: Jev replaced by a random pick

Jev (TypeSafe AI, https://typesafe.ai) does not write text; it picks one option out of a list. So
each turn the code lists every legal move -- captures are compulsory, so when one exists only the
captures are listed -- and Jev picks one as a `Choice`. It sees the board as a grid, where every
piece stands, and what each move does right now (moves, captures, crowns). With --hints each option
also says what the opponent can capture straight after it; that is one ply of lookahead done by the
code, so it is a separate condition.

The fly is the strongest player the project built (experiments/play_server.py): each afterstate is
drawn, shown to the right optic lobe for 0.6 s, the board decoded back off the visual projection
neurons and judged by a linear evaluator, with forced captures played out first.

Squares are numbered 1-32 from the side to move's point of view, so Jev always plays "up the
board" no matter which colour it is.
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkers_play import board_seed, play_matches, search_player
from fly_sim import checkers as ck
from fly_sim import probe, search
from fly_sim.connectome import CACHE_DIR

RESULTS = CACHE_DIR / "jev_checkers.csv"
REPLAY_TEMPLATE = Path(__file__).resolve().parents[1] / "viz" / "jev_replay.html"
REPLAY_PAGE = Path(__file__).resolve().parents[1] / "docs" / "jev_replay.html"
CALLS_LOG = CACHE_DIR / "jev_checkers_calls.jsonl"
ROOT = Path(__file__).resolve().parents[1]

PRICE_PER_MTOK = 0.042  # measured in jev_controller: ~1,000 input tokens a call, outputs free
TOKENS_PER_CALL = 1500  # a checkers state is bigger than a CartPole state; a planning figure
PLIES_PER_GAME = 60  # Jev's share of a typical game, with some margin

INSTRUCTIONS = (
    "You are playing checkers (American draughts, 8x8). Your pieces move up the board, toward row 8; "
    "the opponent's move down. Men move one square diagonally forward; a man that reaches row 8 is crowned "
    "and becomes a king, which moves diagonally both ways. Capturing is compulsory: if any capture exists "
    "you must capture, and a capture continues while further jumps are available with the same piece. "
    "You win when the opponent has no pieces or no legal move. Every option is a legal move. Pick the move "
    "most likely to win the game: keep your pieces safe, win material, and crown men when you can."
)


def square_name(square: int) -> int:
    """1-32, counted from the far-left corner of the side to move's crowning row."""
    return square + 1


def move_label(move: ck.Move) -> str:
    sep = "x" if move.captures else "-"
    return sep.join(str(square_name(s)) for s in move.path)


def grid(position: ck.Position) -> list[str]:
    """Rows 8 down to 1 as seen by the side to move: x/X yours, o/O the opponent's, . empty dark square."""
    board = ck.to_array(position)  # row 0 is where the side to move crowns: row 8 here
    symbols = {0: ".", 1: "x", 2: "X", -1: "o", -2: "O"}
    lines = []
    for r in range(8):
        cells = [symbols[int(board[r, c])] if (r + c) % 2 == 1 else " " for c in range(8)]
        lines.append(f"{8 - r} " + " ".join(cells))
    lines.append("  a b c d e f g h")
    return lines


def where(mask: int) -> list[int]:
    return [square_name(s) for s in ck.bits(mask)]


def describe(move: ck.Move, position: ck.Position, hints: bool) -> str:
    king = bool(position.kings >> move.path[0] & 1)
    piece = "king" if king else "man"
    text = f"{piece} from {square_name(move.path[0])} to {square_name(move.path[-1])}"
    taken = bin(move.captures).count("1")
    if taken:
        text += f", captures {taken} piece{'s' * (taken > 1)}"
    if move.promotes:
        text += ", crowns a king"
    if hints:
        replies = ck.legal_moves(move.result)
        if not replies:
            text += "; the opponent then has no move and loses"
        elif replies[0].captures:
            most = max(bin(r.captures).count("1") for r in replies)
            text += f"; the opponent must then capture, taking up to {most} of your pieces"
        else:
            text += "; the opponent cannot capture straight after"
    return text


def build_state(position: ck.Position) -> dict:
    count = lambda mask: bin(mask).count("1")  # noqa: E731
    return {
        "board": grid(position),
        "your_men": where(position.men),
        "your_kings": where(position.kings),
        "opponent_men": where(position.opp_men),
        "opponent_kings": where(position.opp_kings),
        "material": f"you {count(position.men)} men + {count(position.kings)} kings, "
        f"opponent {count(position.opp_men)} men + {count(position.opp_kings)} kings",
        "square_numbering": "squares 1-4 are row 8 (your crowning row), 29-32 are row 1 (your back row)",
    }


class LimitReached(RuntimeError):
    pass


class JevPlayer:
    """One `Choice` call per move. Calls within a batch of games go out in parallel."""

    def __init__(self, client, model: str, hints: bool, max_calls: int, log, timeout_s: float = 10.0):
        self.client, self.model, self.hints = client, model, hints
        self.max_calls, self.log, self.timeout_s = max_calls, log, timeout_s
        self.calls = self.fallbacks = 0
        self.pool = ThreadPoolExecutor(8)
        self.opponent = ""

    def pick(self, position: ck.Position, moves: list[ck.Move]) -> ck.Move:
        return self.ask(position, moves)[0]

    def ask(self, position: ck.Position, moves: list[ck.Move]) -> tuple[ck.Move, dict]:
        """The chosen move, and Jev's probability for every option."""
        if len(moves) == 1:
            return moves[0], {move_label(moves[0]): 1.0}
        from typesafe_sdk import Choice, TypeSafeError

        options = {move_label(m): describe(m, position, self.hints) for m in moves}
        question = Choice(instructions=INSTRUCTIONS, criteria=options)
        started = time.perf_counter()
        try:
            response = self.client.system_one(
                state=build_state(position),
                questions={"move": question},
                model=self.model,
                timeout=self.timeout_s,
            )
            answer = response.choices["move"]
            label, confidence, probabilities = answer.choice, answer.confidence, dict(answer.probabilities)
        except TypeSafeError as error:
            # keep the game going on the first listed move, and count it so it can be reported
            self.fallbacks += 1
            label, confidence, probabilities = next(iter(options)), None, {"error": str(error)[:200]}
        self.log.write(json.dumps({
            "opponent": self.opponent, "hints": self.hints, "position": list(position),
            "options": list(options), "choice": label, "confidence": confidence,
            "probabilities": probabilities, "seconds": round(time.perf_counter() - started, 3),
        }) + "\n")
        return next(m for m in moves if move_label(m) == label), probabilities

    def __call__(self, batch):
        needed = sum(len(moves) > 1 for _, moves in batch)
        if self.calls + needed > self.max_calls:
            raise LimitReached(f"call cap of {self.max_calls} reached; raise --max-calls to continue")
        self.calls += needed
        return list(self.pool.map(lambda item: self.pick(*item), batch))


class OfflineJev(JevPlayer):
    """Stand-in with no API: picks uniformly at random, to check the loop end to end."""

    def __init__(self, hints: bool, log):
        super().__init__(None, "offline", hints, 10**9, log)
        self.rng = random.Random(0)

    def ask(self, position, moves):
        return self.rng.choice(moves), {move_label(m): 1 / len(moves) for m in moves}


class StrongHead:
    """The play_server evaluator as a `head`: decode the board off the neurons, round it, judge it."""

    def __init__(self):
        d = np.load(CACHE_DIR / "strong_decoder.npz")
        self.decoder = probe.Ridge(d["mean"], d["std"], d["coef"], d["target_mean"], 0.0, 0.0)
        self.weights = np.load(CACHE_DIR / "strong_evaluator.npz")["weights"]

    def predict(self, features: np.ndarray) -> np.ndarray:
        board = self.decoder.predict(features).reshape(len(features), 32, 5).argmax(2)
        # the evaluator scores a board for whoever just moved; negate for the side to move
        return -(np.eye(5)[board].reshape(len(features), -1) @ self.weights)


def fly_player():
    from fly_sim.stimulus import PIECE_V2, render_board
    from fly_sim.vision import BOARD_V2, Eye
    from strong_player import SearchPlayer

    eye = Eye("fly")

    def encode(positions):
        images = [render_board(ck.to_array(p), piece=PIECE_V2) for p in positions]
        return eye.look(images, [board_seed(p) for p in positions], BOARD_V2)[0].astype(np.float64)

    return SearchPlayer(encode, StrongHead(), depth=0)


def opponent(name: str, seed: int):
    if name == "fly":
        return fly_player()
    if name == "random":
        return search_player(0, random.Random(seed))
    return search_player(int(name.removeprefix("depth")), random.Random(seed))


def board_codes(position: ck.Position, mover: str, flipped: bool) -> list[int]:
    """The 32 squares in the replay's fixed frame: 0 empty, 1/2 Jev man/king, 3/4 fly man/king."""
    codes = [0] * ck.SQUARES
    own, opp = (1, 3) if mover == "jev" else (3, 1)
    for mask, code in [(position.men, own), (position.kings, own + 1),
                       (position.opp_men, opp), (position.opp_kings, opp + 1)]:
        for s in ck.bits(mask):
            codes[31 - s if flipped else s] = code
    return codes


def record_game(jev: JevPlayer, jev_first: bool, max_plies: int = 200) -> dict:
    """One Jev vs fly game with every decision kept, for viz/jev_replay.html.

    The replay's frame is the first mover's: its men start on squares 20-31 and crown on 0-3.
    Positions are canonical for the side to move, so every odd ply is rotated back into that frame.
    """
    fly = fly_player()
    order = ["jev", "fly"] if jev_first else ["fly", "jev"]
    position, quiet, plies, winner = ck.START, 0, [], None
    to_abs = lambda s, flipped: 31 - s if flipped else s  # noqa: E731
    for ply in range(max_plies):
        mover, flipped = order[ply % 2], ply % 2 == 1
        moves = ck.legal_moves(position)
        if not moves:
            winner = order[1 - ply % 2]
            break
        entry = {"mover": mover, "board": board_codes(position, mover, flipped)}
        if mover == "jev":
            move, probabilities = jev.ask(position, moves)
            options = [{"path": [to_abs(s, flipped) for s in m.path], "p": float(probabilities.get(move_label(m), 0))}
                       for m in moves]
        else:
            moves, scores = search.move_scores(position, fly.evaluate, depth=0)
            move = moves[int(np.argmax(scores))]
            options = [{"path": [to_abs(s, flipped) for s in m.path], "score": float(v)} for m, v in zip(moves, scores)]
            # what the eye read off the board it chose, in the same codes as the true board
            seen = fly.head.decoder.predict(fly.encode([move.result])).reshape(32, 5).argmax(1)
            after = ck.Position(*[sum(1 << s for s in range(32) if seen[s] == k) for k in (1, 2, 3, 4)])
            entry["fly_saw"] = board_codes(after, "jev", not flipped)
        entry.update(path=[to_abs(s, flipped) for s in move.path], options=options,
                     captured=[to_abs(s, flipped) for s in ck.bits(move.captures)])
        plies.append(entry)
        print(f"{ply + 1:3d} {mover}: {move_label(move)}", flush=True)
        quiet = 0 if move.captures or move.promotes else quiet + 1
        position = move.result
        if quiet >= 60:
            break
    final_flipped = len(plies) % 2 == 1
    return {"order": order, "winner": winner, "plies": plies,
            "final": board_codes(position, order[len(plies) % 2], final_flipped),
            "model": jev.model, "hints": jev.hints}


def write_page(game: dict, path: Path = REPLAY_PAGE):
    """A standalone replay page: the viz template with the game embedded."""
    data = json.dumps(game, separators=(",", ":")).replace("</", "<\\/")
    path.write_text(REPLAY_TEMPLATE.read_text().replace("__GAME__", data))
    print(f"wrote {path}")


def load_dotenv():
    for path in (ROOT / ".env",):
        if path.exists():
            for line in path.read_text().splitlines():
                key, sep, value = line.partition("=")
                if sep and not key.strip().startswith("#") and value.strip():
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def estimate(games: int, opponents: list[str]) -> str:
    calls = games * len(opponents) * PLIES_PER_GAME
    cost = calls * TOKENS_PER_CALL * PRICE_PER_MTOK / 1e6
    return f"at most ~{calls:,} Jev calls for {games} games x {len(opponents)} opponents; ~${cost:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-calls", type=int, default=0, help="hard cap on paid calls; 0 blocks paid calls")
    parser.add_argument("--model", default="jev-latest")
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("replay", help="record one Jev vs fly game for viz/jev_replay.html")
    replay.add_argument("--fly-first", action="store_true")
    replay.add_argument("--hints", action="store_true")
    replay.add_argument("--offline", action="store_true")
    replay.add_argument("--out", type=Path, default=CACHE_DIR / "jev_replay.json")
    page = sub.add_parser("page", help="rebuild docs/jev_replay.html from a recorded game")
    page.add_argument("game", type=Path, nargs="?", default=CACHE_DIR / "jev_replay.json")
    for name in ("estimate", "run"):
        p = sub.add_parser(name)
        p.add_argument("--games", type=int, default=20, help="per opponent; Jev moves first in half")
        p.add_argument("--opponents", nargs="+", default=["random", "depth1", "depth2", "fly"])
        p.add_argument("--hints", action="store_true", help="tell Jev what the opponent can capture after each move")
        p.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.command == "page":
        return write_page(json.loads(args.game.read_text()))
    if args.command != "replay":
        print(estimate(args.games, args.opponents))
    if args.command == "estimate":
        return

    log = CALLS_LOG.open("a")
    if args.offline:
        jev = OfflineJev(args.hints, log)
    else:
        if args.max_calls <= 0:
            sys.exit("paid calls blocked: pass --max-calls N (see `estimate`), or use --offline")
        load_dotenv()
        if not os.environ.get("TYPESAFE_API_KEY"):
            sys.exit("TYPESAFE_API_KEY is not set (put it in .env), or use --offline")
        from typesafe_sdk import RetryPolicy, TypeSafeClient

        jev = JevPlayer(TypeSafeClient(retry=RetryPolicy(max_retries=1)), args.model, args.hints, args.max_calls, log)

    if args.command == "replay":
        jev.opponent = "fly (replay)"
        game = record_game(jev, jev_first=not args.fly_first)
        args.out.write_text(json.dumps(game))
        print(f"winner: {game['winner'] or 'draw'} after {len(game['plies'])} plies; wrote {args.out}")
        write_page(game)
        return

    rows = pd.read_csv(RESULTS).to_dict("records") if RESULTS.exists() else []
    for name in args.opponents:
        jev.opponent, jev.fallbacks = name, 0
        started, calls_before = time.perf_counter(), jev.calls
        winners = play_matches([jev, opponent(name, seed=5)], args.games)
        wins, draws = sum(w == 0 for w in winners), sum(w is None for w in winners)
        losses = args.games - wins - draws
        row = {
            "player": "offline" if args.offline else args.model, "hints": args.hints, "opponent": name,
            "games": args.games, "wins": wins, "draws": draws, "losses": losses,
            "score": (wins + draws / 2) / args.games, "calls": jev.calls - calls_before,
            "fallbacks": jev.fallbacks, "minutes": round((time.perf_counter() - started) / 60, 2),
        }
        print(f"vs {name}: {wins}W {draws}D {losses}L = {row['score']:.1%} · {row['calls']} calls · "
              f"{row['fallbacks']} fallbacks · {row['minutes']} min", flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(RESULTS, index=False)
    log.close()
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
