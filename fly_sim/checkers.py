"""American checkers: rules, and an alpha-beta teacher that supplies training targets.

Positions are always stored from the point of view of the side to move, whose pieces advance
toward row 0 and crown there. Making a move rotates the board 180 degrees and swaps the sides,
so every position the network ever sees is in the same canonical frame.
"""

from __future__ import annotations

import random
from typing import NamedTuple

SQUARES = 32
FULL = (1 << SQUARES) - 1
CROWN_ROW = 0b1111  # squares 0-3, where a man of the side to move is promoted
MAN_DIRS = (0, 1)  # forward-left, forward-right
KING_DIRS = (0, 1, 2, 3)

_REV8 = [int(f"{i:08b}"[::-1], 2) for i in range(256)]


class Position(NamedTuple):
    """Bitboards of the side to move and of its opponent."""

    men: int
    kings: int
    opp_men: int
    opp_kings: int

    @property
    def own(self) -> int:
        return self.men | self.kings

    @property
    def opp(self) -> int:
        return self.opp_men | self.opp_kings


class Move(NamedTuple):
    path: tuple[int, ...]  # squares visited, starting square first
    captures: int  # bitboard of the opponent pieces taken
    result: Position  # the position after the move, seen by the opponent
    promotes: bool


START = Position(men=0xFFF00000, kings=0, opp_men=0xFFF, opp_kings=0)


def square_rowcol(square: int) -> tuple[int, int]:
    """Row and column on the 8x8 board; only the dark squares are playable."""
    row = square // 4
    return row, 2 * (square % 4) + (1 - row % 2)


def _build_tables():
    index = {square_rowcol(s): s for s in range(SQUARES)}
    steps = [[-1] * 4 for _ in range(SQUARES)]
    jumps: list[list[tuple[int, int] | None]] = [[None] * 4 for _ in range(SQUARES)]
    for s in range(SQUARES):
        row, col = square_rowcol(s)
        for d, (dr, dc) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
            over = index.get((row + dr, col + dc))
            land = index.get((row + 2 * dr, col + 2 * dc))
            if over is not None:
                steps[s][d] = over
                if land is not None:
                    jumps[s][d] = (over, land)
    return steps, jumps


STEP, JUMP = _build_tables()


def flip(mask: int) -> int:
    """Rotate a bitboard 180 degrees, which is exactly reversing its 32 bits."""
    return (
        (_REV8[mask & 0xFF] << 24)
        | (_REV8[(mask >> 8) & 0xFF] << 16)
        | (_REV8[(mask >> 16) & 0xFF] << 8)
        | _REV8[mask >> 24]
    )


def bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def legal_moves(position: Position) -> list[Move]:
    """Every legal move. Captures are compulsory, so they hide the quiet moves when they exist."""
    captures: list[Move] = []
    for square in bits(position.own):
        king = bool(position.kings >> square & 1)
        _extend(position, square, king, (square,), 0, captures)
    if captures:
        return captures

    moves = []
    empty = FULL & ~(position.own | position.opp)
    for square in bits(position.own):
        king = bool(position.kings >> square & 1)
        for d in KING_DIRS if king else MAN_DIRS:
            target = STEP[square][d]
            if target >= 0 and empty >> target & 1:
                moves.append(_finish(position, target, king, (square, target), 0))
    return moves


def _extend(position: Position, square: int, king: bool, path: tuple[int, ...], captured: int, out: list[Move]):
    """Depth-first search over jump sequences; captured pieces stay on the board until the move ends."""
    occupied = (position.own & ~(1 << path[0])) | position.opp
    grew = False
    for d in KING_DIRS if king else MAN_DIRS:
        jump = JUMP[square][d]
        if jump is None:
            continue
        over, land = jump
        if not (position.opp >> over & 1) or captured >> over & 1 or occupied >> land & 1:
            continue
        grew = True
        crowned = not king and (CROWN_ROW >> land & 1)
        if crowned:  # promotion ends the move, even mid-jump
            out.append(_finish(position, land, True, (*path, land), captured | 1 << over))
        else:
            _extend(position, land, king, (*path, land), captured | 1 << over, out)
    if not grew and captured:
        out.append(_finish(position, square, king, path, captured))


def _finish(position: Position, square: int, king: bool, path: tuple[int, ...], captured: int) -> Move:
    """Place the moved piece on `square` and hand the position to the opponent."""
    moved = 1 << square
    promotes = not king and bool(CROWN_ROW & moved)
    men = position.men & ~(1 << path[0])
    kings = position.kings & ~(1 << path[0])
    if king or promotes:
        kings |= moved
    else:
        men |= moved
    return Move(
        path=path,
        captures=captured,
        result=Position(
            men=flip(position.opp_men & ~captured),
            kings=flip(position.opp_kings & ~captured),
            opp_men=flip(men),
            opp_kings=flip(kings),
        ),
        promotes=promotes,
    )


MAN_VALUE, KING_VALUE = 100, 160
WIN = 100_000
_ROW_OF = [square_rowcol(s)[0] for s in range(SQUARES)]
_EDGE = sum(1 << s for s in range(SQUARES) if square_rowcol(s)[1] in (0, 7))
_HOME = 0xF0000000  # the side to move's own back row, squares 28-31


def evaluate(position: Position) -> int:
    """Static score in millipawns-of-a-man, from the point of view of the side to move."""
    men, kings = bin(position.men).count("1"), bin(position.kings).count("1")
    opp_men, opp_kings = bin(position.opp_men).count("1"), bin(position.opp_kings).count("1")
    score = MAN_VALUE * (men - opp_men) + KING_VALUE * (kings - opp_kings)

    # Men are worth more the closer they are to crowning; the mover advances toward row 0.
    score += 3 * sum(7 - _ROW_OF[s] for s in bits(position.men))
    score -= 3 * sum(7 - _ROW_OF[s] for s in bits(flip(position.opp_men)))
    # Holding the back row denies the opponent a crown; the edge is safe but passive.
    score += 6 * (bin(position.men & _HOME).count("1") - bin(position.opp_men & _HOME).count("1"))
    score -= 2 * (bin(position.own & _EDGE).count("1") - bin(position.opp & _EDGE).count("1"))
    return score


def search(position: Position, depth: int, alpha: int = -WIN, beta: int = WIN) -> int:
    """Negamax with alpha-beta; forced-capture sequences are searched out past the horizon."""
    moves = legal_moves(position)
    if not moves:
        return -WIN  # the side to move is trapped or has no pieces left, and loses
    if depth <= 0 and not moves[0].captures:
        return evaluate(position)

    best = -WIN
    for move in moves:
        score = -search(move.result, depth - 1, -beta, -alpha)
        if score > best:
            best = score
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best


def best_move(position: Position, depth: int, rng: random.Random | None = None) -> tuple[Move | None, int]:
    """The teacher's move and its value, picking uniformly among equally good moves."""
    moves = legal_moves(position)
    if not moves:
        return None, -WIN
    best, alpha = [], -WIN - 1
    for move in moves:
        score = -search(move.result, depth - 1, -WIN, -alpha if alpha > -WIN else WIN)
        if score > alpha:
            best, alpha = [move], score
        elif score == alpha:
            best.append(move)
    return (rng.choice(best) if rng else best[0]), alpha


def to_array(position: Position):
    """8x8 board: +1/+2 for the side to move's man/king, -1/-2 for the opponent's, row 0 at the top."""
    import numpy as np

    board = np.zeros((8, 8), dtype=np.int8)
    for value, mask in [(1, position.men), (2, position.kings), (-1, position.opp_men), (-2, position.opp_kings)]:
        for square in bits(mask):
            board[square_rowcol(square)] = value
    return board


class GameResult(NamedTuple):
    winner: int | None  # 0 or 1, or None for a draw
    positions: list[Position]  # every position that was on the board, each seen by its side to move
    moves: list[Move]


def play_game(policies, max_plies: int = 200, quiet_limit: int = 60) -> GameResult:
    """Run one game. policies[ply % 2] picks a move; a long stretch without progress is a draw."""
    position, positions, moves, quiet = START, [], [], 0
    for ply in range(max_plies):
        positions.append(position)
        options = legal_moves(position)
        if not options:
            return GameResult(1 - ply % 2, positions, moves)
        move = policies[ply % 2](position, options)
        quiet = 0 if move.captures or move.promotes else quiet + 1
        if quiet >= quiet_limit:
            return GameResult(None, positions, moves)
        moves.append(move)
        position = move.result
    return GameResult(None, positions, moves)


def main():
    """Check the rules against themselves over random games, and the teacher against weaker play."""
    import collections

    counts = collections.Counter()
    for game in range(300):
        rng = random.Random(game)
        result = play_game([lambda position, moves: rng.choice(moves)] * 2)
        for position, move in zip(result.positions, result.moves):
            own, opp = bin(position.own).count("1"), bin(position.opp).count("1")
            assert position.own & position.opp == 0 and position.men & position.kings == 0, "a square holds two pieces"
            assert own <= 12 and opp <= 12, "pieces appeared from nowhere"
            assert not move.captures or all(m.captures for m in legal_moves(position)), "a capture was optional"
            assert bin(move.result.opp).count("1") == own, "the mover lost a piece by moving"
            assert bin(move.result.own).count("1") == opp - bin(move.captures).count("1"), "capture accounting"
            counts["captures"] += bool(move.captures)
            counts["multi-jumps"] += len(move.path) > 2
            counts["promotions"] += move.promotes
        counts["plies"] += len(result.moves)
        counts["decided"] += result.winner is not None
    print(f"300 random games: {dict(counts)}")

    for depth, opponent in [(4, 0), (4, 1), (4, 2)]:
        wins = [0, 0, 0]
        for game in range(20):
            rng = random.Random(game)
            players = [
                lambda position, moves: best_move(position, depth, rng)[0],
                lambda position, moves: rng.choice(moves) if opponent == 0 else best_move(position, opponent, rng)[0],
            ]
            result = play_game(players[:: 1 if game % 2 == 0 else -1])
            winner = result.winner if game % 2 == 0 else (None if result.winner is None else 1 - result.winner)
            wins[2 if winner is None else winner] += 1
        print(f"depth {depth} vs {'random' if opponent == 0 else f'depth {opponent}'}: {wins[0]} won, {wins[1]} lost, {wins[2]} drawn")


if __name__ == "__main__":
    main()
