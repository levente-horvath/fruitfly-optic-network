"""Look ahead before asking the network what it thinks.

Search is worth more here than any plausible improvement to the evaluator: the handcrafted
evaluation matches the depth-6 teacher's move 42.5% of the time when used statically, 50.0% once
forced captures are resolved, and 71.5% at three plies.

The tree is expanded full width rather than with alpha-beta. Pruning is inherently sequential, and
here a single board costs 80 sparse matrix multiplies while the tree bookkeeping costs nothing --
so enumerating every leaf and scoring them all in one batch beats pruning to a third as many
evaluations that have to be made one at a time.
"""

from typing import Callable

import numpy as np

from fly_sim import checkers as ck

LOSS = -1e9  # a position whose side to move has no move at all; worse than any evaluation,
# on any scale -- a value merely "very bad" would be preferred to a real loss by whatever scale it is not on


class Node:
    """Either a leaf to be scored, a terminal loss, or a list of children one ply deeper."""

    __slots__ = ("position", "children", "terminal")

    def __init__(self, position=None, children=None, terminal=False):
        self.position, self.children, self.terminal = position, children, terminal


def expand(position: ck.Position, depth: int, quiescence: int = 6) -> Node:
    """Build the tree below `position`, continuing past `depth` while captures are forced."""
    moves = ck.legal_moves(position)
    if not moves:
        return Node(terminal=True)
    if depth <= 0 and not moves[0].captures:
        return Node(position=position)
    if depth <= 0 and quiescence <= 0:
        return Node(position=position)
    step = (depth - 1, quiescence) if depth > 0 else (0, quiescence - 1)
    return Node(children=[expand(move.result, *step) for move in moves])


def leaves(node: Node, out: list):
    if node.terminal:
        return
    if node.children is None:
        out.append(node.position)
        return
    for child in node.children:
        leaves(child, out)


def fold(node: Node, values: dict) -> float:
    """Negamax: a position is worth the best of what the side to move can reach."""
    if node.terminal:
        return LOSS
    if node.children is None:
        return values[node.position]
    return max(-fold(child, values) for child in node.children)


def move_scores(
    position: ck.Position,
    evaluate: Callable[[list[ck.Position]], np.ndarray],
    depth: int = 1,
) -> tuple[list[ck.Move], np.ndarray]:
    """Every legal move and its value to the side to move, searched `depth` plies past the move.

    `evaluate` scores a list of positions for whichever side is to move in each. depth=0 reproduces
    the old behaviour of scoring each afterstate directly, with forced captures resolved.
    """
    moves = ck.legal_moves(position)
    trees = [expand(move.result, depth) for move in moves]
    wanted: list[ck.Position] = []
    for tree in trees:
        leaves(tree, wanted)
    unique = list(dict.fromkeys(wanted))
    values = dict(zip(unique, evaluate(unique))) if unique else {}
    return moves, np.array([-fold(tree, values) for tree in trees])


def count_leaves(position: ck.Position, depth: int) -> int:
    out: list = []
    for move in ck.legal_moves(position):
        leaves(expand(move.result, depth), out)
    return len(set(out))
