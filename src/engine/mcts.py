from __future__ import annotations
from typing import Any, Dict, Optional, List, Callable
import multiprocessing
import time

import chess
import random
import torch
import numpy as np

from data_utils.layer_builder import board_to_all_layers


class Node:
    def __init__(
        self,
        state: Any,
        value: Optional[float],
        parent: Optional[Node] = None,
        children: Dict[str, Node] = None,
    ):
        self.state = state
        self.value = value
        self.parent = parent
        self.children = children or {}


def _select(node: Node) -> Node:
    if not node.children:
        return node
    # children not yet evaluated
    if node.children[next(iter(node.children.keys()))].value is None:
        return None
    move = random.choices(
        population=[m for m in node.children],
        weights=[c.value for c in node. children.values()],
    )[0]
    child = node.children[move]
    return _select(child)


def expand_state_chess(state) -> Dict[str, Any]:
    children = {}
    for move in state.legal_moves:
        # TODO: can we push/pop to aviod copies?
        board = state.copy()
        board.push(move)
        children[move.uci()] = board
    return children


def build_chess_state_simulator(model):
    def simulate_states_chess(states: List[Any]):
        # TODO: this seems stupid an inefficient
        translation_start = time.time()
        pool = multiprocessing.Pool(8)
        layers = np.array(pool.map(board_to_all_layers, (s.copy() for s in states)))
        # layers = np.array([board_to_all_layers(n.board.copy()) for n in nodes])
        translation_end = time.time()
        tensor = torch.from_numpy(layers.astype(np.float32))
        print('translation time: %s' % (translation_end - translation_start))
        tensor = tensor.to(torch.device('cuda'))
        print(tensor.size())
        print('start model')
        _policies, values, _targets = model(tensor)
        print('done model')
        # split value of tie between players
        # this happens to weigh wins vs ties nicely
        # and allows us to treat this as a zero sum game
        return [v[0] + (v[1] / 2) for v in values]
    return simulate_states_chess


def _back_propagate(node: Optional[Node]):
    if node is None:
        # reached the top
        return
    if len(node.children) == 0:
        # reached the bottom
        return node.value
    # white: move a -> black: 80% chance to win
    # white: move b -> black: 40% chance to win
    # white: 60% chance to win
    min_oppenent_value = min(c.value for c in node.children.values())
    # NOTE: this assumes value ranges from 0-1
    max_value = 1 - min_oppenent_value
    if node.value == max_value:
        # no further changes
        return
    node.value = max_value
    _back_propagate(node.parent)


def mcts(
    state: Any,
    expand_state: Callable[[Any], Dict[str, Any]],
    simulate_states: Callable[[List[Any]], float],
    batches=10,
    batch_size=100
) -> str:
    root = Node(state, None)

    for i in range(batches):
        nodes = []
        start_expand = time.time()
        for j in range(batch_size):
            node = _select(root)
            # kinda hacky, this will give us inconsistent batch sizes
            # TODO: maybe eval blindly instead?
            if node is None:
                continue
            child_states = expand_state(node.state)
            node.children = {m: Node(s, None, node) for m, s in child_states.items()}
            nodes.append(node)
        done_expand = time.time()
        print('expand time: %s' % (done_expand - start_expand))
        simulate_start = time.time()
        child_nodes = [c for n in nodes for c in n.children.values()]
        child_values = simulate_states([c.state for c in child_nodes])
        for node, value in zip(child_nodes, child_values):
            node.value = value
        simulate_done = time.time()
        print('simulate time: %s' % (simulate_done - simulate_start))
        backprop_start = time.time()
        for node in nodes:
            _back_propagate(node)
        backprop_done = time.time()
        print('backprop time %s' % (backprop_done - backprop_start))
        print({m: c.value for m, c in root.children.items()})
        print('---')

    best_move = min(root.children, key=lambda c: root.children[c].value)
    print(root.value)
    print({m: c.value for m, c in root.children.items()})
    return best_move, root.value
