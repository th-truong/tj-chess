from __future__ import annotations
from typing import Dict, Optional, List
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
        board: chess.Board,
        value: Optional[float],
        parent: Optional[Node] = None,
        children: Dict[chess.Move, Node] = None,
    ):
        self.board = board
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


def _expand(node: Node) -> Dict[chess.Move, Node]:
    children = {}
    for move in node.board.legal_moves:
        # TODO: can we push/pop to aviod copies?
        board = node.board.copy()
        board.push(move)
        child = Node(board, None, parent=node)
        children[move] = child
    return children



def _simulate(nodes: List[Node], model):
    # TODO: this seems stupid an inefficient
    translation_start = time.time()
    pool = multiprocessing.Pool(8)
    layers = np.array(pool.map(board_to_all_layers, (n.board.copy() for n in nodes)))
    # layers = np.array([board_to_all_layers(n.board.copy()) for n in nodes])
    translation_end = time.time()
    tensor = torch.from_numpy(layers.astype(np.float32))
    print('translation time: %s' % (translation_end - translation_start))
    tensor = tensor.to(torch.device('cuda'))
    print(tensor.size())
    print('start model')
    _policies, values, _targets = model(tensor)
    print('done model')
    for i, node in enumerate(nodes):
        # split value of tie between players
        # this happens to weigh wins vs ties nicely
        # and allows us to treat this as a zero sum game
        node.value = values[i][0] + (values[i][1] / 2)


def _back_propagate(node: Optional[Node]):
    if node is None:
        # reached the top
        return
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


def mcts(board: chess.Board, model, batches=10, batch_size=100) -> chess.Move:
    root = Node(board, None)

    with torch.no_grad():
        for i in range(batches):
            nodes = []
            start_expand = time.time()
            for j in range(batch_size):
                node = _select(root)
                # kinda hacky, this will give us inconsistent batch sizes
                # TODO: maybe eval blindly instead?
                if node is None:
                    continue
                node.children = _expand(node)
                nodes.append(node)
            done_expand = time.time()
            print('expand time: %s' % (done_expand - start_expand))
            simulate_start = time.time()
            _simulate([c for n in nodes for c in n.children.values()], model)
            simulate_done = time.time()
            print('simulate time: %s' % (simulate_done - simulate_start))
            backprop_start = time.time()
            for node in nodes:
                _back_propagate(node)
            backprop_done = time.time()
            print('backprop time %s' % (backprop_done - backprop_start))
            print({m.uci(): c.value for m, c in root.children.items()})
            print('---')

    best_move = min(root.children, key=lambda c: root.children[c].value)
    print(root.value)
    print({m.uci(): c.value for m, c in root.children.items()})
    return best_move
