from __future__ import annotations
from typing import Any, Dict, Optional, List, Callable

import chess
import random
from numpy.lib.function_base import flip
import torch
import numpy as np

from data_utils.layer_builder import board_to_all_layers, board_to_layers, flip_layers, HISTORY


class Node:
    def __init__(
        self,
        state: Any,
        value: Optional[float],
        parent: Optional[Node] = None,
        children: Dict[str, Node] = None,
        cache: Any = None,
    ):
        self.state = state
        self.value = value
        self.parent = parent
        self.children = children or {}
        self.cache = cache


def _select(node: Node) -> Node:
    if not node.children:
        return node
    # children not yet evaluated
    if node.children[next(iter(node.children.keys()))].value is None:
        return None
    move = random.choices(
        population=[m for m in node.children],
        weights=[1-c.value for c in node. children.values()],
    )[0]
    child = node.children[move]
    return _select(child)


def expand_node_chess(node: Node) -> Dict[str, Node]:
    children = {}
    for move in node.state.legal_moves:
        # TODO: can we push/pop to aviod copies?
        board = node.state.copy()
        board.push(move)
        child_node = Node(board, None, parent=node, cache={})
        children[move.uci()] = child_node
    return children


def build_chess_state_expand(model):
    def expand_states_chess(state: chess.Board) -> Dict[str, float]:
        pass


def node_to_all_layers(node):
    hist_layers = []
    cur = node
    for i in range(HISTORY):
        if cur is None:
            break
        # if cur is None:
        #     hist_layers.extend(board_to_layers(None, None))
        else:
            if i % 2 == 0:
                hist_layers.extend(cur.cache['layers'])
            else:
                hist_layers.extend(flip_layers(cur.cache['layers']))
            cur = cur.parent
    hist_layers = np.array(hist_layers[:112])
    return hist_layers


def build_chess_state_simulator(model):
    def simulate_states_chess(nodes: List[Node]) -> List[float]:
        for node in nodes:
            # TODO: jamming layers into the node is hacky
            if node.parent is None:
                node.cache['layers'] = board_to_all_layers(node.state.copy())
            else:
                node.cache['layers'] = board_to_layers(node.state, node.state.turn)

        all_layers = []
        for node in nodes:
            all_layers.append(node_to_all_layers(node))
        all_layers = np.array(all_layers)

        tensor = torch.from_numpy(all_layers.astype(np.float32))

        tensor = tensor.to(torch.device('cuda'))
        torch.cuda.synchronize()

        _policies, values, _targets = model(tensor)
        torch.cuda.synchronize()

        # split value of tie between players
        # this happens to weigh wins vs ties nicely
        # and allows us to treat this as a zero sum game
        results = [(v[0] + (v[1] / 2)).item() for v in values]
        return results
    return simulate_states_chess


def count_nodes(node):
    return sum(count_nodes(c) for c in node.children.values()) + 1


def build_chess_limit_search(limit):
    def limit_search_chess(root):
        if limit is None:
            return False
        if limit.nodes is not None:
            num_nodes = count_nodes(root)
            if num_nodes > limit.nodes:
                return True
        return False
    return limit_search_chess


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
    root: Node,
    expand_node: Callable[[Node], Dict[str, Node]],
    simulate_states: Callable[[List[Node]], float],
    limit_search: Optional[Callable[[Node], bool]] = None,
    batches=10,
    batch_size=100
) -> str:
    values = simulate_states([root])
    root.value = values[0]

    for i in range(batches):
        if limit_search is not None and limit_search(root):
            break

        nodes = []
        for j in range(batch_size):
            node = _select(root)
            # kinda hacky, this will give us inconsistent batch sizes
            # TODO: maybe eval blindly instead?
            if node is None:
                continue
            node.children = expand_node(node)
            nodes.append(node)

        child_nodes = [c for n in nodes for c in n.children.values()]
        child_values = simulate_states(child_nodes)
        print('simlated %d states' % len(child_nodes))

        for node, value in zip(child_nodes, child_values):
            node.value = value

        for node in nodes:
            _back_propagate(node)
        print({m: c.value for m, c in root.children.items()})
        print('---')

    best_move = min(root.children, key=lambda c: root.children[c].value)
    print(root.value)
    print({m: c.value for m, c in root.children.items()})
    return best_move, root.value
