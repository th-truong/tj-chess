#!/usr/bin/env python
import os

import numpy as np
import chess.pgn

import config as cfg

HISTORY = cfg.HISTORY
SIZE = cfg.SIZE
REPETITIONS = 2


class MetaLayers(object):
    def __init__(self, meta, layers):
        self.meta = meta
        self.layers = layers


class Meta(object):
    def __init__(self, turn, result, next_move):
        self.turn = turn
        self.result = result
        self.next_move = next_move


def board_to_layers(board, turn):
    if board is None:
        return [np.zeros(SIZE) for _ in range(len(chess.COLORS) * len(chess.PIECE_TYPES) + REPETITIONS)]

    if turn == chess.WHITE:
        colors = [chess.WHITE, chess.BLACK]
    else:
        colors = [chess.BLACK, chess.WHITE]
        board = board.transform(chess.flip_vertical).transform(chess.flip_horizontal)

    board_layers = []
    for color in colors:
        for piece in chess.PIECE_TYPES:
            pieces = board.pieces(piece, color)
            # transform uint64 -> 8x8 np array
            layer = np.unpackbits(np.array(np.uint64(pieces)).view(dtype=np.uint8, type=np.matrix)).reshape(SIZE)
            board_layers.append(layer)

    for i in range(REPETITIONS):
        if board.is_repetition(i+1):
            board_layers.append(np.ones(SIZE))
        else:
            board_layers.append(np.zeros(SIZE))
    return board_layers


def board_to_all_layers(board):
    """
    WARNING!! this kills the board's move stack. please pass in a copy
    """
    turn = board.turn
    all_board_layers = []
    for _ in range(HISTORY):
        board_layers = board_to_layers(board, turn)
        all_board_layers.extend(board_layers)
        try:
            if board is not None:
                board.pop()
        except IndexError:
            board = None
    layers = np.array(all_board_layers)
    return layers


def game_to_layers(game):
    game_node = game
    while True:
        board = game_node.board()
        next_game_node = game_node.next()
        if next_game_node is None:
            break

        meta = Meta(
            board.turn,
            game.headers['Result'],
            next_game_node.move,
        )
        layers = board_to_all_layers(board)
        meta_layers = MetaLayers(meta, layers)
        yield meta_layers

        game_node = next_game_node


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()

    np.set_printoptions(threshold=np.inf)

    with open(args.path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            print(game.headers)
            for i, meta_layers in enumerate(game_to_layers(game)):
                print('turn: %s' % i)
                print('color: %s' % ('white' if meta_layers.meta.turn == chess.WHITE else 'black'))
                print('next move: %s' % (meta_layers.meta.next_move))
                print(meta_layers.layers[0])
                print(meta_layers.layers[6])
                print(meta_layers.layers.shape)
                if i == 2:
                    break
            break
