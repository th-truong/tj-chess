import os

import chess


def stream_games(data):
    for path in os.listdir(data):
        with open(os.path.join(data, path)) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                yield game


def stream_boards(game: chess.pgn.Game):
    """
    NOTE: a single board is modified in-place
    """
    board = chess.Board()
    for move in game.mainline_moves():
        board.push(move)
        yield board
