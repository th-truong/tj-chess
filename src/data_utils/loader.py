import os

import chess.pgn


def stream_games(data, shard_index=None, total_shards=None):
    for path in os.listdir(data):
        current_index = 0
        with open(os.path.join(data, path)) as f:
            while True:
                if total_shards is None or current_index % total_shards == shard_index:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    yield game
                else:
                    game_found = chess.pgn.skip_game()
                    if not game_found:
                        break
                current_index += 1


def stream_headers(data):
    for path in os.listdir(data):
        with open(os.path.join(data, path)) as f:
            while True:
                headers = chess.pgn.read_headers(f)
                if headers is None:
                    break
                yield headers


def stream_boards(game: chess.pgn.Game):
    """
    NOTE: a single board is modified in-place
    """
    board = chess.Board()
    for move in game.mainline_moves():
        board.push(move)
        yield board
