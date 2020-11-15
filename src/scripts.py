import sys
import importlib.util

import torch
import numpy as np
from PyQt5.QtWidgets import QApplication
import chess.pgn

import config as cfg
from gui import pyqt_classes


def display_gui(args):
    app = QApplication(sys.argv)

    spec = importlib.util.spec_from_file_location("", args.training_cfg_dir)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    window = pyqt_classes.chessMainWindow(args.lichess_db, stockfish=args.stockfish_exe, model=args.model, cfg=cfg)
    window.show()

    sys.exit(app.exec_())


def python_chess_ex():
    pgn = open(cfg.LICHESS_DB / "lichess_elite_2013-09.pgn", encoding="utf-8")

    first_game = chess.pgn.read_game(pgn)
    second_game = chess.pgn.read_game(pgn)

    first_game.headers["Event"]
    'IBM Man-Machine, New York USA'

    # Iterate through all moves and play them on a board.
    board = first_game.board()
    for move in first_game.mainline_moves():
        board.push(move)
        board

    board


def quick_bot_test():
    from network_utils.load_tj_model import load_tj_model
    import chess
    from network_utils import network_out_interpreter as noi
    from data_utils import layer_builder
    import numpy as np

    model = load_tj_model(r"D:\paper_repos\tj-chess\CE_loss_model_tensorboard\25001_steps.tar")
    model.eval()

    board = chess.Board()
    print(board)
    print('')
    while True:
        meta = layer_builder.Meta(
            board.turn,
            None,
            None
        )
        input_tensor = np.array(layer_builder.board_to_layers(board, meta))

        input_tensor = torch.from_numpy(input_tensor.astype(np.float32))

        # out = model(input_tensor)
        # indices = torch.argmax(out, keepdim=True)

        print(input_tensor.shape)
        break
