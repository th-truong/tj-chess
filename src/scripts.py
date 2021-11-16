import sys

import torch
import numpy as np
from PyQt5.QtWidgets import QApplication
import chess.pgn

import config as cfg
from gui import pyqt_classes
from network_utils.engine import TjMctsEngine, TjPolicyEngine


def display_gui(args):
    app = QApplication(sys.argv)

    engines = []
    if args.model is not None:
        engines.append(TjMctsEngine.load(args.model))
        engines.append(TjPolicyEngine.load(args.model))
    if args.stockfish_exe is not None:
        engines.append(chess.engine.SimpleEngine.popen_uci(args.stockfish_exe))

    window = pyqt_classes.chessMainWindow(
        args.chess_db,
        engines=engines,
        training_cfg_dir=args.training_cfg_dir,
    )
    window.show()

    sys.exit(app.exec_())


def benchmark(args):
    board = chess.Board()
    engine = TjMctsEngine.load(args.model)
    for i in range(args.n):
        print(f'turn {i}')
        result = engine.play(board, chess.engine.Limit(time=0.5))
        if result.move is not None:
            board.push(result.move)


def python_chess_ex():
    pgn = open(cfg.CHESS_DB / "lichess_elite_2013-09.pgn", encoding="utf-8")

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
