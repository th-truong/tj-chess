import config as cfg
from gui import pyqt_classes
from scripts import display_gui

import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn


if __name__ == "__main__":
    # display_gui()
    # engine = chess.engine.SimpleEngine.popen_uci(str(cfg.STOCKFISH_ENGINE_PATH))

    # board = chess.Board()
    # while not board.is_game_over():
    #     result = engine.play(board, chess.engine.Limit(time=0.1))
    #     board.push(result.move)
    #     print(board)

    # engine.quit()

