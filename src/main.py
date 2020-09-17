import proj_constants as pc
from gui import pyqt_classes

from pathlib import Path
from PyQt5.QtWidgets import QApplication
import chess.pgn

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = pyqt_classes.chessMainWindow()
    ex.show()
    sys.exit(app.exec_())


def python_chess_ex():
    pgn = open(pc.LICHESS_DB / "lichess_elite_2013-09.pgn", encoding="utf-8")

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
