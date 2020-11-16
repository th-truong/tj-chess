import sys
import importlib.util

import chess
from PyQt5.QtWidgets import (QWidget, QGridLayout, QVBoxLayout, QPushButton,
                             QLabel, QTextEdit, QLineEdit, QFileDialog, QApplication,
                             QListWidget, QPlainTextEdit, QMainWindow, QTabWidget,
                             QCheckBox)
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtGui import QWheelEvent
from pathlib import Path

import config as cfg
from network_utils.engine import TjEngine


class chessMainWindow(QMainWindow):
    def __init__(self, lichess_db, stockfish=None, model=None, training_cfg_dir=None):
        super().__init__()

        self.engines = []
        if model is not None:
            spec = importlib.util.spec_from_file_location("", training_cfg_dir)
            cfg = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cfg)

            self.engines.append(TjEngine.load(model, cfg))
        if stockfish is not None:
            self.engines.append(chess.engine.SimpleEngine.popen_uci(stockfish))

        self.title = 'Chess Viewer and Player'
        self.left = 0
        self.top = 0
        self.width = 1000
        self.height = 700
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.tab_widget = chessTabs(self, lichess_db, self.engines)
        self.setCentralWidget(self.tab_widget)

    def closeEvent(self, event):
        for engine in self.engines:
            engine.close()


class chessTabs(QWidget):
    def __init__(self, parent, lichess_db, engines):
        super(QWidget, self).__init__(parent)

        self.tabs = QTabWidget()
        viewer_tab = ViewerTab(self, lichess_db)
        player_tab = PlayerTab(self, engines)

        self.tabs.addTab(viewer_tab, "Viewer")
        self.tabs.addTab(player_tab, "Player")

        layout = QGridLayout(self)
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def wheelEvent(self, event: QWheelEvent):
        tab = self.tabs.currentWidget()
        if event.angleDelta().y() > 0:
            tab.fwd_btn_click()
        elif event.angleDelta().y() < 0:
            tab.back_btn_click()


class PlayerTab(QWidget):
    def __init__(self, parent, engines):
        super(QWidget, self).__init__(parent)

        self.engines = engines
        self.current_engine = None

        layout = QGridLayout(self)

        # SVG display for board
        self.svg_widget = QSvgWidget(parent=self)
        self.svg_widget.setGeometry(10, 10, 1000, 1000)
        layout.addWidget(self.svg_widget, 1, 0, 8, 8)
        self.board = chess.Board()
        self.paint_board()
        # keeps track of current move number in game
        self.move_counter = 0

        # textbox for entering move
        self.move_input = QLineEdit(self)
        layout.addWidget(self.move_input, 0, 0, 1, 7)
        self.move_input.setText("Enter move here.")
        self.move_input.returnPressed.connect(self.move_push)

        # step back move button
        self.back_btn = QPushButton("<")
        self.back_btn.clicked.connect(self.back_btn_click)
        layout.addWidget(self.back_btn, 9, 2)

        # step forward (get computer move and play it)
        self.fwd_btn = QPushButton(">")
        self.fwd_btn.clicked.connect(self.fwd_btn_click)
        layout.addWidget(self.fwd_btn, 9, 5)

        # push move button
        self.push_move_btn = QPushButton("")
        self.push_move_btn.clicked.connect(self.move_push)
        layout.addWidget(self.push_move_btn, 0, 7, 1, 1)

        # listbox for games in current pgn
        self.engine_list = QListWidget()
        self.engine_list.clicked.connect(self.engine_list_click)
        layout.addWidget(self.engine_list, 0, 8, 4, 7)
        for i, engine in enumerate(engines):
            self.engine_list.insertItem(i, engine.id['name'])

        # textbox displaying current game moves
        self.current_game_moves_txt = QPlainTextEdit(self)
        layout.addWidget(self.current_game_moves_txt, 4, 8, 6, 7)

        self.setLayout(layout)

    def paint_board(self):
        self.board_SVG = chess.svg.board(self.board).encode('UTF-8')
        self.svg_widget.load(self.board_SVG)

    def move_push(self):
        try:
            self.board.push_uci(self.move_input.text())
            self.move_counter += 1

            self.paint_board()
            self.move_input.setText("")
        except ValueError:
            self.current_game_moves_txt.clear()
            self.current_game_moves_txt.insertPlainText("Invalid move.")

    def back_btn_click(self):
        if self.move_counter > 0:
            # undo last move
            self.board.pop()
            # diplay result
            self.paint_board()
            # reduce move counter
            self.move_counter -= 1

    def fwd_btn_click(self):
        if self.current_engine is None:
            return
        result = self.current_engine.play(self.board, chess.engine.Limit(time=0.1))
        if result.move is not None:
            self.board.push(result.move)
            self.paint_board()
            self.move_counter += 1

    def engine_list_click(self, qmodelindex):
        # set game as current selected game from list and display it
        self.current_engine = self.engines[self.engine_list.currentRow()]


class ViewerTab(QWidget):
    def __init__(self, parent, lichess_db):
        super(QWidget, self).__init__(parent)
        layout = QGridLayout(self)

        self.lichess_db = lichess_db

        self.setLayout(layout)
        self.current_game = None

        # textbox for locating .pgn file to open
        self.pgn_file_txt = QLineEdit(parent)
        self.pgn_file_txt.setText("Enter .pgn path here.")
        layout.addWidget(self.pgn_file_txt, 0, 0, 1, 7)

        # load file button
        self.open_file_btn = QPushButton("")
        self.open_file_btn.clicked.connect(self.open_file_btn_click)
        layout.addWidget(self.open_file_btn, 0, 7, 1, 1)

        # step back move button
        self.back_btn = QPushButton("<")
        self.back_btn.clicked.connect(self.back_btn_click)
        layout.addWidget(self.back_btn, 9, 2)

        # step Forward move button
        self.fwd_btn = QPushButton(">")
        self.fwd_btn.clicked.connect(self.fwd_btn_click)
        layout.addWidget(self.fwd_btn, 9, 5)

        # SVG display for board
        self.svg_widget = QSvgWidget(parent=parent)
        self.svg_widget.setGeometry(10, 10, 1000, 1000)
        layout.addWidget(self.svg_widget, 1, 0, 8, 8)
        self.board = chess.Board()
        self.paint_board()

        # listbox for games in current pgn
        self.games_list = QListWidget()
        self.games_list.clicked.connect(self.games_list_click)
        layout.addWidget(self.games_list, 0, 8, 4, 7)

        # textbox displaying current game moves
        self.current_game_moves_txt = QPlainTextEdit(parent)
        layout.addWidget(self.current_game_moves_txt, 4, 8, 6, 7)

    def games_list_click(self, qmodelindex):
        # set game as current selected game from list and display it
        self.current_game = self.games[self.games_list.currentRow()]
        self.current_game_moves_txt.clear()
        self.current_game_moves_txt.insertPlainText(str(self.current_game))

        # load in mainline moves
        self.mainline_moves = [move for move in self.current_game.mainline_moves()]
        # used to track which move the display is currently at
        self.move_counter = 0

        # set board display to current game data
        self.board = self.current_game.board()
        self.paint_board()

    def open_file_btn_click(self):
        # get file path
        fname = QFileDialog.getOpenFileName(self, 'Open file', str(self.lichess_db), "PGN files (*.pgn)")
        fname = Path(str(fname[0]))
        self.pgn_file_txt.setText(str(fname))

        if Path(self.pgn_file_txt.text()).exists() and Path(self.pgn_file_txt.text()).suffix == '.pgn':
            # open game file
            pgn_file = open(Path(self.pgn_file_txt.text()))
            self.games = []
            game = chess.pgn.read_game(pgn_file)
            while game is not None:
                self.games.append(game)
                game = chess.pgn.read_game(pgn_file)

            # populate list with games
            self.games_list.clear()
            game_openers = [i.headers['Opening'] for i in self.games]

            list_count = 0
            for opener in game_openers:
                self.games_list.insertItem(list_count, opener)
                list_count += 1

    def back_btn_click(self):
        if self.current_game is None:
            self.current_game_moves_txt.clear()
            self.current_game_moves_txt.insertPlainText("Need to load a game! Select a .pgn file and then select a game from the list above.")
        else:
            if self.move_counter > 0:
                # undo last move
                self.board.pop()
                # diplay result
                self.paint_board()

                self.move_counter -= 1

    def fwd_btn_click(self):
        if self.current_game is None:
            self.current_game_moves_txt.clear()
            self.current_game_moves_txt.insertPlainText("Need to load a game! Select a .pgn file and then select a game from the list above.")
        else:
            if self.move_counter < len(self.mainline_moves):
                # push the next move
                self.board.push(self.mainline_moves[self.move_counter])
                # display result
                self.paint_board()

                self.move_counter += 1

    def paint_board(self):
        self.board_SVG = chess.svg.board(self.board).encode('UTF-8')
        self.svg_widget.load(self.board_SVG)
