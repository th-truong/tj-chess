import sys
import chess
from PyQt5.QtWidgets import (QWidget, QGridLayout, QVBoxLayout, QPushButton,
                             QLabel, QTextEdit, QLineEdit, QFileDialog, QApplication,
                             QListWidget, QPlainTextEdit, QMainWindow, QTabWidget,
                             QCheckBox)
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtGui import QWheelEvent
from pathlib import Path

import config as cfg


class chessMainWindow(QMainWindow):
    def __init__(self, lichess_db, engine):
        super().__init__()

        self.title = 'Chess Viewer and Player'
        self.left = 0
        self.top = 0
        self.width = 1000
        self.height = 700
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.tab_widget = chessTabs(self, lichess_db, engine)
        self.setCentralWidget(self.tab_widget)

        self.show()


class chessTabs(QWidget):
    def __init__(self, parent, lichess_db, engine):
        super(QWidget, self).__init__(parent)

        self.lichess_db = lichess_db
        self.engine = engine

        self.layout = QGridLayout(self)

        self.tabs = QTabWidget()
        self.viewer_tab = QWidget()
        self.player_tab = QWidget()

        self.tabs.addTab(self.viewer_tab, "Viewer")
        self.tabs.addTab(self.player_tab, "Player")

        self._initialize_viewer_tab()
        self._initialize_player_tab()

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def viewer_open_file_btn_click(self):
        # get file path
        fname = QFileDialog.getOpenFileName(self, 'Open file', str(self.lichess_db), "PGN files (*.pgn)")
        fname = Path(str(fname[0]))
        self.viewer_pgn_file_txt.setText(str(fname))

        if Path(self.viewer_pgn_file_txt.text()).exists() and Path(self.viewer_pgn_file_txt.text()).suffix == '.pgn':
            # open game file
            pgn_file = open(Path(self.viewer_pgn_file_txt.text()))
            self.viewer_games = []
            game = chess.pgn.read_game(pgn_file)
            while game is not None:
                self.viewer_games.append(game)
                game = chess.pgn.read_game(pgn_file)

            # populate list with viewer_games
            self.viewer_games_list.clear()
            game_openers = [i.headers['Opening'] for i in self.viewer_games]

            list_count = 0
            for opener in game_openers:
                self.viewer_games_list.insertItem(list_count, opener)
                list_count += 1

    def viewer_games_list_click(self, qmodelindex):
        # set game as current selected game from list and display it
        self.viewer_current_game = self.viewer_games[self.viewer_games_list.currentRow()]
        self.viewer_current_game_moves_txt.clear()
        self.viewer_current_game_moves_txt.insertPlainText(str(self.viewer_current_game))

        # load in mainline moves
        self.viewer_mainline_moves = [move for move in self.viewer_current_game.mainline_moves()]
        # used to track which move the display is currently at
        self.viewer_move_counter = 0

        # set board display to current game data
        self.viewer_board = self.viewer_current_game.board()
        self.viewer_paint_board()

    def viewer_back_btn_click(self):
        if self.viewer_current_game is None:
            self.viewer_current_game_moves_txt.clear()
            self.viewer_current_game_moves_txt.insertPlainText("Need to load a game! Select a .pgn file and then select a game from the list above.")
        else:
            if self.viewer_move_counter > 0:
                # undo last move
                self.viewer_board.pop()
                # diplay result
                self.viewer_paint_board()

                self.viewer_move_counter -= 1

    def viewer_fwd_btn_click(self):
        if self.viewer_current_game is None:
            self.viewer_current_game_moves_txt.clear()
            self.viewer_current_game_moves_txt.insertPlainText("Need to load a game! Select a .pgn file and then select a game from the list above.")
        else:
            if self.viewer_move_counter < len(self.viewer_mainline_moves):
                # push the next move
                self.viewer_board.push(self.viewer_mainline_moves[self.viewer_move_counter])
                # display result
                self.viewer_paint_board()

                self.viewer_move_counter += 1

    def viewer_paint_board(self):
        self.viewer_board_SVG = chess.svg.board(self.viewer_board).encode('UTF-8')
        self.viewer_SVG_widget.load(self.viewer_board_SVG)

    def _initialize_viewer_tab(self):
        self.viewer_tab.layout = QGridLayout(self.viewer_tab)

        self.viewer_tab.setLayout(self.viewer_tab.layout)
        self.viewer_current_game = None

        # textbox for locating .pgn file to open
        self.viewer_pgn_file_txt = QLineEdit(self)
        self.viewer_pgn_file_txt.setText("Enter .pgn path here.")
        self.viewer_tab.layout.addWidget(self.viewer_pgn_file_txt, 0, 0, 1, 7)

        # load file button
        self.viewer_open_file_btn = QPushButton("")
        self.viewer_open_file_btn.clicked.connect(self.viewer_open_file_btn_click)
        self.viewer_tab.layout.addWidget(self.viewer_open_file_btn, 0, 7, 1, 1)

        # step back move button
        self.viewer_back_btn = QPushButton("<")
        self.viewer_back_btn.clicked.connect(self.viewer_back_btn_click)
        self.viewer_tab.layout.addWidget(self.viewer_back_btn, 9, 2)

        # step Forward move button
        self.viewer_fwd_btn = QPushButton(">")
        self.viewer_fwd_btn.clicked.connect(self.viewer_fwd_btn_click)
        self.viewer_tab.layout.addWidget(self.viewer_fwd_btn, 9, 5)

        # SVG display for board
        self.viewer_SVG_widget = QSvgWidget(parent=self)
        self.viewer_SVG_widget.setGeometry(10, 10, 1000, 1000)
        self.viewer_tab.layout.addWidget(self.viewer_SVG_widget, 1, 0, 8, 8)
        self.viewer_board = chess.Board()
        self.viewer_paint_board()

        # listbox for viewer_games in current pgn
        self.viewer_games_list = QListWidget()
        self.viewer_games_list.clicked.connect(self.viewer_games_list_click)
        self.viewer_tab.layout.addWidget(self.viewer_games_list, 0, 8, 4, 7)

        # textbox displaying current game moves
        self.viewer_current_game_moves_txt = QPlainTextEdit(self)
        self.viewer_tab.layout.addWidget(self.viewer_current_game_moves_txt, 4, 8, 6, 7)

    def player_paint_board(self):
        self.player_board_SVG = chess.svg.board(self.player_board).encode('UTF-8')
        self.player_SVG_widget.load(self.player_board_SVG)

    def player_move_push(self):
        try:
            self.player_board.push_uci(self.player_move_input.text())
            self.player_move_counter += 1

            self.player_paint_board()
            self.player_move_input.setText("")
        except ValueError:
            self.player_current_game_moves_txt.clear()
            self.player_current_game_moves_txt.insertPlainText("Invalid move.")

    def player_back_btn_click(self):
        if self.player_move_counter > 0:
            # undo last move
            self.player_board.pop()
            # diplay result
            self.player_paint_board()
            # reduce move counter
            self.player_move_counter -= 1

    def player_fwd_btn_click(self):
        result = self.engine.play(self.player_board, chess.engine.Limit(time=0.1))
        if result.move is not None:
            self.player_board.push(result.move)
            self.player_paint_board()
            self.player_move_counter += 1

    def wheelEvent(self, event: QWheelEvent):
        if self.tabs.currentIndex() == 0:  # viewer window
            if event.angleDelta().y() > 0:
                self.viewer_fwd_btn_click()
            elif event.angleDelta().y() < 0:
                self.viewer_back_btn_click()
        elif self.tabs.currentIndex() == 1:  # palyer window
            if event.angleDelta().y() > 0:
                self.player_fwd_btn_click()
            elif event.angleDelta().y() < 0:
                self.player_back_btn_click()

    def _initialize_player_tab(self):
        self.player_tab.layout = QGridLayout(self.player_tab)
        self.player_tab.setLayout(self.player_tab.layout)

        # SVG display for board
        self.player_SVG_widget = QSvgWidget(parent=self)
        self.player_SVG_widget.setGeometry(10, 10, 1000, 1000)
        self.player_tab.layout.addWidget(self.player_SVG_widget, 1, 0, 8, 8)
        self.player_board = chess.Board()
        self.player_paint_board()
        # keeps track of current move number in game
        self.player_move_counter = 0

        # textbox for entering move
        self.player_move_input = QLineEdit(self)
        self.player_tab.layout.addWidget(self.player_move_input, 0, 0, 1, 7)
        self.player_move_input.setText("Enter move here.")
        self.player_move_input.returnPressed.connect(self.player_move_push)

        # step back move button
        self.player_back_btn = QPushButton("<")
        self.player_back_btn.clicked.connect(self.player_back_btn_click)
        self.player_tab.layout.addWidget(self.player_back_btn, 9, 2)

        # step forward (get computer move and play it)
        self.player_fwd_btn = QPushButton(">")
        self.player_fwd_btn.clicked.connect(self.player_fwd_btn_click)
        self.player_tab.layout.addWidget(self.player_fwd_btn, 9, 5)

        # push move button
        self.player_push_move_btn = QPushButton("")
        self.player_push_move_btn.clicked.connect(self.player_move_push)
        self.player_tab.layout.addWidget(self.player_push_move_btn, 0, 7, 1, 1)

        # textbox displaying current game moves
        self.player_current_game_moves_txt = QPlainTextEdit(self)
        self.player_tab.layout.addWidget(self.player_current_game_moves_txt, 0, 8, 10, 7)
