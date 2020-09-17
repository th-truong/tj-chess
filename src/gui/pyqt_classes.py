import sys
import chess
from PyQt5.QtWidgets import (QWidget, QGridLayout, QVBoxLayout, QPushButton,
                             QLabel, QTextEdit, QLineEdit, QFileDialog, QApplication,
                             QListWidget, QPlainTextEdit)
from PyQt5.QtSvg import QSvgWidget
from pathlib import Path

import proj_constants as pc


class chessMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        layout = QGridLayout()
        # fix column and row stretching so they don't warp the widgets
        for i in range(0, 10):
            layout.setRowStretch(i, 1)
        for i in range(0, 15):
            layout.setColumnStretch(i, 1)

        self.setLayout(layout)
        self.current_game = None

        # textbox for locating .pgn file to open
        self.pgn_file_txt = QLineEdit(self)
        layout.addWidget(self.pgn_file_txt, 0, 0, 1, 7)

        # load file button
        self.open_file_btn = QPushButton("...")
        self.open_file_btn.clicked.connect(self.open_file_btn_click)
        layout.addWidget(self.open_file_btn, 0, 7)

        # step back move button
        self.back_btn = QPushButton("<")
        self.back_btn.clicked.connect(self.back_btn_click)
        layout.addWidget(self.back_btn, 9, 3)

        # step Forward move button
        self.fwd_btn = QPushButton(">")
        self.fwd_btn.clicked.connect(self.fwd_btn_click)
        layout.addWidget(self.fwd_btn, 9, 5)

        # SVG display for board
        self.SVG_widget = QSvgWidget(parent=self)
        self.SVG_widget.setGeometry(10, 10, 1000, 1000)
        layout.addWidget(self.SVG_widget, 1, 0, 8, 8)
        self.board = chess.Board()
        self.paint_board()

        # listbox for games in current pgn
        self.games_list = QListWidget()
        self.games_list.clicked.connect(self.games_list_click)
        layout.addWidget(self.games_list, 0, 8, 4, 7)

        # textbox displaying current game moves
        self.current_game_moves_txt = QPlainTextEdit(self)
        layout.addWidget(self.current_game_moves_txt, 4, 8, 6, 7)

    def open_file_btn_click(self):
        # get file path
        fname = QFileDialog.getOpenFileName(self, 'Open file', str(pc.LICHESS_DB), "PGN files (*.pgn)")
        fname = Path(str(fname[0]))
        self.pgn_file_txt.setText(str(fname))

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
        self.SVG_widget.load(self.board_SVG)
