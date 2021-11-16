import sys
import importlib.util

import chess
from PyQt5.QtWidgets import (QInputDialog, QMenuBar, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QTextEdit, QLineEdit, QFileDialog, QApplication,
                             QListWidget, QPlainTextEdit, QMainWindow, QTabWidget,
                             QCheckBox)
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtGui import QWheelEvent
from pathlib import Path

import config as cfg
from network_utils.engine import TjMctsEngine, TjPolicyEngine


class chessMainWindow(QMainWindow):
    def __init__(self, chess_db, engines=None, training_cfg_dir=None):
        super().__init__()
        self.engines = engines or []
        self.title = 'Chess Viewer and Player'
        self.left = 0
        self.top = 0
        self.width = 1000
        self.height = 700
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.tab_widget = chessTabs(self, chess_db, self.engines)
        self.setCentralWidget(self.tab_widget)

    def closeEvent(self, event):
        for engine in self.engines:
            engine.close()


class chessTabs(QWidget):
    def __init__(self, parent, chess_db, engines):
        super(QWidget, self).__init__(parent)

        self.tabs = QTabWidget()
        viewer_tab = ViewerTab(self, chess_db)
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

        self.board = chess.Board()

        self.engines = engines
        self.current_engine = engines[0]

        engine_back_btn = QPushButton("<")
        engine_back_btn.clicked.connect(self.back_btn_click)
        engine_fwd_btn = QPushButton(">")
        engine_fwd_btn.clicked.connect(self.fwd_btn_click)

        engine_move_layout = QHBoxLayout()
        engine_move_layout.addWidget(engine_back_btn)
        engine_move_layout.addWidget(engine_fwd_btn)

        self.engine_list = QListWidget()
        self.engine_list.clicked.connect(self.engine_list_click)
        for i, engine in enumerate(engines):
            self.engine_list.insertItem(i, engine.id['name'])

        self.engine_move_info = QPlainTextEdit()

        engine_options_layout = QVBoxLayout()
        engine_options_layout.addWidget(self.engine_list)
        engine_options_layout.addWidget(self.engine_move_info)

        engine_layout = QVBoxLayout()
        engine_layout.addLayout(engine_options_layout)
        engine_layout.addLayout(engine_move_layout)

        pgn_load_btn = QPushButton("Load")
        pgn_load_btn.clicked.connect(self.pgn_load_btn_click)
        pgn_save_btn = QPushButton("Save")
        pgn_save_btn.clicked.connect(self.pgn_save_btn_click)

        # SVG display for board
        self.svg_widget = QSvgWidget()

        # textbox for entering move
        self.move_input = QLineEdit()
        self.move_input.setText("Enter move here.")
        self.move_input.returnPressed.connect(self.move_push)

        # push move button
        player_move_btn = QPushButton("Move")
        player_move_btn.clicked.connect(self.move_push)

        player_move_layout = QHBoxLayout()
        player_move_layout.addWidget(self.move_input)
        player_move_layout.addWidget(player_move_btn)

        board_layout = QVBoxLayout()
        board_layout.addWidget(self.svg_widget)
        board_layout.addLayout(player_move_layout)

        menu_bar = QMenuBar()
        file_menu = menu_bar.addMenu("File")
        load_pgn_action = file_menu.addAction("Load PGN")
        load_pgn_action.triggered.connect(self.pgn_load_btn_click)
        save_pgn_action = file_menu.addAction("Save PGN")
        save_pgn_action.triggered.connect(self.pgn_save_btn_click)
        file_menu.addSeparator()
        load_model_action = file_menu.addAction("Load Model")
        load_model_action.triggered.connect(self.load_model_click)

        main_layout = QHBoxLayout()
        main_layout.setMenuBar(menu_bar)
        main_layout.addLayout(board_layout)
        main_layout.addLayout(engine_layout)

        self.setLayout(main_layout)
        self.paint_board()

    def paint_board(self):
        self.board_SVG = chess.svg.board(self.board).encode('UTF-8')
        self.svg_widget.load(self.board_SVG)
        self.analyse_position()

    def move_push(self):
        try:
            self.board.push_uci(self.move_input.text())
            self.paint_board()
            self.move_input.setText("")
        except ValueError:
            self.engine_move_info.clear()
            self.engine_move_info.insertPlainText("Invalid move.")

    def back_btn_click(self):
        if len(self.board.move_stack) == 0:
            return
        self.board.pop()
        self.paint_board()

    def fwd_btn_click(self):
        if self.board.is_game_over():
            return
        if self.current_engine is None:
            return
        result = self.current_engine.play(self.board, chess.engine.Limit(time=0.5))
        if result.move is not None:
            self.board.push(result.move)
            self.paint_board()

    def engine_list_click(self, qmodelindex):
        # set game as current selected game from list and display it
        self.current_engine = self.engines[self.engine_list.currentRow()]

    def analyse_position(self):
        info_list = self.current_engine.analyse(self.board, chess.engine.Limit(time=0.1), multipv=5)
        self.engine_move_info.clear()
        if info_list is None or len(info_list) == 0:
            return
        self.engine_move_info.insertPlainText(str(info_list[0]['score']))
        self.engine_move_info.insertPlainText("\n\n")
        for info in info_list:
            self.engine_move_info.insertPlainText(str(info['pv'][0]))
            self.engine_move_info.insertPlainText("\n\n")

    def pgn_load_btn_click(self):
        # get file path
        name, _ = QFileDialog.getOpenFileName(self, caption='Load PGN', filter='PGN files (*.pgn)')
        if name == '':
            return
        games = []
        with open(name, 'r') as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                games.append(game)
        # TODO: assumes unique names
        game_names = [g.headers.get('Opening', f'Game {i+1}') for i, g in enumerate(games)]
        item, ok = QInputDialog().getItem(self, "Open Game", "Select a game", game_names)
        if not ok:
            return
        game = games[game_names.index(item)]
        self.board = game.board()
        for move in game.mainline_moves():
            self.board.push(move)
        self.paint_board()

    def pgn_save_btn_click(self):
        game = chess.pgn.Game.from_board(self.board)
        name, _ = QFileDialog.getSaveFileName(self, 'Save PGN', '.pgn')
        if name == '':
            return
        with open(name, 'w') as f:
            print(game, file=f, end='\n\n')
    
    def load_model_click(self):
        name, _ = QFileDialog.getOpenFileName(self, caption='Load Model', filter='TAR files (*.tar)')
        if name == '':
            return
        for engine in (
            TjPolicyEngine.load(name),
            TjMctsEngine.load(name),
        ):
            self.engine_list.insertItem(len(self.engines), engine.id['name'])
            self.engines.append(engine)


class ViewerTab(QWidget):
    def __init__(self, parent, chess_db):
        super(QWidget, self).__init__(parent)
        layout = QGridLayout(self)

        self.chess_db = chess_db

        self.setLayout(layout)
        self.current_game = None

        # textbox for locating .pgn file to open
        self.pgn_file_txt = QLineEdit(parent)
        self.pgn_file_txt.setText("Enter .pgn path here.")
        layout.addWidget(self.pgn_file_txt, 0, 0, 1, 7)

        # load file button
        self.open_file_btn = QPushButton("Load")
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
        self.engine_move_info = QPlainTextEdit(parent)
        layout.addWidget(self.engine_move_info, 4, 8, 6, 7)

    def games_list_click(self, qmodelindex):
        # set game as current selected game from list and display it
        self.current_game = self.games[self.games_list.currentRow()]
        self.engine_move_info.clear()
        self.engine_move_info.insertPlainText(str(self.current_game))

        # load in mainline moves
        self.mainline_moves = [move for move in self.current_game.mainline_moves()]
        # used to track which move the display is currently at
        self.move_counter = 0

        # set board display to current game data
        self.board = self.current_game.board()
        self.paint_board()

    def open_file_btn_click(self):
        # get file path
        fname = QFileDialog.getOpenFileName(self, 'Open file', str(self.chess_db), "PGN files (*.pgn)")
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
            game_openers = [g.headers.get('Opening', f'Game {i+1}') for i, g in enumerate(self.games)]

            list_count = 0
            for opener in game_openers:
                self.games_list.insertItem(list_count, opener)
                list_count += 1

    def back_btn_click(self):
        if self.current_game is None:
            self.engine_move_info.clear()
            self.engine_move_info.insertPlainText("Need to load a game! Select a .pgn file and then select a game from the list above.")
        else:
            if self.move_counter > 0:
                # undo last move
                self.board.pop()
                # diplay result
                self.paint_board()

                self.move_counter -= 1

    def fwd_btn_click(self):
        if self.current_game is None:
            self.engine_move_info.clear()
            self.engine_move_info.insertPlainText("Need to load a game! Select a .pgn file and then select a game from the list above.")
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
