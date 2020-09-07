import sys
from PyQt5.QtWidgets import (QWidget, QGridLayout, QVBoxLayout, QPushButton,
                             QLabel, QTextEdit, QLineEdit, QFileDialog, QApplication)
from pathlib import Path

import proj_constants as pc


class chessMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        layout = QGridLayout()
        for i in range(0, 5):
            layout.setColumnStretch(i, 1)

        self.setLayout(layout)

        # textbox for locating .pgn file to open
        self.pgn_file_txt = QLineEdit(self)
        layout.addWidget(self.pgn_file_txt, 0, 0, 1, 3)

        #
        self.open_file_btn = QPushButton("...")
        self.open_file_btn.clicked.connect(self.getfile)
        layout.addWidget(self.open_file_btn, 0, 4)

        self.back_btn = QPushButton("Back")
        layout.addWidget(self.back_btn, 1, 0, 4, 1)

    def getfile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', str(pc.LICHESS_DB), "PGN files (*.pgn)")
        fname = Path(str(fname[0]))
        self.pgn_file_txt.setText(str(fname))


class basicWindow(QWidget):
    def __init__(self):
        super().__init__()
        grid_layout = QGridLayout()
        self.setLayout(grid_layout)

        button = QPushButton('1-3')
        grid_layout.addWidget(button, 0, 0, 1, 3)

        button = QPushButton('4, 7')
        grid_layout.addWidget(button, 1, 0, -1, 1)

        for x in range(1, 3):
            for y in range(1, 3):
                button = QPushButton(str(str(3*x+y)))
                grid_layout.addWidget(button, x, y)

        self.setWindowTitle('Basic Grid Layout')
