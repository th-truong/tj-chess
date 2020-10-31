import config as cfg
from gui import pyqt_classes
from data_utils import pt_loader
from scripts import display_gui

import sys
from pathlib import Path
import torch

import chess
import chess.engine
import chess.pgn


if __name__ == "__main__":
    # display_gui()
    dataset = pt_loader.MoveLoader()
    foo = list(torch.utils.data.DataLoader(dataset, num_workers=4, worker_init_fn=pt_loader.worker_init_fn))
