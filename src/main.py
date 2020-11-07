import config as cfg
from gui import pyqt_classes
from data_utils import pt_loader
from network_utils import network_out_interpreter as noi
from scripts import display_gui

import sys
from pathlib import Path
import torch

import chess
import chess.engine
import chess.pgn


if __name__ == "__main__":
    # display_gui()

    # dataset = pt_loader.MoveLoader()
    # foo = list(torch.utils.data.DataLoader(dataset, num_workers=4, worker_init_fn=pt_loader.worker_init_fn))

    # interpreter = noi.NetInterpreter()
    # noi.test_interpreter(interpreter)

    from network_utils.model_modules import create_vrb_model

    model = create_vrb_model()
    model.eval()
    input_tensor = torch.zeros([2, 112, 8, 8])
    out = model(input_tensor)
