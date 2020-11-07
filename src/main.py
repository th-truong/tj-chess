import config as cfg
from gui import pyqt_classes
from data_utils import pt_loader
from network_utils import network_out_interpreter as noi
from scripts import display_gui

import sys
from pathlib import Path
import torch
import numpy as np

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
    from data_utils import layer_builder

    model = create_vrb_model()
    model.eval()
    pgn_path = r"D:\paper_repos\tj-chess\Lichess Elite Database\lichess_elite_2020-05.pgn"
    with open(pgn_path) as f:
        game = "holder"
        while game is not None:
            game = chess.pgn.read_game(f)
            print(game.headers)
            for i, meta_layers in enumerate(layer_builder.game_to_layers(game)):
                input_tensor = meta_layers.layers
                input_tensor = torch.from_numpy(input_tensor.astype(np.float32)).unsqueeze(dim=0)
                out = model(input_tensor)
                print(i)
