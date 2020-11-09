#!/usr/bin/env python
import sys
from pathlib import Path
import argparse
import shutil

import torch
import numpy as np
import chess
import chess.engine
import chess.pgn


import config as cfg
from scripts import display_gui, train_first_tj_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--log-dir', default=cfg.LOG_DIR)
    parser.add_argument('--lichess-db', default=cfg.LICHESS_DB)
    parser.add_argument('--stockfish-exe', default=shutil.which('stockfish'))
    args = parser.parse_args()

    if args.gui:
        display_gui(args)

    if args.training:
        train_first_tj_model(args)
