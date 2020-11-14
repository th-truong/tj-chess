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
    subparsers = parser.add_subparsers()

    parser_gui = subparsers.add_parser('gui')
    parser_gui.set_defaults(func=display_gui)
    parser_gui.add_argument('--lichess-db', default=cfg.LICHESS_DB)
    parser_gui.add_argument('--model')
    parser_gui.add_argument('--stockfish-exe', default=shutil.which('stockfish'))

    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(func=train_first_tj_model)
    parser_train.add_argument('--log-dir', default=cfg.LOG_DIR)
    parser_train.add_argument('--lichess-db', default=cfg.LICHESS_DB)

    args = parser.parse_args()

    args.func(args)
