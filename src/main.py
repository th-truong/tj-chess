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
from scripts import display_gui, benchmark
from training_utils.train_tj_chess import train_tj_chess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    parser_gui = subparsers.add_parser('gui')
    parser_gui.set_defaults(func=display_gui)
    parser_gui.add_argument('--chess-db', default=cfg.CHESS_DB)
    parser_gui.add_argument('--model')
    parser_gui.add_argument('--training-cfg-dir')
    parser_gui.add_argument('--stockfish-exe', default=shutil.which('stockfish'))

    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(func=train_tj_chess)
    parser_train.add_argument('--log-dir', default=cfg.LOG_DIR)
    parser_train.add_argument('--training-cfg-dir', required=True, help='The config.py file to be used for training.')
    parser_train.add_argument('--chess-db', default=cfg.CHESS_DB)

    parser_benchmark = subparsers.add_parser('benchmark')
    parser_benchmark.set_defaults(func=benchmark)
    parser_benchmark.add_argument('--model', required=True)
    parser_benchmark.add_argument('-n', type=int, default=1)

    args = parser.parse_args()

    args.func(args)
