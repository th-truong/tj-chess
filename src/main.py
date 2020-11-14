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
from scripts import display_gui
from training_utils.train_tj_chess import train_tj_chess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_gui = subparsers.add_parser('gui')
    parser_gui.set_defaults(func=display_gui)
    parser_gui.add_argument('--lichess-db', default=cfg.LICHESS_DB)
    parser_gui.add_argument('--model')
    parser_gui.add_argument('--stockfish-exe', default=shutil.which('stockfish'))

    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(func=train_tj_chess)
    parser_train.add_argument('--log-dir', default=cfg.LOG_DIR)
    parser_train.add_argument('--training-cfg-dir', help='The config.py file to be used for training.')
    parser_train.add_argument('--lichess-db', default=cfg.LICHESS_DB)

    args = parser.parse_args()

    args.func(args)
    if args.training:
        if args.training_cfg_dir is not None:
            train_tj_chess(args)
        else:
            raise ValueError("--training-cfg-dir must be pointed towards a valid trianing configuration file for training.")
