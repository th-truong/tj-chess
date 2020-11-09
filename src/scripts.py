import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import chess.pgn

import config as cfg

from gui import pyqt_classes
from network_utils.model_modules import create_tj_model
from network_utils import network_out_interpreter as noi
from network_utils.load_tj_model import load_tj_model
from data_utils import pt_loader
from data_utils.layer_builder import board_to_all_layers


class TjEngine(object):
    """
    not a real chess engine, but looks close enough for now
    """

    def __init__(self, model):
        self.model = model

    def play(self, board, limit=None):
        layers = board_to_all_layers(board.copy())
        input_tensor = torch.from_numpy(layers.astype(np.float32)).unsqueeze(dim=0)
        policy, value, _targets = self.model(input_tensor)

        interpreter = noi.NetInterpreter()
        interpreter.set_colour_to_play('white' if board.turn == chess.WHITE else 'black')

        mask = torch.zeros_like(policy)
        for legal_move in board.legal_moves:
            move_indicies = interpreter.interpret_UCI_move(legal_move.uci())
            print(move_indicies)
            print(mask.shape)
            mask[0, move_indicies[2], move_indicies[0], move_indicies[1]] = 1

        masked_policy = policy * mask

        policy_indicies = np.unravel_index(torch.argmax(masked_policy), policy.shape)

        uci = interpreter.interpret_net_move(policy_indicies[2], policy_indicies[3], policy_indicies[1])
        move = chess.Move.from_uci(uci)

        result = chess.engine.PlayResult(move, None)
        return result


def display_gui(args):
    app = QApplication(sys.argv)

    if args.model is not None:
        model = load_tj_model(args.model)
        model.eval()
        engine = TjEngine(model)
    else:
        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_exe)

    ex = pyqt_classes.chessMainWindow(args.lichess_db, engine)
    ex.show()
    sys.exit(app.exec_())


def train_first_tj_model(args):
    # start tensorboard logging
    writer = SummaryWriter(log_dir=args.log_dir)

    # create model
    model = create_tj_model()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    # set up network interpreter
    interpreter = noi.NetInterpreter()

    # load dataset objects
    dataset = pt_loader.MoveLoader(args.lichess_db)
    pt_dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.BATCH_SIZE,
                                                num_workers=cfg.LOADER_WORKERS, worker_init_fn=pt_loader.worker_init_fn)

    # configure training parameters
    learning_rate = cfg.LEARNING_RATE
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn_policy = torch.nn.CrossEntropyLoss()
    loss_fn_value = torch.nn.CrossEntropyLoss()

    num_steps = cfg.MAX_ITERATIONS

    for current_step, out in tqdm(enumerate(pt_dataloader)):
        if current_step == 0:
            # set the learning rate very low for warm up
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate / 1000
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=cfg.SCHEDULER_PATIENCE,
                                                                   factor=cfg.SCHEDULER_FACTOR)
        elif current_step == cfg.WARM_UP_STEPS:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=cfg.SCHEDULER_PATIENCE,
                                                                   factor=cfg.SCHEDULER_FACTOR)
        moves = out[0].to(device)
        targets = {k: v.to(device) for k, v in out[1].items()}

        pred_policies, pred_values, targets = model(moves, targets)

        target_policies = []
        for i, next_move in enumerate(targets['next_move']):
            target_policy = torch.zeros(73,8,8)
            target_policy[int(next_move[2]), int(next_move[0]), int(next_move[1])] = 1
            target_policies.append(target_policy)
        target_policies = torch.stack(target_policies)

        target_policies_labels = torch.argmax(torch.flatten(target_policies, start_dim=1, end_dim=-1), dim=1)
        pred_policies = torch.flatten(pred_policies, start_dim=1, end_dim=-1)

        policy_loss = loss_fn_policy(pred_policies, target_policies_labels.to(device))
        value_loss = loss_fn_value(pred_values, torch.argmax(targets['result'], dim=1))
        total_loss = 0.99 * policy_loss + 0.01 * value_loss

        writer.add_scalar('loss/train', total_loss, current_step)
        writer.add_scalar('lr', optimizer.defaults['lr'], current_step)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)  # must call this after the optimizer step

        if current_step % 5000 == 0:
            torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "global_step": current_step
            }, str(current_step) + "_steps.tar")

        if current_step == num_steps:
            break



def python_chess_ex():
    pgn = open(cfg.LICHESS_DB / "lichess_elite_2013-09.pgn", encoding="utf-8")

    first_game = chess.pgn.read_game(pgn)
    second_game = chess.pgn.read_game(pgn)

    first_game.headers["Event"]
    'IBM Man-Machine, New York USA'

    # Iterate through all moves and play them on a board.
    board = first_game.board()
    for move in first_game.mainline_moves():
        board.push(move)
        board

    board
