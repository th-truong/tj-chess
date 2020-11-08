import sys
from pathlib import Path

import torch
import numpy as np
import chess
import chess.engine
import chess.pgn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import config as cfg
from gui import pyqt_classes
from data_utils import pt_loader
from network_utils import network_out_interpreter as noi
from network_utils.model_modules import create_vrb_model
from scripts import display_gui


if __name__ == "__main__":
    # start tensorboard logging
    writer = SummaryWriter(log_dir=cfg.LOG_DIR)

    # create model
    model = create_vrb_model()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    # set up network interpreter
    interpreter = noi.NetInterpreter()

    # load dataset objects
    dataset = pt_loader.MoveLoader()
    pt_dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.BATCH_SIZE,
                                                num_workers=cfg.LOADER_WORKERS, worker_init_fn=pt_loader.worker_init_fn)

    # configure training parameters
    learning_rate = cfg.LEARNING_RATE
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn_policy = torch.nn.MSELoss()
    loss_fn_value = torch.nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=cfg.SCHEDULER_PATIENCE,
                                                           factor=cfg.SCHEDULER_FACTOR)
    num_steps = cfg.MAX_ITERATIONS

    for i, out in tqdm(enumerate(pt_dataloader)):
        if i == 0:
            # set the learning rate very low for warm up
            optimizer.defaults['lr'] = learning_rate / 1000
        elif i == cfg.WARM_UP_STEPS:
            optimizer.defaults['lr'] = learning_rate

        moves = out[0].to(device)
        targets = {k: v.to(device) for k, v in out[1].items()}

        pred_policies, pred_values, targets = model(moves, targets)

        target_policies = []
        for i, next_move in enumerate(targets['next_move']):
            target_policy = torch.zeros(73,8,8)
            target_policy[int(next_move[2]), int(next_move[0]), int(next_move[1])] = 1
            target_policies.append(target_policy)
        target_policies = torch.stack(target_policies)

        policy_loss = loss_fn_policy(pred_policies, target_policies)
        value_loss = loss_fn_value(pred_values, targets['result'])
        total_loss = 0.99 * policy_loss + 0.01 * value_loss

        writer.add_scalar('loss/train', total_loss, i)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)  # must call this after the optimizer step

        if i == num_steps:
            break
