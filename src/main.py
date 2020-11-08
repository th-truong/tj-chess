import config as cfg
from gui import pyqt_classes
from data_utils import pt_loader
from network_utils import network_out_interpreter as noi
from network_utils.model_modules import create_vrb_model
from scripts import display_gui

import sys
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

import chess
import chess.engine
import chess.pgn


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
    loss_fn = torch.nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=cfg.SCHEDULER_PATIENCE,
                                                           factor=cfg.SCHEDULER_FACTOR)
    num_steps = cfg.MAX_ITERATIONS

    for i, out in tqdm(enumerate(pt_dataloader)):
        if i == 0:
            # set the learning rate very low for warm up
            optimizer.defaults['lr'] = learning_rate / 1000
        elif i == cfg.WARM_UP_STEPS:
            optimizer.defaults['lr'] = learning_rate

        moves = [move.to(device) for move in out[0]]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        pred_move, loss = model(move, targets)

        writer.add_scalar('loss/train', loss, i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)  # must call this after the optimizer step

        if i == num_steps:
            break
