"""[summary]
"""
import sys

import importlib
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np

from network_utils.load_tj_model import load_tj_model
from network_utils import network_out_interpreter as noi
from data_utils import pt_loader


def train_tj_chess(args):
    # TODO: add support for displaying games on tensorboard as validation
    spec = importlib.util.spec_from_file_location("training_config", args.training_cfg_dir)
    training_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training_config)
    sys.modules['training_config'] = training_config

    cfg = training_config.tj_train_config

    # start tensorboard logging
    writer = SummaryWriter(log_dir=args.log_dir)
    model_save_dir = Path(args.log_dir) / "models"
    if model_save_dir.exists() and len(list(model_save_dir.glob("*.tar"))) > 0:
        last_model_path = list(model_save_dir.glob("*.tar"))[-1]
        continue_training_flag = True  # used to configure training to continue or to start fresh
    elif model_save_dir.exists():
        continue_training_flag = False
    else:
        model_save_dir.mkdir(parents=True)
        continue_training_flag = False

    # create model
    if continue_training_flag:
        model, checkpoint = load_tj_model(cfg, weights_path=str(last_model_path), training=True, training_config=training_config)
        cfg = checkpoint['cfg']
    else:
        model = load_tj_model(cfg, training=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    # set up network interpreter
    interpreter = noi.NetInterpreter()

    # load dataset objects
    dataset = pt_loader.MoveLoader(args.lichess_db)
    pt_dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'],
                                                num_workers=cfg['loader_workers'], worker_init_fn=pt_loader.worker_init_fn)

    # configure training parameters
    if continue_training_flag:
        optimizer = cfg['optimizer'](model.parameters(), **cfg['optimizer_kwargs'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        optimizer = cfg['optimizer'](model.parameters(), **cfg['optimizer_kwargs'])

    loss_fn_policy = cfg['policy_loss']()
    loss_fn_value = cfg['value_loss']()

    num_steps = cfg['max_iterations']

    if continue_training_flag:
        current_step = checkpoint['global_step']
    else:
        current_step = 0

    # TODO: clean up this training loop, probably put it into a separate function
    for out in tqdm(pt_dataloader):
        if continue_training_flag:
            if current_step == cfg['warm_up_steps']:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 100
                if cfg['scheduler'] is not None:
                    scheduler = cfg['scheduler'](optimizer, **cfg['scheduler_kwargs'])
        else:
            if current_step == 0:
                # set the learning rate very low for warm up
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cfg['optimizer_kwargs']['lr'] / 100
                if cfg['scheduler'] is not None:
                    scheduler = cfg['scheduler'](optimizer, **cfg['scheduler_kwargs'])
            elif current_step == cfg['warm_up_steps']:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 100
                if cfg['scheduler'] is not None:
                    scheduler = cfg['scheduler'](optimizer, **cfg['scheduler_kwargs'])
        moves = out[0].to(device)
        targets = {k: v.to(device) for k, v in out[1].items()}

        pred_policies, pred_values, targets = model(moves, targets)

        target_policies = []
        for i, next_move in enumerate(targets['next_move']):
            target_policy = torch.zeros(73, 8, 8)
            target_policy[int(next_move[2]), int(next_move[0]), int(next_move[1])] = 1
            target_policies.append(target_policy)
        target_policies = torch.stack(target_policies)

        target_policies_labels = torch.argmax(torch.flatten(target_policies, start_dim=1, end_dim=-1), dim=1)
        pred_policies = torch.flatten(pred_policies, start_dim=1, end_dim=-1)

        policy_loss = loss_fn_policy(pred_policies, target_policies_labels.to(device))
        value_loss = loss_fn_value(pred_values, targets['result'])
        total_loss = 0.99 * policy_loss + 0.01 * value_loss

        writer.add_scalar('loss/train', total_loss, current_step)
        writer.add_scalar('lr', np.mean([param_group['lr'] for param_group in optimizer.param_groups]), current_step)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if cfg['scheduler'] is not None:
            scheduler.step(total_loss)  # must call this after the optimizer step

        if current_step % cfg['save_freq'] == 0:
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        "global_step": current_step,
                        'cfg': cfg},
                       str(model_save_dir / (str(current_step).zfill(10) + ".tar")))

        if current_step == num_steps:
            break

        current_step += 1

        if (model_save_dir / 'pause.txt').exists():
            model.to('cpu')
            while (model_save_dir / 'pause.txt').exists():
                pass
            model.to(device)  # bring model back to gpu
