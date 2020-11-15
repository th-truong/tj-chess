"""[summary]
"""
import importlib.util
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
from pathlib import Path

from network_utils.load_tj_model import load_tj_model
from network_utils import network_out_interpreter as noi
from data_utils import pt_loader


def train_tj_chess(args):
    # TODO: add support for continuing training
    # TODO: add support for displaying games on tensorboard as validation
    spec = importlib.util.spec_from_file_location("", args.training_cfg_dir)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    # start tensorboard logging
    writer = SummaryWriter(log_dir=args.log_dir)
    model_save_dir = Path(args.log_dir) / "models"
    if not model_save_dir.exists():
        model_save_dir.mkdir(parents=True)

    # create model
    model = load_tj_model(cfg)
    print(model)
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
    optimizer = cfg.OPTIMIZER(model.parameters(), lr=learning_rate)

    loss_fn_policy = cfg.POLICY_LOSS()
    loss_fn_value = cfg.VALUE_LOSS()

    num_steps = cfg.MAX_ITERATIONS

    for current_step, out in tqdm(enumerate(pt_dataloader)):
        if current_step == 0:
            # set the learning rate very low for warm up
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate / 1000
            scheduler = cfg.SCHEDULER(optimizer, 'min', patience=cfg.SCHEDULER_PATIENCE,
                                      factor=cfg.SCHEDULER_FACTOR)
        elif current_step == cfg.WARM_UP_STEPS:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            scheduler = cfg.SCHEDULER(optimizer, 'min', patience=cfg.SCHEDULER_PATIENCE,
                                      factor=cfg.SCHEDULER_FACTOR)
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
        value_loss = loss_fn_value(pred_values, torch.argmax(targets['result'], dim=1))
        total_loss = 0.99 * policy_loss + 0.01 * value_loss

        writer.add_scalar('loss/train', total_loss, current_step)
        writer.add_scalar('lr', np.max([param_group['lr'] for param_group in optimizer.param_groups]), current_step)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)  # must call this after the optimizer step

        if current_step % cfg.SAVE_FREQ == 0:
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        "global_step": current_step},
                       str(model_save_dir / (str(current_step) + "_steps.tar")))

        if current_step == num_steps:
            break
