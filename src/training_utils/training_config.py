from pathlib import Path
import torch

from network_utils import model_modules as mm

# ********** Model Architecture Parameters ********** #

INPUT_MODULE = mm.TJInputModule
INPUT_MODULE_KWARGS = {'input_size': [112, 8, 8],
                       'input_filters': 128}

BODY_MODULE = mm.TJSEBlockBody
BODY_MODULE_KWARGS = {'num_blocks': 10,  # number of squeeze and excitation layers
                      'num_filters': 128,
                      'num_se_channels': 32}

POLICY_MODULE = mm.PolicyHead
POLICY_MODULE_KWARGS = {'num_filters': 128}

VALUE_MODULE = mm.ValueHead
VALUE_MODULE_KWARGS = {'num_filters': 128}


# input sizes
HISTORY = 8
SIZE = (8, 8)

# ********** Training Parameters ********** #
LEARNING_RATE = 0.0001
BATCH_SIZE = 2000
LOADER_WORKERS = 1
MAX_ITERATIONS = 500000
WARM_UP_STEPS = 20000
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5
SAVE_FREQ = 2000

OPTIMIZER = torch.optim.Adam
OPTIMIZER_KWARGS = {'lr': 0.0002}

SCHEDULER = None
SCHEDULER_KWARGS = None

POLICY_LOSS = torch.nn.CrossEntropyLoss
VALUE_LOSS = torch.nn.MSELoss

tj_train_config = {'history': HISTORY,
                   'size': SIZE,
                   'input_module': INPUT_MODULE,
                   'input_module_kwargs': INPUT_MODULE_KWARGS,
                   'body_module': BODY_MODULE,
                   'body_module_kwargs': BODY_MODULE_KWARGS,
                   'policy_module': POLICY_MODULE,
                   'policy_module_kwargs': POLICY_MODULE_KWARGS,
                   'value_module': VALUE_MODULE,
                   'value_module_kwargs': VALUE_MODULE_KWARGS,
                   # training configs
                   'batch_size': BATCH_SIZE,
                   'loader_workers': LOADER_WORKERS,
                   'max_iterations': MAX_ITERATIONS,
                   'warm_up_steps': WARM_UP_STEPS,
                   'scheduler_patience': SCHEDULER_PATIENCE,
                   'scheduler_factor': SCHEDULER_FACTOR,
                   'save_freq': SAVE_FREQ,
                   'optimizer': OPTIMIZER,
                   'optimizer_kwargs': OPTIMIZER_KWARGS,
                   'scheduler': SCHEDULER,
                   'scheduler_kwargs': SCHEDULER_KWARGS,
                   'policy_loss': POLICY_LOSS,
                   'value_loss': VALUE_LOSS
                   }
