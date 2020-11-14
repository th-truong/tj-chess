"""[summary]
"""
from pathlib import Path
import torch

# ********** Library Paths and Chess Engine Paths ********** #

# points to the location of the .pgn files from the lichess elite db from a reddit user
# found at https://www.reddit.com/r/chess/comments/gz8acg/introducing_the_lichess_elite_database/
LICHESS_DB = Path(r"D:\paper_repos\tj-chess\Lichess Elite Database")

STOCKFISH_ENGINE_PATH = Path(r"D:\paper_repos\tj-chess\stockfish_20090216_x64_bmi2.exe")

# ********** Model Architecture Parameters ********** #

SE_BLOCKS = 20  # number of Squeeze and Excitation layers
SE_CHANNELS = 32
FILTERS = 256

INPUT_SIZE = [112, 8, 8]

# input sizes
HISTORY = 8
SIZE = (8, 8)

# ********** Training Parameters ********** #
LEARNING_RATE = 0.004
BATCH_SIZE = 512
LOADER_WORKERS = 8
MAX_ITERATIONS = 500000
WARM_UP_STEPS = 5000
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5
SAVE_FREQ = 1000

OPTIMIZER = torch.optim.Adam
POLICY_LOSS = torch.nn.CrossEntropyLoss
VALUE_LOSS = torch.nn.CrossEntropyLoss
SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau

LOG_DIR = Path(r"D:\paper_repos\tj-chess\CE_loss_model_tensorboard")
