"""[summary]
"""
from pathlib import Path


# ********** Library Paths and Chess Engine Paths ********** #

# points to the location of the .pgn files from the lichess elite db from a reddit user
# found at https://www.reddit.com/r/chess/comments/gz8acg/introducing_the_lichess_elite_database/
LICHESS_DB = Path(r"D:\paper_repos\tj-chess\Lichess Elite Database")

STOCKFISH_ENGINE_PATH = Path(r"D:\paper_repos\tj-chess\stockfish_20090216_x64_bmi2.exe")

# ********** Model Architecture Parameters ********** #

SE_BLOCKS = 10  # number of Squeeze and Excitation layers
SE_CHANNELS = 32
FILTERS = 128

INPUT_SIZE = [112, 8, 8]
