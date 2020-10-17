"""[summary]
"""

import torch


class MoveLoader(torch.utils.data.IterableDataset):
    def __init__(self, board, engine):
        super(MoveLoader).__init__()
        self.board = board
        self.engine = engine

    def __iter__():