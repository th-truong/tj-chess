"""[summary]
"""

import torch


class MoveLoader(torch.utils.data.IterableDataset):
    def __init__(self, board, engine):
        super(MoveLoader).__init__()
        self.board = board
        self.engine = engine

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if work_info is None: # single process loading
            pass
        else:
