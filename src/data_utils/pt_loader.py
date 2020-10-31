"""[summary]
"""
import config as cfg

import torch
import numpy as np




class MoveLoader(torch.utils.data.IterableDataset):
    def __init__(self, dataset_path=cfg.LICHESS_DB):
        super(MoveLoader).__init__()
        if not dataset_path.exists():
            raise ValueError(f"{str(dataset_path)} does not exist, please configure config.py to point to a valid path.")

        self.all_pgn_files = list(dataset_path.glob("*.pgn"))

        pgn_file_sizes = [file.stat().st_size for file in self.all_pgn_files]
        self.total_size = np.sum(np.array(pgn_file_sizes, dtype=np.uint))

        self.start = 0
        self.end = len(self.all_pgn_files)

        # worker_pgn_files is reassigned in worker_init_fn if multiple workers are used
        self.worker_pgn_files = self.all_pgn_files

    def __iter__(self):
        # torch.utils.data.DataLoader doesn't handle the Pathlib objects, so convert to a string
        foo = [str(pgn_file) for pgn_file in self.worker_pgn_files]
        return iter(foo)


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id

    dataset = worker_info.dataset  # the dataset copy in this worker process

    size_per_worker = dataset.total_size / worker_info.num_workers

    files_per_worker = []  # used to store the .pgn files for each worker

    # iterate through the dataset .pgn files and evenly partition into {num_workers} parts of equal size (memory)
    files = []
    current_total_size = np.array(0, dtype=np.uint)
    for pgn_file in dataset.all_pgn_files:
        if current_total_size > size_per_worker:
            files_per_worker.append(files)
            files = []
            current_total_size = np.array(0, dtype=np.uint)
        files.append(pgn_file)
        current_total_size = current_total_size + pgn_file.stat().st_size

    if len(files) > 0:  # append the files for the last worker if there are unassigned files left
        files_per_worker.append(files)

    # reassign the .pgn files for this worker's copy of the dataset
    dataset.worker_pgn_files = files_per_worker[worker_id]
