"""[summary]
"""
import torch
import numpy as np
import chess
from pathlib import Path

from data_utils import layer_builder
import config as cfg
from network_utils import network_out_interpreter as noi


class MoveLoader(torch.utils.data.IterableDataset):
    def __init__(self, dataset_path):
        super(MoveLoader).__init__()
        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise ValueError(f"{str(dataset_path)} does not exist, please configure config.py to point to a valid path.")

        self.all_pgn_files = list(dataset_path.glob("*.pgn"))  # order from newest -> oldest, which also happens to roughly be largest -> smallest files
        self.all_pgn_files.reverse()

        # worker_pgn_files is reassigned in worker_init_fn if multiple workers are used
        self.worker_pgn_files = self.all_pgn_files

        self.move_interpreter = noi.NetInterpreter()

    def __iter__(self):
        # torch.utils.data.DataLoader doesn't handle the Pathlib objects, so convert to a string
        files_list = [str(pgn_file) for pgn_file in self.worker_pgn_files]
        for file in files_list:
            with open(file) as f:
                game = chess.pgn.read_game(f)
                while game is not None:
                    for meta_layer in layer_builder.game_to_layers(game):
                        turn = meta_layer.meta.turn
                        if turn == chess.WHITE:
                            self.move_interpreter.set_colour_to_play("white")
                        else:
                            self.move_interpreter.set_colour_to_play("black")
                        next_move = torch.Tensor(self.move_interpreter.interpret_UCI_move(str(meta_layer.meta.next_move)))

                        # the result scalar will be [current player winning, draw, other player winning]
                        if meta_layer.meta.result == "1-0":
                            if turn == chess.WHITE:
                                result = torch.Tensor([1, 0, 0])
                            else:
                                result = torch.Tensor([0, 0, 1])
                        elif meta_layer.meta.result == "0-1":
                            if turn == chess.BLACK:
                                result = torch.Tensor([1, 0, 0])
                            else:
                                result = torch.Tensor([0, 0, 1])
                        else:
                            result = torch.Tensor([0, 1, 0])

                        targets = {"next_move": next_move,
                                   "result": result}
                        input_tensor = torch.from_numpy(meta_layer.layers.astype(np.float32))
                        yield input_tensor, targets
                    game = chess.pgn.read_game(f)


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id

    dataset = worker_info.dataset  # the dataset copy in this worker process

    # used to store the .pgn files for each worker
    files_per_worker = [[] for _ in range(0, worker_info.num_workers)]

    for i, pgn_file in enumerate(dataset.all_pgn_files):
        files_per_worker[i % worker_info.num_workers].append(pgn_file)

    dataset.worker_pgn_files = files_per_worker[worker_id]
