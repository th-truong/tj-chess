"""[summary]
"""
import torch
import numpy as np
import chess
from pathlib import Path

from data_utils import layer_builder
import config as cfg
from network_utils import network_out_interpreter as noi
from data_utils.loader import stream_games


class MoveLoader(torch.utils.data.IterableDataset):
    def __init__(self, dataset_path):
        super(MoveLoader).__init__()
        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise ValueError(f"{str(dataset_path)} does not exist, please configure config.py to point to a valid path.")

        self.dataset_path = dataset_path
        self.move_interpreter = noi.NetInterpreter()

        self.stream_games_fn = stream_games
        self.game_generator = None  # reassigned later

    def __iter__(self):
        for game in self.game_generator:
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


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id

    dataset = worker_info.dataset  # the dataset copy in this worker process

    dataset.game_generator = dataset.stream_games_fn(dataset.dataset_path, shard_index=worker_id, total_shards=worker_info.num_workers)
