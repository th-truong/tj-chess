import chess
import torch
import numpy as np

from network_utils.model_modules import create_tj_model
from network_utils import network_out_interpreter as noi
from network_utils.load_tj_model import load_tj_model
from data_utils.layer_builder import board_to_all_layers


class TjEngine(object):
    """
    not a real chess engine, but looks close enough for now
    """

    def __init__(self, model):
        self.model = model
        self.id = {'name': 'tj chess'}

    def play(self, board, limit=None):
        layers = board_to_all_layers(board.copy())
        input_tensor = torch.from_numpy(layers.astype(np.float32)).unsqueeze(dim=0)
        policy, value, _targets = self.model(input_tensor)

        interpreter = noi.NetInterpreter()
        interpreter.set_colour_to_play('white' if board.turn == chess.WHITE else 'black')

        mask = torch.zeros_like(policy)
        for legal_move in board.legal_moves:
            move_indicies = interpreter.interpret_UCI_move(legal_move.uci())
            mask[0, move_indicies[2], move_indicies[0], move_indicies[1]] = 1

        masked_policy = policy * mask

        policy_indicies = np.unravel_index(torch.argmax(masked_policy), policy.shape)

        uci = interpreter.interpret_net_move(policy_indicies[2], policy_indicies[3], policy_indicies[1])
        print(value)
        print(uci)
        # TODO: figure out why the interpreter doesn't mask out illegal moves properly when kings are on the corners of the board at the end
        move = chess.Move.from_uci(uci)

        result = chess.engine.PlayResult(move, None)
        return result

    def close(self):
        pass

    @classmethod
    def load(cls, model_path, cfg):
        model = load_tj_model(cfg=cfg, weights_path=model_path)
        model.eval()
        return cls(model)
