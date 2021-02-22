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
        self.model.eval()
        self.id = {'name': 'tj chess'}
        self.interpreter = noi.NetInterpreter()

    def play(self, board, limit=None):
        layers = board_to_all_layers(board.copy())
        input_tensor = torch.from_numpy(layers.astype(np.float32)).unsqueeze(dim=0)
        policy, value, _targets = self.model(input_tensor)

        self.interpreter.set_colour_to_play('white' if board.turn == chess.WHITE else 'black')

        mask = torch.zeros_like(policy)
        for legal_move in board.legal_moves:
            move_indicies = self.interpreter.interpret_UCI_move(legal_move.uci())
            mask[0, move_indicies[2], move_indicies[0], move_indicies[1]] = 1

        masked_policy = policy * mask

        policy_indices = np.unravel_index(torch.argmax(masked_policy), policy.shape)

        uci = self.interpreter.interpret_net_move(policy_indices[2], policy_indices[3], policy_indices[1])

        if chess.Move.from_uci(uci) in board.legal_moves:  # this should be true for all cases because of the move mask, minus queen promotions
            move = chess.Move.from_uci(uci)
        else:  # is a queen promotion
            move = chess.Move.from_uci(uci + "q")

        result = chess.engine.PlayResult(move, None)
        return result

    def analyse(self, board, limit: chess.engine.Limit, multipv=5):
        self.interpreter.set_colour_to_play('white' if board.turn == chess.WHITE else 'black')

        layers = board_to_all_layers(board.copy())
        input_tensor = torch.from_numpy(layers.astype(np.float32)).unsqueeze(dim=0)
        policy, value, _targets = self.model(input_tensor)
        policy = policy.cpu().detach().numpy()
        value = value.cpu().detach().numpy().squeeze()

        # find top multipv values, unsorted
        flat_ind = np.argpartition(policy.flatten(), -multipv)[-multipv:]
        # sort
        flat_ind = flat_ind[np.argsort(policy.flatten()[flat_ind])][::-1]
        print(policy.flatten()[flat_ind])

        policy_indices = [np.unravel_index(x, policy.shape) for x in flat_ind]
        uci_moves = [self.interpreter.interpret_net_move(x[2],x[3],x[1]) for x in policy_indices]
        info = [{"score": str(value),
                 "pv": [uci_move]} for uci_move in uci_moves]

        return info

    def close(self):
        pass

    @classmethod
    def load(cls, model_path):
        model = load_tj_model(weights_path=model_path)
        model.eval()
        return cls(model)
