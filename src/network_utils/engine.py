import os.path

import chess
import torch
import numpy as np
from engine.mcts import mcts, Node, build_chess_state_simulator, expand_state_chess

from network_utils import network_out_interpreter as noi
from network_utils.load_tj_model import load_tj_model
from data_utils.layer_builder import board_to_all_layers


class TjMctsEngine:
    def __init__(self, model, name=None):
        self.model = model
        self.model.eval()
        self.id = {'name': f'tj mcts - {name}'}
        self.root = None

    def analyse(self, board, limit: chess.engine.Limit, multipv=5):
        return []

    def play(self, board, limit=None):
        if self.root is None or self.root.state != board:
            self.root = Node(board, None)
        with torch.no_grad():
            simulate_states_chess = build_chess_state_simulator(self.model)
            uci, _value = mcts(self.root, expand_state_chess, simulate_states_chess)
        self.root = self.root.children[uci]
        self.root.parent = None
        move = chess.Move.from_uci(uci)
        result = chess.engine.PlayResult(move, None)
        return result

    def close(self):
        pass

    @classmethod
    def load(cls, model_path, name=None):
        model = load_tj_model(weights_path=model_path)
        model = model.to(torch.device('cuda'))
        model.eval()
        if name is None:
            name = os.path.basename(model_path)
        return cls(model, name=name)


class TjPolicyEngine(object):
    """
    not a real chess engine, but looks close enough for now
    """
    def __init__(self, model, name=None):
        self.model = model
        self.model.eval()
        self.id = {'name': f'tj policy - {name}'}
        self.interpreter = noi.NetInterpreter()

    def play(self, board, limit=None):
        layers = board_to_all_layers(board.copy())
        input_tensor = torch.from_numpy(layers.astype(np.float32)).unsqueeze(dim=0)
        input_tensor = input_tensor.to(torch.device('cuda'))
        policy, value, _targets = self.model(input_tensor)

        self.interpreter.set_colour_to_play('white' if board.turn == chess.WHITE else 'black')

        policy = policy.cpu()

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
        input_tensor = input_tensor.to(torch.device('cuda'))
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
    def load(cls, model_path, name=None):
        model = load_tj_model(weights_path=model_path)
        model = model.to(torch.device('cuda'))
        model.eval()
        if name is None:
            name = os.path.basename(model_path)
        return cls(model, name=name)
