"""[summary]
"""
from network_utils.model_modules import create_tj_model
import torch


def load_tj_model(cfg, weights_path=None):
    model = create_tj_model(cfg)
    # TODO: don't just load this on cpu!!
    # TODO: return the global step as well

    if weights_path is not None:
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))

        model.load_state_dict(checkpoint['model'])

    return model
