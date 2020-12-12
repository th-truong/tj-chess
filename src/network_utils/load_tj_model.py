"""[summary]
"""
import sys

import importlib
import torch

from network_utils.model_modules import create_tj_model


def load_tj_model(cfg, weights_path=None, training=False, training_config=None):
    model = create_tj_model(cfg)
    # TODO: don't just load this on cpu!!
    # TODO: return the global step as well

    if weights_path is not None:
        sys.modules['training_config'] = training_config
        if training:
            checkpoint = torch.load(weights_path)
        else:
            checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))

        model.load_state_dict(checkpoint['model'])

        if training:
            return model, checkpoint
        else:
            return model
    else:
        return model
