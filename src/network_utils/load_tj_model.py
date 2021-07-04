"""[summary]
"""
import sys

import importlib
import torch

from network_utils.model_modules import create_tj_model


def load_tj_model(cfg=None, weights_path=None, training=False, training_config=None):
    if weights_path is not None:
        checkpoint = torch.load(weights_path)

        model = create_tj_model(checkpoint['cfg'])
        model.load_state_dict(checkpoint['model'])

        if training:
            return model, checkpoint
        else:
            return model
    elif cfg is not None:
        model = create_tj_model(cfg)
        return model
