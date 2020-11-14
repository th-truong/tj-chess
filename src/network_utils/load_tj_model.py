"""[summary]
"""
from network_utils.model_modules import create_tj_model
import torch


def load_tj_model(weights_path):
    model = create_tj_model()
    # TODO: don't just load this on cpu!!
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model'])

    return model
