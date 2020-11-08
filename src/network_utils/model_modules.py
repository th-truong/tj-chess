"""[summary]https://lczero.org/dev/backend/nn/#network-topology
"""
import torch
import config as cfg
import numpy as np

def create_vrb_model():

    input_module = TJInputModule()
    body = TJSEBlockBody()
    model = TJChessModel(input_module=input_module,
                         body=body,
                         policy_head=None,
                         value_head=None)

    return model


class TJChessModel(torch.nn.Module):

    def __init__(self, input_module, body, policy_head, value_head):
        super(TJChessModel, self).__init__()
        self.input_module = input_module
        self.body = body
        self.policy_head = policy_head
        self.value_head = value_head

    def forward(self, input_board, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        molded_input_board = self.input_module(input_board)

        body_out = self.body(molded_input_board)

        # policy = self.policy_head(body_out)

        # value = self.value_head(body_out)

        policy = body_out
        value = None

        return policy, value, targets


class TJInputModule(torch.nn.Module):

    def __init__(self, input_size=cfg.INPUT_SIZE, input_filters=cfg.FILTERS):
        super(TJInputModule, self).__init__()
        self.input_conv2d = torch.nn.Conv2d(input_size[0], input_filters,
                                            kernel_size=3, padding=1, stride=1)

    def forward(self, input_board):
        molded_input = self.input_conv2d(input_board)
        return molded_input


class TJSEBlockBody(torch.nn.Module):

    def __init__(self, num_blocks=cfg.SE_BLOCKS, num_filters=cfg.FILTERS, num_se_channels=cfg.SE_CHANNELS):
        super(TJSEBlockBody, self).__init__()
        self.num_blocks = num_blocks
        self.num_filters = num_filters
        self.num_se_channels = num_se_channels

        self.se_blocks = torch.nn.ModuleList([SEBlock(num_filters=self.num_filters, num_se_channels=self.num_se_channels)
                                             for i in range(0, self.num_blocks)])
        self.sequential_se_blocks = torch.nn.Sequential(*self.se_blocks)

    def forward(self, molded_input):
        features_out = self.sequential_se_blocks(molded_input)
        return features_out


class SEBlock(torch.nn.Module):

    def __init__(self, num_filters=cfg.FILTERS, num_se_channels=cfg.SE_CHANNELS):
        super(SEBlock, self).__init__()
        self.num_filters = num_filters
        self.num_se_channels = num_se_channels

        self.conv2d_0 = torch.nn.Conv2d(self.num_filters, self.num_filters,
                                        kernel_size=3, padding=1, stride=1)
        self.conv2d_1 = torch.nn.Conv2d(self.num_filters, self.num_filters,
                                        kernel_size=3, padding=1, stride=1)

        # used to reduce the [batch_size, channels, 8, 8] to [batch_size, channels, 1, 1]
        self.avg_pool = torch.nn.AvgPool2d(8)

        self.flatten = torch.nn.Flatten()

        self.fc0 = torch.nn.Linear(num_filters, num_se_channels)
        self.relu0 = torch.nn.ReLU(inplace=True)

        self.fc1 = torch.nn.Linear(num_se_channels, num_filters * 2)

        self.sigmoid = torch.nn.Sigmoid()

        self.relu1 = torch.nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        conv2d_0_out = self.conv2d_0(input_tensor)
        conv2d_1_out = self.conv2d_1(conv2d_0_out)

        avg_out = self.avg_pool(conv2d_1_out)
        flatten_out = self.flatten(avg_out)
        fc0_out = self.fc0(flatten_out)
        relu0_out = self.relu0(fc0_out)
        fc1_out = self.fc1(fc0_out)
        ff1_out_split = torch.split(fc1_out, split_size_or_sections=[self.num_filters, self.num_filters], dim=1)

        z = self.sigmoid(ff1_out_split[0]).unsqueeze(-1).unsqueeze(-1)  # z = sigmoid(w)
        b = ff1_out_split[1].unsqueeze(-1).unsqueeze(-1)

        SE_out = (z * input_tensor) + b  # (Z * input) + B

        residual_out = SE_out + input_tensor

        se_block_out = self.relu1(SE_out)
        return se_block_out