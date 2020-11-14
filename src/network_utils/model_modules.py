"""[summary]https://lczero.org/dev/backend/nn/#network-topology
"""
import torch
import numpy as np


def create_tj_model(cfg):
    # TODO: modify this function to create a model given a config object that contains the model hyperparameters
    input_module = TJInputModule(cfg)
    policy_head = PolicyHead(cfg)
    value_head = ValueHead(cfg)
    body = TJSEBlockBody(cfg)
    model = TJChessModel(input_module=input_module,
                         body=body,
                         policy_head=policy_head,
                         value_head=value_head)

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

        policy = self.policy_head(body_out)
        value = self.value_head(body_out)

        return policy, value, targets


class TJInputModule(torch.nn.Module):

    def __init__(self, cfg):
        super(TJInputModule, self).__init__()
        self.input_size = cfg.INPUT_SIZE
        self.input_filters = cfg.FILTERS
        self.input_conv2d = torch.nn.Conv2d(self.input_size[0], self.input_filters,
                                            kernel_size=3, padding=1, stride=1)

    def forward(self, input_board):
        molded_input = self.input_conv2d(input_board)
        return molded_input


class TJSEBlockBody(torch.nn.Module):

    def __init__(self, cfg):
        super(TJSEBlockBody, self).__init__()
        self.num_blocks = cfg.SE_BLOCKS
        self.num_filters = cfg.FILTERS
        self.num_se_channels = cfg.SE_CHANNELS

        self.se_blocks = torch.nn.ModuleList([SEBlock(cfg)
                                             for i in range(0, self.num_blocks)])
        self.sequential_se_blocks = torch.nn.Sequential(*self.se_blocks)

    def forward(self, molded_input):
        features_out = self.sequential_se_blocks(molded_input)
        return features_out


class SEBlock(torch.nn.Module):

    def __init__(self, cfg):
        super(SEBlock, self).__init__()
        self.num_filters = cfg.FILTERS
        self.num_se_channels = cfg.SE_CHANNELS

        self.conv2d_0 = torch.nn.Conv2d(self.num_filters, self.num_filters,
                                        kernel_size=3, padding=1, stride=1)
        self.conv2d_1 = torch.nn.Conv2d(self.num_filters, self.num_filters,
                                        kernel_size=3, padding=1, stride=1)

        # used to reduce the [batch_size, channels, 8, 8] to [batch_size, channels, 1, 1]
        self.avg_pool = torch.nn.AvgPool2d(8)

        self.flatten = torch.nn.Flatten()

        self.fc0 = torch.nn.Linear(self.num_filters, self.num_se_channels)
        self.relu0 = torch.nn.ReLU(inplace=True)

        self.fc1 = torch.nn.Linear(self.num_se_channels, self.num_filters * 2)

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


class ValueHead(torch.nn.Module):

    def __init__(self, cfg):
        super(ValueHead, self).__init__()
        self.num_filters = cfg.FILTERS

        self.conv2d_0 = torch.nn.Conv2d(self.num_filters, 32,
                                        kernel_size=3, padding=1, stride=1)
        self.conv2d_1 = torch.nn.Conv2d(32, 128,
                                        kernel_size=8, padding=0, stride=1)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.fc0 = torch.nn.Linear(128, 3)
        self.sm0 = torch.nn.Softmax(1)

    def forward(self, input_tensor):
        conv2d_0_out = self.conv2d_0(input_tensor)
        conv2d_1_out = self.conv2d_1(conv2d_0_out)
        relu1_out = self.relu1(conv2d_1_out).squeeze(2).squeeze(2)

        fc0_out = self.fc0(relu1_out)

        if self.training:
            # for training we do no use the softmax because torch.nn.CrossEntropyLoss applies a log-softmax
            # TODO: add this as a layer into the network later and just return losses when self.training is true
            value_head_out = fc0_out
        else:
            value_head_out = self.sm0(fc0_out)

        return value_head_out


class PolicyHead(torch.nn.Module):

    def __init__(self, cfg):
        super(PolicyHead, self).__init__()
        self.num_filters = cfg.FILTERS

        self.conv2d_0 = torch.nn.Conv2d(self.num_filters, self.num_filters,
                                        kernel_size=3, padding=1, stride=1)
        self.conv2d_1 = torch.nn.Conv2d(self.num_filters, 73,
                                        kernel_size=3, padding=1, stride=1)

    def forward(self, input_tensor):
        conv2d_0_out = self.conv2d_0(input_tensor)
        policy_head_out = self.conv2d_1(conv2d_0_out)
        return policy_head_out


if __name__ == '__main__':
    model = create_tj_model()
    dummy = torch.rand(2, 112, 8, 8)

    policy, value, targets = model(dummy, targets=1)

    print(f'policy shape: {policy.shape}')
    print(f'value shape: {value.shape}')
