import os
import pdb
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

random.seed(1)
np.random.seed(1)

import torch
torch.manual_seed(1)
from torch import nn
from torch import optim
from torch.autograd import Variable

from baseline import fixed_prediction
from DataLoader import DataLoader
from utils import *

class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias).cuda()

    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())


class LSTM_BKGD(nn.ModuleList):
    def __init__(self, hidden_dim, kernel_size, num_layers, input_dim=1,
                 bias=True, return_all_layers=False, input_size=(1080//2, 1920//2)):
        super(LSTM_BKGD, self).__init__()
        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)

        self.bias = bias
        self.input_dim  = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.height, self.width = input_size

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    def forward(self, input_tensor):
        # (b, seq, c, h, w)
        layer_output_list, last_state_list = [], []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # pdb.set_trace()
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        return layer_output_list[-1:], last_state_list[-1:]

model = LSTM_BKGD(hidden_dim=[64, 64, 128],
                 kernel_size=(3, 3),
                 num_layers=3)

a = np.array(Image.open('/home/demo/carla_sam_newest/carla/Output/1_2/Background/camera-1347-image-0.png'))
a = Variable(torch.from_numpy(a).float()).cuda()
a = a.unsqueeze(0)
a = a.unsqueeze(0)
a = a.unsqueeze(0)
# a = np.expand_dims(a, 0)
# a = np.expand_dims(a, 0)
# a = np.expand_dims(a, 0)

print a.shape
q1, q2 = model(a)
print q1
print q2
