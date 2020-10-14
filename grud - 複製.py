import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
import math


class GRUDCell(nn.Module):
    def __init__(self, input_size_cnn, input_size, hidden_size, bias=True):
        super(GRUDCell, self).__init__()

        self.convfc_size = 4

        self.conv1 = nn.Conv1d(1, 32, kernel_size=2)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=4)
        self.convfc = nn.Linear(32 * 16, self.convfc_size)
        self.drop = nn.Dropout(0.5)

        self.input_size_cnn = input_size_cnn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size
        self.bias = bias
        self.x2h = nn.Linear(input_size + self.convfc_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.m2h = nn.Linear(input_size + self.convfc_size, 3 * hidden_size, bias=bias)
        self.d2r = nn.Linear(input_size + self.convfc_size, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x_cnn, x, mask, delta, hidden):

        # cnn historical price data, to get the trend of the stock
        x_cnn = x_cnn.view(-1, 1, self.input_size_cnn)
        x_conv1 = F.relu(self.conv1(x_cnn))
        x_conv2 = F.relu(self.conv2(x_conv1))
        x_conv2 = x_conv2.view(x_conv2.size(0), -1)
        x_fc = self.convfc(x_conv2)
        x_fc = self.drop(x_fc)

        # GRUD
        x = x.view(-1, x.size(1))
        x = torch.cat((x,x_fc), 1)
        delta = torch.cat((delta, torch.ones((x.size(0),self.convfc_size)).cuda()), 1)
        mask = torch.cat((mask, torch.ones((x.size(0), self.convfc_size)).cuda()), 1)

        rt = torch.exp(-torch.max(torch.zeros(self.hidden_size).cuda(), self.d2r(delta)))

        gate_x = self.x2h(x)
        gate_h = self.h2h(rt * hidden)
        gate_m = self.m2h(mask)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        gate_m = gate_m.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        m_r, m_i, m_n = gate_m.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r + m_r)
        inputgate = F.sigmoid(i_i + h_i + m_i)
        newgate = F.tanh(i_n + (resetgate * h_n) + m_n)

        hy = newgate + inputgate * (hidden - newgate)

        return hy


class GRUDModel(nn.Module):
    def __init__(self, input_dim_cnn, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(GRUDModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.gru_cell = GRUDCell(input_dim_cnn, input_dim, hidden_dim, layer_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self,x_cnn, x, mask, delta):

        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        # print(x.shape,"x.shape")100, 28, 28
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        outs = []

        hn = h0[0, :, :]

        for seq in range(x.size(1)):
            hn = self.gru_cell(x_cnn[:, seq, :], x[:, seq, :], mask[:, seq, :], delta[:, seq, :], hn)
            outs.append(hn)

        out = outs[-1].squeeze()

        out = self.fc(out)
        # out = F.softmax(out, dim=1)
        return out