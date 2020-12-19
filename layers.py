import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import math


class GraphConvolution(Module):
    """
    Simple GCN layer
    """
    def __init__(self, input, output, bias=True):
        super(GraphConvolution, self).__init__()
        self.input = input
        self.output = output
        self.weight = Parameter(torch.FloatTensor(input, output))

        if bias:
            self.bias = Parameter(torch.FloatTensor(output))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = np.sqrt(6.0 / (self.weight.shape[0] + self.weight.shape[1]))
        # stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphAttentionLayer(nn.Module):  #for class
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(in_features, out_features))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)



        self.a = nn.Parameter(torch.zeros(2*out_features, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)


        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)

        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
