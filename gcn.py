import torch.nn as nn

from layers import GraphConvolution
import torch.nn.functional as F
class GCN(nn.Module):
    def __init__(self,inputfeature,hidden_units,nclass,dropout):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(inputfeature,hidden_units)

        self.gc2 = GraphConvolution(hidden_units,nclass)

    def forward(self,x,adj ):
        x = self.gc1(x,adj)
        x = F.dropout(x,self.dropout)
        x = self.gc2(x,adj)
        return F.log_softmax(x, dim=1)

