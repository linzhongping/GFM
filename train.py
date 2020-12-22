import os
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
import pandas as pd
import jieba
import re
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
sns.set(style='white')
style.use("fivethirtyeight")


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset,DataLoader 
from torch.optim import Adam,SGD,RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, kaiming_normal_
from torch.nn.parameter import Parameter
import time
import gc
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
os.environ["CUDA_VISIBLE_DEVICES"] = '2' 


from sklearn import metrics
from utils import * 
import pickle as pkl
datasets = ['mr','ohsumed','R8','R52','weibo_yiqing']
d = 'ohsumed'
if d not in datasets:
    print("error dataset")
else:
    A, X, Y, _, _, _, _, _, _ = load_data(d)



print('loading total set')
A, M = preprocess_adj(A)
X = preprocess_features(X)
# print('loading validation set')
# val_adj, val_mask = preprocess_adj(val_adj)
# val_feature = preprocess_features(val_feature)
# print('loading test set')
# test_adj, test_mask = preprocess_adj(test_adj)
# test_feature = preprocess_features(test_feature)

total_samples = X.shape[0]
total_samples

Y = np.where(Y)[1]


train_adj = torch.Tensor(A[:int(0.8 * total_samples)])
train_feature = torch.Tensor(X[:int(0.8 * total_samples)])
train_y = torch.LongTensor(Y[:int(0.8 * total_samples)])

val_adj = torch.Tensor(A[int(0.8 * total_samples):int(0.9 * total_samples)])
val_feature = torch.Tensor(X[int(0.8 * total_samples):int(0.9 * total_samples)])
val_y = torch.LongTensor(Y[int(0.8 * total_samples):int(0.9 * total_samples)])

test_adj = torch.Tensor(A[int(0.9 * total_samples):])
test_feature = torch.Tensor(X[int(0.9 * total_samples):])
test_y = torch.LongTensor(Y[int(0.9 * total_samples):])

train_mask = torch.Tensor(M[:int(0.8 * total_samples)])
val_mask =torch.Tensor(M[int(0.8 * total_samples):int(0.9 * total_samples)])
test_mask =torch.Tensor(M[int(0.9 * total_samples):])



# split mini-batch
def getBatch(i, bs, A, X, Y,mask):
    return A[i*bs:(i+1)*bs],X[i*bs:(i+1)*bs],Y[i*bs:(i+1)*bs],mask[i*bs:(i+1)*bs]

# parameters
lr = 0.01
batch_size = 32
epochs = 200
weight_decay = 0.

num_class = 23
train_samples = train_y.shape[0]
test_samples = test_y.shape[0]
val_samples = val_y.shape[0]
print(train_samples,test_samples)

from torch.nn.init import xavier_uniform_, kaiming_normal_
class GFMGC(nn.Module):
    def __init__(self, num_class, input_dim, fb_size):
        super(GFMGC,self).__init__()
        
        self.num_class = num_class
        self.input_dim = input_dim
#         self.W = nn.Parameter(torch.FloatTensor(input_dim, fb_size))
        
        self.gru = nn.GRU(self.input_dim,
                          128,
                          bidirectional = True,
                          batch_first = True,
                          bias = True)
    
        self.fc1 = nn.Sequential(nn.Linear(300 + 128 * 2,128),
                                 nn.ReLU(inplace = True),
                                 nn.Dropout(0.5),
                                 nn.Linear(128,64),
                                 nn.ReLU(inplace = True),
                                 nn.Dropout(0.5),
                                 nn.Linear(64,num_class),
                                 
                                 
            )
        self._init_weights()
    
    def cal_gfm(self,x,adj,bs,seq):# x-[bs,seq,emb_size]  adj:[bs,seq,seq]

        left = x.repeat(1,1,seq).view(bs,seq * seq ,-1)
        right = x.repeat(1,seq,1)
        fi = left * right   
        adj = adj.view(bs,-1).unsqueeze(2)
        return torch.sum(fi,dim = 1)
                                                                              
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                kaiming_normal_(p)
        
    def forward(self,x,adj,mask):# x:[bs,seq,emb_size]  adj:[bs,seq,seq]
        x = x * mask
        bs, seq, emb = x.shape
        h = self.gru(x)[0][:,-1,:]
#         print(h.shape)
        gfm = self.cal_gfm(x,adj,bs,seq)
#         gfm = F.dropout(gfm,0.1,training = self.training)
        logit = self.fc1(torch.cat([h,gfm],dim=1))
        return logit

model = GFMGC(num_class = num_class, input_dim = 300,fb_size=100).cuda()

optimizer = Adam(model.parameters(),lr = lr,weight_decay = weight_decay)
lossfunc = nn.CrossEntropyLoss()


def train():
    best_acc = 0.0
    for epoch in range(epochs):
        model.train() 
        print('epoch {}'.format(epoch + 1))
        train_loss = []
        train_acc = 0.
        for i in tqdm(range(train_samples // batch_size + 1)):
            adj_batch,feature_batch, y_batch, mask_batch = getBatch(i, batch_size, train_adj, train_feature, train_y,train_mask)
            optimizer.zero_grad()
            logits = model(feature_batch.cuda(),adj_batch.cuda(),mask_batch.cuda())
            loss = lossfunc(logits, y_batch.cuda())
            
            train_loss.append(loss.item())
            
            pred = torch.max(logits,1)[1]
            
            train_correct = (pred.cpu()  == y_batch).sum()
            
            train_acc += train_correct
            
            loss.backward()
            optimizer.step()
            
        print('train_loss = {:0.4f}, train_acc = {:0.4f}'.format(np.mean(train_loss), train_acc / train_samples))
        
        model.eval()
        
        val_loss = []
        val_acc = 0.
        with torch.no_grad():
            for i in tqdm(range(test_samples // batch_size + 1)):
                adj_batch,feature_batch, y_batch, mask_batch = getBatch(i, batch_size, test_adj, test_feature, test_y,test_mask)
                logits = model(feature_batch.cuda(),adj_batch.cuda(),mask_batch.cuda())
                loss = lossfunc(logits, y_batch.cuda())
                val_loss.append(loss.item())
                pred = torch.max(logits,1)[1]
                val_correct = (pred.cpu() == y_batch).sum()

                val_acc += val_correct
            # best_acc
            if best_acc < val_acc / test_samples:
                best_acc = val_acc / test_samples
            print('test_loss = {:0.4f},  test_acc = {:0.4f}, best_acc = {:0.4f}'.format(np.mean(val_loss), val_acc / test_samples,\
                                                                                     best_acc))

if __name__ == '__main__':
    train()                                                                                  