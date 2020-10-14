#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gru2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
from utils import *
import matplotlib.pyplot as plt
import math

batch_size = 16
n_iters = 6000
# num_epochs = n_iters / (len(train_dataset) / batch_size)
# num_epochs = int(num_epochs)
num_epochs = 20

input_dim_cnn = 20
input_dim = 12
hidden_dim = 128
layer_dim = 1  
output_dim = 3
seq_dim = 20

# model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
model = gru2.GRUDModel(input_dim_cnn, input_dim, hidden_dim, layer_dim, output_dim)

if torch.cuda.is_available():
    model.cuda()
    
# loss function and optimizer
# criterion = nn.MSELoss()  # regression
criterion = nn.CrossEntropyLoss()  # classification (包含softmax)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

train_loss_list = []
val_loss_list = []


# In[2]:


def test(data, X_data, Y_data, model, batch_size):
    model.eval()
    total_loss = 0
    n_samples = 0
    predict = None
    test = None
    predict_list = None
    test_list = None
    for X, Y in data.get_batches(X_data, Y_data, batch_size, False):
        output = model(X[:,:, input_dim*3:], X[:,:,:input_dim], X[:,:,input_dim:input_dim*2], X[:,:,input_dim*2:input_dim*3])
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))
    
#     print(test.shape,predict.shape)
#     x_axis = [x+1 for x in range(len(predict))]
#     plt.plot(x_axis,predict.cpu().detach().numpy() ,x_axis,test.cpu().detach().numpy() )
#     plt.show()
    predict = [x.index(max(x)) for x in predict.tolist()]
    result = {0:[0,0],1:[0,0],2:[0,0]}
    test = test.tolist()
    for i in range(len(predict)):
        if test[i][0] == predict[i]:
            result[test[i][0]][1] += 1
        result[test[i][0]][0] += 1
    print(result)
    


# In[3]:


def loss_graph():
    x_axis = [x+1 for x in range(len(train_loss_list))]
    plt.plot(x_axis,train_loss_list,x_axis,val_loss_list)
    plt.show()


# In[4]:


def evaluate(data, X_data, Y_data, model, criterion, batch_size):
    model.eval()
    total_loss = 0
    n_samples = 0
    predict = None
    test = None
    for X, Y in data.get_batches(X_data, Y_data, batch_size, False):
        output = model(X[:,:, input_dim*3:], X[:,:,:input_dim], X[:,:,input_dim:input_dim*2], X[:,:,input_dim*2:input_dim*3])
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))
        
        # Y = torch.tensor(Y, dtype=torch.long).clone().cuda().squeeze()
        Y = Y.long().resize(Y.size()[0])
        total_loss += criterion(output, Y)
        # n_samples += (output.size(0) * data.m)
        n_samples += (output.size(0) * 1)
    val_loss_list.append(total_loss / n_samples)
    return total_loss/n_samples


# In[5]:


def train(data, X_data, Y_data, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    for X, Y in data.get_batches(X_data, Y_data, batch_size, True):
        model.zero_grad()
        output = model(X[:,:, input_dim*3:], X[:,:,:input_dim], X[:,:,input_dim:input_dim*2], X[:,:,input_dim*2:input_dim*3])

        # Y = torch.as_tensor(Y, dtype=torch.long).cuda().squeeze()
#         print(output.shape,Y.shape)
        # print(output,Y)
        Y = Y.long().resize(Y.size()[0])
        loss = criterion(output, Y)
        loss.backward()
        optim.step()
        total_loss += loss.data  # adjust
        n_samples += (output.size(0) * 1)
    train_loss_list.append(total_loss / n_samples)
    return total_loss / n_samples


# In[6]:


Data = Data_utility('./data/stock.txt', 0.6, 0.2, True, 1, 1)
for epoch in range(1,num_epochs+1):
#     print(Data.train[0].shape,Data.train[1].shape)
    train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optimizer, batch_size)
    val_loss = evaluate(Data, Data.valid[0], Data.
                        valid[1], model, criterion, batch_size)
    if epoch % 10 == 0:
        print(epoch,train_loss.item(),val_loss.item())
loss_graph()
test(Data, Data.valid[0], Data.valid[1], model, batch_size)
# test_loss = evaluate(Data, Data.test[0], Data.test[1], model, criterion, batch_size)
# print(test_loss)



# In[7]:


test(Data, Data.valid[0], Data.valid[1], model, batch_size)

