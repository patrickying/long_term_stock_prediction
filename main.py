import gru2
import gru
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
from utils import *
import time

import math


def evaluate(data, X, Y, model, criterion, batch_size):
    model.eval()
    loss_list = []
    n_samples = 0
    correct = 0
    total = 0
    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X)
        _, predict = torch.max(output.data, 1)
        test = Y

        total += len(Y)
        for i,d in enumerate(predict):
            if d == test[i]: correct += 1


        Y = torch.tensor(Y, dtype=torch.long).clone().cuda().squeeze()
        loss = criterion(output, Y)
        loss_list.append(loss.item())
        # n_samples += (output.size(0) * data.m)
        loss_list[-1] /= (output.size(0) * 1)
        # n_samples += (output.size(0) * 1)

    return sum(loss_list),correct,total

def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    loss_list = []
    n_samples = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        output = model(X)

        Y = torch.as_tensor(Y, dtype=torch.long).cuda().squeeze()
        # print(output.shape,Y.shape)
        # print(output,Y)

        loss = criterion(output, Y)
        loss.cuda()
        loss.backward()
        optim.step()
        # total_loss += loss.data  # adjust
        loss_list.append(loss.item())
        loss_list[-1] /= (output.size(0) * 1)
        # n_samples += (output.size(0) * 1)
    return sum(loss_list)


batch_size = 128
# n_iters = 6000
# num_epochs = n_iters / (len(train_dataset) / batch_size)
# num_epochs = int(num_epochs)
num_epochs = 5

input_dim_cnn = 20
input_dim = 3
hidden_dim = 128
layer_dim = 1  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 3
seq_dim = 20

# model = gru2.GRUDModel(input_dim_cnn, input_dim, hidden_dim, layer_dim, output_dim)
model = gru.GRUModel(input_dim, hidden_dim, layer_dim, output_dim)
#######################
#  USE GPU FOR MODEL  #
#######################

if torch.cuda.is_available():
    model.cuda()

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Number of steps to unroll

loss_list = []
iter = 0

Data = Data_utility('./data/train_small', train=0.6, valid=0.2, normalize=1, y_label=1)

min_loss = float('inf')
for epoch in range(num_epochs):
    print('epoch:' + str(epoch+1))
    start_time = time.time()
    train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optimizer, batch_size)
    val_loss,acc,total = evaluate(Data, Data.valid[0], Data.valid[1], model, criterion, batch_size)
    if val_loss < min_loss:
        min_loss = val_loss
        torch.save(model.state_dict(), 'temp.pt')
    print('execution time(s): ' + str(int(time.time()-start_time)))
    print(train_loss,val_loss,acc,total)

model.load_state_dict(torch.load('temp.pt'))
test_loss,acc,total = evaluate(Data, Data.test[0], Data.test[1], model, criterion, batch_size)
print(test_loss,acc,total)

