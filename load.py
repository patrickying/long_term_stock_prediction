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

model = torch.load('model.pt')

if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()
batch_size = 128

Data = Data_utility('./data/train_small', train=0.6, valid=0.2, normalize=1, y_label=1)
test_loss,acc,total = evaluate(Data, Data.test[0], Data.test[1], model, criterion, batch_size)
print(test_loss,acc,total)