#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import data.build_graph
import numpy as np
import torch
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os

num_epochs = 150
batch_size = 32
learning_rate = 1e-3


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 2, stride=1, padding=1),  # b, 8, 12, 12
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 8, 6, 6
            nn.Conv2d(8, 16, 2, stride=1, padding=1),  # b, 16, 7,7
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 8, 3, 3
            nn.Conv2d(16, 32, 2, stride=1, padding=1),  # b, 16, 4,4
            nn.ReLU(True),
            nn.MaxPool2d(4, stride=4)  # b, 8, 3, 3

        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32 ,16, 3, stride=1,padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 5, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
        x = self.decoder(x)
#         return x


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

# data
pic = data.build_graph.brightness()
pic_set = [[x[2]] for x in pic]
pic_set = np.array(pic_set)/255
data_len = len(pic_set)
pic_set = torch.FloatTensor(pic_set)

# training 
# for epoch in range(num_epochs):
#     for batch in range(data_len//batch_size):
#         img = pic_set[batch_size*batch:batch_size*(batch+1)]

#         img = Variable(img).cuda()
#         # ===================forward=====================
#         output = model(img)
#         loss = criterion(output, img)
#         # ===================backward====================
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     # ===================log========================
#     print('epoch [{}/{}], loss:{:.4f}'
#           .format(epoch+1, num_epochs, loss.data))
#     if epoch % 10 == 9:
#         pic = output.cpu().data
#         pic = torchvision.utils.make_grid(pic)
#         save_image(pic, './data/dc_img/imagenew_{}.png'.format(epoch))
#         save_image(img, './data/dc_img/imageold_{}.png'.format(epoch))

# torch.save(model.state_dict(), './conv_autoencoder.pth')


# # 输出图片

# In[2]:


from matplotlib import pyplot as plt
model.load_state_dict(torch.load('./conv_autoencoder.pth'))
model.eval()
cnt = 0
print(pic_set.shape)
result = np.array([])
Y = []
for batch in range(data_len//batch_size):
    img = pic_set[batch_size*batch:batch_size*(batch+1)]
    img = Variable(img).cuda()
    # ===================forward=====================
    output = model(img)
    if len(result) == 0:
        result = output.cpu().data
    else:
        result = np.concatenate((result,(output.cpu().data)),0)

    
# show pic
# plt.figure(figsize=(18,18))
# for i,pic in enumerate(result[:16]):
#     p = pic[0]
#     plt.subplot(8, 4, cnt + 1)
#     cnt += 1
#     plt.imshow(p, cmap='gray')
#     plt.subplot(8, 4, cnt + 1)
#     cnt += 1
#     plt.imshow(pic_set[i][0], cmap='gray')


# # 输出特征

# In[9]:


from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt

def company():
    model.load_state_dict(torch.load('./conv_autoencoder.pth'))
    model.eval()
    cnt = 0
    result = np.array([])
    Y = []
    for batch in range(data_len//batch_size + 1):
        img = pic_set[batch_size*batch:batch_size*(batch+1)]
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        if len(result) == 0:
            result = output.cpu().data
        else:
            result = np.concatenate((result,(output.cpu().data)),0)

    # show pic
    # plt.figure(figsize=(18,18))
    # for i,pic in enumerate(result[:16]):
    #     p = pic[0]
    #     plt.subplot(8, 4, cnt + 1)
    #     cnt += 1
    #     plt.imshow(p, cmap='gray')
    #     plt.subplot(8, 4, cnt + 1)
    #     cnt += 1
    #     plt.imshow(pic_set[i][0], cmap='gray')
    result = np.reshape(result,(-1,32))

    print(result.shape, np.array(pic).shape)
    company_feature = {}
    quarter = {'Q1': '-03-', 'Q2': '-06-', 'Q3': '-09-', 'Q4': '-12-'}
    for i, x in enumerate(pic):
        if x[0] not in company_feature:
            company_feature[x[0]] = []

        date_str = x[1][:4] + quarter[x[1][4:]] + '01'
        #         print(result[i],[date_str])
        company_feature[x[0]].append([date_str] + result[i].tolist())

    # print(company_feature)

    return company_feature

# company()


# # 画特征图

# In[ ]:


def draw(result):
    with open('data/stock-sector.txt','r') as f:
        lines = f.read()
    lines = lines.split('\n')
    lines = [x.split(',') for x in lines]
    stock_sector = {}
    sector = {}
    for x in lines:
        if len(x) >= 2:
            stock_sector[x[0]] = x[1]
            sector[x[1]] = []

    X = np.array(result)
    X_embedded = TSNE(n_components=2).fit_transform(X)

    X_embedded_sector = [[]*len(sector)]
    for i,x in enumerate(X_embedded):
        if pic[i][1] == '2018Q4':
            sector[stock_sector[pic[i][0]]].append(x)

    color_cnt = 20
    axis_x,axis_y,c = [], [],[]
    for key,value in sector.items():
        axis_x += [x[0] for x in value]
        axis_y += [x[1] for x in value]
        c += ([color_cnt]*len(value))    
        color_cnt += 20
    plt.scatter(axis_x,axis_y,c=c,cmap='gray')
    # print(X_embedded_sector)
    # axis_x, axis_y = [x[0] for x in X_embedded], [x[1] for x in X_embedded]
    # plt.scatter(axis_x,axis_y)
    plt.show()



