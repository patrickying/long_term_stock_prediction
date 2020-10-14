import torch
import numpy as np
from torch.autograd import Variable
import os
from scipy import stats
seed = 12345
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file, train, valid, normalize=2, y_label=0):
        self.rawdat = np.array([]) # stock,time,dim
        stock_count = 0
        for root, dirs, files in os.walk(file):
            for f in files:
                fin = open(root + '/' + f)
                n = np.loadtxt(fin, delimiter=',')
                stock_count += 1
                self.rawdat = np.append(self.rawdat, n)
                dim = n.shape
                self.rawdat = self.rawdat.reshape(stock_count, dim[0], dim[1])
        self.P = 12  # RNN长度
        self.h = 1  # 预测未来h期
        self.cuda = True
        self.dat = np.zeros(self.rawdat.shape)  # 标准化后的数据
        self.s, self.n, self.m = self.dat.shape
        self.y_label = y_label
        self.normalize = normalize
        self.scale = np.ones(self.m)
        self.scale_add = np.zeros(self.m)
        self.train_percentage = train
        self._normalized(normalize)
        
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)
        # classification threshold
        self.bottom = 0.0
        self.top = 0.0

    def _normalized(self, normalize):
        self.train_norm = self.rawdat[:, :int(self.train_percentage * self.n), :] # 用训练集来做标准化
        if (normalize == 0): # no normalize
            self.dat = self.rawdat
        #  normlized by the MinMaxScalar
        if (normalize == 1):
            for i in range(self.m ):
                # self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                ma = np.max(self.train_norm[:, :, i])
                mi = np.min(self.train_norm[:, :, i])
                if (ma - mi) != 0:
                    self.dat[:, :, i] = (self.rawdat[:, :, i] - mi) / (ma-mi)
                    self.scale[i] = ma - mi
                    self.scale_add[i] = mi
                else:
                    self.dat[:, :, i] = self.rawdat[:, :, i]
                # self.dat[:, i] = stats.zscore(self.rawdat[:, i])

        # normlized by the z-score.
        if (normalize == 2): 
            for i in range(self.m):
#                 self.scale[i] = np.max(np.abs(self.rawdat[:, i]));
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]));

        temp = sorted(list(self.dat[:,:,-1].flatten()))
        self.bottom, self.top = temp[len(temp) // 3], temp[len(temp) * 2 // 3]

    def _split(self, train, valid, test):
        train_set = range(self.P, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n*self.s, self.P, self.m-self.y_label))  # 资料数量,RNN长度,特征数
        Y = torch.zeros((n*self.s, max(1, self.y_label)))
        
        # threshold for classifcation problem
        for s in range(self.s):  # each stock
            for i in range(n):  # each time
                end = idx_set[i] - self.h + 1
                start = end - self.P

                if self.y_label:

                    X[i+s*n, :, :] = torch.from_numpy(self.dat[s, start:end, :-self.y_label])
                    #Y[i+c*n, :] = torch.from_numpy(self.dat[end, -self.y_label:])  # regression

                    # classification
                    class_value = np.where(self.dat[s,end-1, -self.y_label:]>=self.top,2,np.where(self.dat[s,end-1, -self.y_label:]<=self.bottom,0,1))
    #                     class_onehot = np.zeros([class_value.size, 3])
    #                     class_onehot[np.arange(class_value.size), class_value.reshape(1, class_value.size)]  = 1
    #                     print(class_value,Y.shape)
                    Y[i+s*n, :] = torch.from_numpy(class_value)
                else:
                    X[i+s*n, :, :] = torch.from_numpy(self.dat[s,start:end, :])
                    Y[i+s*n, :] = torch.from_numpy(self.dat[s,idx_set[i], 0:1])
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            if (self.cuda):
                X = X.cuda()
                Y = Y.cuda()
            yield Variable(X), Variable(Y)
            start_idx += batch_size
if __name__ == '__main__':
    Data = Data_utility('./data/train', train=0.6, valid=0.2, normalize=1, y_label=1)

    print(Data.train[0].shape,Data.train[1].shape)
    dic = {0:0,1:0,2:0}
    for x in Data.train[1]:
        dic[int(x[0])] += 1
    print(dic)

    dic = {0: 0, 1: 0, 2: 0}
    for x in Data.valid[1]:
        dic[int(x[0])] += 1
    print(dic)

    dic = {0: 0, 1: 0, 2: 0}
    for x in Data.test[1]:
        dic[int(x[0])] += 1
    print(dic)

