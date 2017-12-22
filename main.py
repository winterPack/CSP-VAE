from __future__ import print_function
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from csp_tools import *
from model import CNN_VAE
import sys
import gc

learning_rate = 0.001
batch_size = 20
max_epoch = 1000

#load data
#X_train, y_train = load_data_range(range(19),range(40))
#X_val, y_val = load_data_range(range(19),range(40,45))
X_train = np.memmap('X_train.dat',dtype='float32',mode='r',shape=(760, 50, 50, 50, 1))
X_val = np.memmap('X_val.dat',dtype='float32',mode='r',shape=(95, 50, 50, 50, 1))


net = CNN_VAE()
optimizer = optim.Adam(net.parameters(), lr = learning_rate)

for epoch in range(max_epoch):
    n = X_train.shape[0]
    nBatch = n//batch_size
    idx = np.random.permutation(n)
    b_idx = np.array_split(idx,nBatch)
    for i, b in enumerate(b_idx):
        X_batch = X_train[b]
        X_batch = random_roll(X_batch)
        #If CSP > 2, it is a defect voxel
        X_batch = (X_batch>2).astype(np.float32)
        X_batch = Variable(torch.from_numpy(X_batch))
        X_batch = X_batch.transpose(1,4)
        optimizer.zero_grad()
        l1, l2 = net.calculate_loss(X_batch)
        loss = l1+l2
        loss.backward()
        optimizer.step()
        print('epoch {}, batch {}, l1 {:8.4f}, l2 {:8.4f}, loss {:8.4f}'.format(epoch, i, l1.data[0], l2.data[0], loss.data[0]))
        sys.stdout.flush()
    #check performance on validation set
    X_batch = Variable(torch.from_numpy(X_val.astype(np.float32)), volatile=True).transpose(1,4)
    l1, l2 = net.calculate_loss(X_batch)
    loss = l1+l2
    print('--validation l1 {:8.4f}, l2 {:8.4f}, loss {:8.4f}'.format(l1.data[0], l2.data[0], loss.data[0]))
    if epoch%10==0:
        torch.save(net.state_dict(),'net.epoch{:06d}.th'.format(epoch))



