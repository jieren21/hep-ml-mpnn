""" Message Passing Neural Network for Event Classification. (Internal version)

Author: Jie Ren <renjie@itp.ac.cn>
Last modified: Nov 7, 2018

Dependences:
1. Python 3 (>=3.5)
2. numpy
3. h5py
4. PyTorch with CUDA

Please cite our paper arXiv:1807.09088 [hep-ph].

Disclaimer: this program is an internal version which comes without any guarantees.
"""

import os
import sys
import numpy as np
import h5py

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


N_VERTICE = 21
N_FEATURE = 30
BATCH_SIZE = 500


class DataLoader:
    def __init__(self, features, distances, masks, targets):
        self.features = torch.FloatTensor(features)
        self.distances = torch.FloatTensor(distances)
        self.masks = torch.FloatTensor(masks)
        self.targets = torch.FloatTensor(targets)

        self.n_event = len(self.targets)

        i = torch.randperm(self.n_event)
        self.features = self.features[i].cuda()
        self.distances = self.distances[i].cuda()
        self.masks = self.masks[i].cuda()
        self.targets = self.targets[i].cuda()

    def __len__(self):
        return self.n_event

    def __iter__(self):
        for i in range(0, self.n_event, BATCH_SIZE):
            features = self.features[i:i + BATCH_SIZE]
            distances = self.distances[i:i + BATCH_SIZE]
            masks = self.masks[i:i + BATCH_SIZE]
            targets = self.targets[i:i + BATCH_SIZE]
            yield features, distances, masks, targets


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fe = nn.Linear(7, N_FEATURE)
        self.fm1 = nn.Linear(N_FEATURE + 21, N_FEATURE)
        self.fu1 = nn.Linear(N_FEATURE * 2, N_FEATURE)
        self.fm2 = nn.Linear(N_FEATURE + 21, N_FEATURE)
        self.fu2 = nn.Linear(N_FEATURE * 2, N_FEATURE)
        #self.fm3 = nn.Linear(N_FEATURE + 21, N_FEATURE)
        #self.fu3 = nn.Linear(N_FEATURE * 2, N_FEATURE)
        self.fr = nn.Linear(N_FEATURE, 1)

        self.distance_nodes = torch.linspace(0, 5, 21).view(1, -1).cuda()

    def forward(self, x, d, mask):
        x = x.view(-1, 7)
        d = d.view(-1, 1)
        mask = mask.view(-1, N_VERTICE)
        N = mask.sum(1).view(-1, 1)

        d = torch.exp(-(d - self.distance_nodes)**2 / 2 / 0.25**2)

        # embedding
        h = F.relu(self.fe(x))

        # message passing
        hh = h.view(-1, N_VERTICE * N_FEATURE).repeat(1, N_VERTICE).view(-1, N_FEATURE)
        m = torch.cat([hh, d], dim=1)
        m = F.relu(self.fm1(m))
        m = m.view(-1, N_VERTICE**2, N_FEATURE) * mask.repeat(1, N_VERTICE).view(-1, N_VERTICE**2, 1)
        m = m.view(-1, N_VERTICE, N_FEATURE).sum(dim=1)
        m = torch.cat([h, m], dim=1)
        h = F.relu(self.fu1(m))

        # message passing
        hh = h.view(-1, N_VERTICE * N_FEATURE).repeat(1, N_VERTICE).view(-1, N_FEATURE)
        m = torch.cat([hh, d], dim=1)
        m = F.relu(self.fm2(m))
        m = m.view(-1, N_VERTICE**2, N_FEATURE) * mask.repeat(1, N_VERTICE).view(-1, N_VERTICE**2, 1)
        m = m.view(-1, N_VERTICE, N_FEATURE).sum(dim=1)
        m = torch.cat([h, m], dim=1)
        h = F.relu(self.fu2(m))

        # message passing
        #hh = h.view(-1, N_VERTICE * N_FEATURE).repeat(1, N_VERTICE).view(-1, N_FEATURE)
        #m = torch.cat([hh, d], dim=1)
        #m = F.relu(self.fm3(m))
        #m = m.view(-1, N_VERTICE**2, N_FEATURE) * mask.repeat(1, N_VERTICE).view(-1, N_VERTICE**2, 1)
        #m = m.view(-1, N_VERTICE, N_FEATURE).sum(dim=1)
        #m = torch.cat([h, m], dim=1)
        #h = F.relu(self.fu3(m))

        # readout
        y = F.sigmoid(self.fr(h))
        y = y.view(-1, N_VERTICE)
        y = y * mask.view(-1, N_VERTICE)
        y = y.sum(dim=1, keepdim=True) / N

        return y


# create data loader
print('create data loader')
ds_file = 'input.h5'
datasets = h5py.File(ds_file, 'r')
train_loader = DataLoader(datasets['Train/Feature'][:], datasets['Train/Distance'][:], datasets['Train/Mask'][:], datasets['Train/Target'][:])
valid_loader = DataLoader(datasets['Validation/Feature'][:], datasets['Validation/Distance'][:], datasets['Validation/Mask'][:], datasets['Validation/Target'][:])
del datasets

# create model
print('create model')
model = Model().cuda()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# logging
LOGS_DIR = 'logs'
os.makedirs(LOGS_DIR + '/loss')
os.makedirs(LOGS_DIR + '/model')
os.makedirs(LOGS_DIR + '/pred')
f_loss_epoch = open(LOGS_DIR + '/loss/loss_epoch.txt', 'a')

# main loop
print('start training')
for epoch in range(300):
    # train
    model.train()
    for step, (x, d, mask, y) in enumerate(train_loader):
        x, d, mask, y = x.cuda(), d.cuda(), mask.cuda(), y.cuda()
        y_ = model(Variable(x), Variable(d), Variable(mask))
        loss = criterion(y_, Variable(y))

        if step % 100 == 0:
            print('train', epoch, step, loss.data.item(), end='          \r')

        # optimize the model
        model.zero_grad()
        loss.backward()
        optimizer.step()

    # save model
    torch.save(model, LOGS_DIR + '/model/%d.model' % epoch)

    # calc training loss
    model.eval()
    train_loss = 0
    ys, ys_ = np.empty([0, 1]), np.empty([0, 1])
    for step, (x, d, mask, y) in enumerate(train_loader):
        x, d, mask, y = x.cuda(), d.cuda(), mask.cuda(), y.cuda()
        y_ = model(Variable(x), Variable(d), Variable(mask))
        loss = criterion(y_, Variable(y))

        if step % 100 == 0:
            print('train loss', epoch, step, loss.data.item(), end='          \r')

        train_loss += loss.data.item()
        ys = np.vstack([ys, y.cpu().numpy()])
        ys_ = np.vstack([ys_, y_.data.cpu().numpy()])
    train_loss /= len(train_loader) / BATCH_SIZE
    train_pred = np.hstack([ys, ys_])

    # calc validation loss
    model.eval()
    valid_loss = 0
    ys, ys_ = np.empty([0, 1]), np.empty([0, 1])
    for step, (x, d, mask, y) in enumerate(valid_loader):
        x, d, mask, y = x.cuda(), d.cuda(), mask.cuda(), y.cuda()
        y_ = model(Variable(x), Variable(d), Variable(mask))
        loss = criterion(y_, Variable(y))

        if step % 100 == 0:
            print('valid', epoch, step, loss.data.item(), end='          \r')

        valid_loss += loss.data.item()
        ys = np.vstack([ys, y.cpu().numpy()])
        ys_ = np.vstack([ys_, y_.data.cpu().numpy()])
    valid_loss /= len(valid_loader) / BATCH_SIZE
    valid_pred = np.hstack([ys, ys_])

    np.savez_compressed(LOGS_DIR + '/pred/%d.npz' % epoch, train=train_pred, valid=valid_pred)

    print(epoch, train_loss, valid_loss)
    f_loss_epoch.write('%d %g %g\n' % (epoch, train_loss, valid_loss))
    f_loss_epoch.flush()
