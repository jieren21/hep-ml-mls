""" Eggbox benchmark for Machine Learning Scan.

Author: Jie Ren <renjie@itp.ac.cn>
Last modified: Jun 22, 2018

Dependences:
1. Python 3 (>=3.5)
2. Numpy
3. PyTorch with CUDA

Please cite our paper arXiv:1708.06615 [hep-ph].

Disclaimer: this program is an internal test version which comes without any guarantees.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


N_DIM = 2


def parameters(n):
    return np.random.rand(n, N_DIM) * 10 * np.pi


def observables(x):
    return (2 + np.cos(x[:, 0, np.newaxis] / 2) * np.cos(x[:, 1, np.newaxis] / 2))**5


def log_likelihood(y):
    logL = -(y - 100)**2 / (2 * 10**2)
    return logL.reshape(-1)


model = nn.Sequential(nn.Linear(N_DIM, 100), nn.ReLU(),
                      nn.Linear(100, 100), nn.ReLU(),
                      nn.Linear(100, 100), nn.ReLU(),
                      nn.Linear(100, 100), nn.ReLU(),
                      nn.Linear(100, 1)).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.MSELoss()

Tx = parameters(200)
Ty = observables(Tx)
Tl = log_likelihood(Ty)

for iteration in range(100):
    print('Iteration', iteration)

    # train
    for epoch in range(2000):
        y = model(Variable(torch.FloatTensor(Tx).cuda()))
        loss = criterion(y, Variable(torch.FloatTensor(Ty).cuda()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(epoch, loss.data.item())

    # sample
    Dx = np.zeros([0, N_DIM])
    while len(Dx) < 90:
        x = parameters(10000)
        y = model(Variable(torch.FloatTensor(x).cuda())).data.cpu().numpy()
        logL = log_likelihood(y)
        x = x[np.log(np.random.rand(len(x))) < logL - np.max(logL)]
        if len(x) > 0:
            Dx = np.concatenate([Dx, x])
    Dx = np.concatenate([Dx[:90], np.random.rand(10, N_DIM) * 10 * np.pi])
    Dy = observables(Dx)
    Dl = log_likelihood(Dy)

    plt.clf()
    cmap = LinearSegmentedColormap.from_list('mycmap', [[0.8, 0.8, 1], [0, 0, 1]])
    plt.scatter(Tx[:, 0], Tx[:, 1], s=1, c=np.exp(Tl), cmap=cmap, vmin=-0.5, vmax=1)
    plt.colorbar()
    plt.scatter(Dx[:, 0], Dx[:, 1], s=5, c='r')
    plt.xlim(0, 10 * np.pi)
    plt.ylim(0, 10 * np.pi)
    plt.title(iteration)
    plt.pause(1e-3)

    Tx = np.concatenate([Tx, Dx])
    Ty = np.concatenate([Ty, Dy])
    Tl = np.concatenate([Tl, Dl])

    torch.save(model, 'model_%d.torch' % iteration)
    np.save('data_%d.npy' % iteration, np.hstack([Tx, Ty, Tl.reshape(-1, 1)]))

plt.show()
