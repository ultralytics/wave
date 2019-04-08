import copy
import os
import time

import scipy.io
import torch.nn as nn

from utils.torch_utils import *
from utils.utils import *

ONNX_EXPORT = False


class patienceStopper(object):
    def __init__(self, patience=10, verbose=True, epochs=1000, printerval=10):
        self.patience = patience
        self.verbose = verbose
        self.bestepoch = 0
        self.bestmodel = None
        self.epoch = -1
        self.epochs = epochs - 1  # max epochs
        self.reset()
        self.t0 = time.time()
        self.t = self.t0
        self.printerval = printerval

    def reset(self):
        self.bestloss = float('inf')
        self.bestmetrics = None
        self.num_bad_epochs = 0

    def step(self, loss, metrics=None, model=None):
        loss = loss.item()
        self.num_bad_epochs += 1
        self.epoch += 1
        self.first(model) if self.epoch == 0 else None
        self.printepoch(self.epoch, loss, metrics) if self.epoch % self.printerval == 0 else None

        if loss < self.bestloss:
            self.bestloss = loss
            self.bestmetrics = metrics
            self.bestepoch = self.epoch
            self.num_bad_epochs = 0
            if model:
                if self.bestmodel:
                    self.bestmodel.load_state_dict(model.state_dict())  # faster than deepcopy
                else:
                    self.bestmodel = copy.deepcopy(model)

        if self.num_bad_epochs > self.patience:
            self.final('%g Patience exceeded at epoch %g.' % (self.patience, self.epoch))
            return True
        elif self.epoch >= self.epochs:
            self.final('WARNING: %g Patience not exceeded by epoch %g (train longer).' % (self.patience, self.epoch))
            return True
        else:
            return False

    def first(self, model):
        if model:
            model_info(model)
        s = ('epoch', 'time', 'loss', 'metric(s)')
        print('%12s' * len(s) % s)

    def printepoch(self, epoch, loss, metrics):
        s = (epoch, time.time() - self.t, loss)
        if metrics is not None:
            for i in range(len(metrics)):
                s += (metrics[i],)
        print('%12.5g' * len(s) % s)
        self.t = time.time()

    def final(self, msg):
        dt = time.time() - self.t0
        print('%s\nFinished %g epochs in %.3fs (%.3f epochs/s). Best results:' % (
            msg, self.epochs + 1, dt, (self.epochs + 1) / dt))
        self.printepoch(self.bestepoch, self.bestloss, self.bestmetrics)


pathd = 'data/'
pathr = 'results/'
labels = ['train', 'validate', 'test']
torch.manual_seed(1)


def runexample(H, model, str, lr=0.001):
    epochs = 50000
    patience = 3000
    printerval = 1000
    data = 'wavedata25ns.mat'

    cuda = torch.cuda.is_available()
    os.makedirs(pathr + 'models', exist_ok=True)
    name = (data[:-4] + '%s%glr%s' % (H[:], lr, str)).replace(', ', '.').replace('[', '_').replace(']', '_')
    print('Running ' + name)

    device = select_device()

    if not os.path.isfile(pathd + data):
        os.system('wget -P data/ https://storage.googleapis.com/ultralytics/' + data)
    mat = scipy.io.loadmat(pathd + data)
    x = mat['inputs']  # inputs (nx512) [waveform1 waveform2]
    y = mat['outputs'][:, 0:2]  # outputs (nx4) [position(mm), time(ns), PE, E(MeV)]
    nz, nx = x.shape
    ny = y.shape[1]

    x, _, _ = normalize(x, 1)  # normalize each input row
    y, ymu, ys = normalize(y, 0)  # normalize each output column
    x, y = torch.Tensor(x), torch.Tensor(y)
    x, y, xv, yv, xt, yt = splitdata(x, y, train=0.70, validate=0.15, test=0.15, shuffle=False)

    # torch.nn.init.constant_(model.out.weight.data, ys.item(0))
    # torch.nn.init.constant_(model.out.bias.data, ymu.item(0))
    # ys = 1

    if cuda:
        x, xv, xt = x.to(device), xv.to(device), xt.to(device)
        y, yv, yt = y.to(device), yv.to(device), yt.to(device)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)

    # criteria and optimizer
    criteria = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9, weight_decay=5e-4)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1000, factor=0.66, min_lr=1E-4, verbose=True)
    stopper = patienceStopper(epochs=epochs, patience=patience, printerval=printerval)

    model.train()
    L = np.full((epochs, 3), np.nan)
    for i in range(epochs):
        # scheduler.step(lossv)

        if ONNX_EXPORT:
            y_ = torch.onnx._export(model, x, 'model.onnx', verbose=True);
            return

        # train
        y_ = model(x)
        loss = criteria(y_, y)
        L[i, 0] = loss.item()  # train

        # Zero gradients, backward pass, update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # test
        yv_ = model(xv)
        lossv = criteria(yv_, yv)
        L[i, 1] = lossv.item()  # validate

        if i % printerval == 0:
            std = (yv_ - yv).std(0).detach().cpu().numpy()  # * ys

        if stopper.step(lossv, model=model, metrics=std):
            break

    # torch.save(stopper.bestmodel.state_dict(), pathr + 'models/' + name + '.pt')

    stopper.bestmodel.eval()
    loss, std = np.zeros(3), np.zeros((3, ny))
    for i, (xi, yi) in enumerate(((x, y), (xv, yv), (xt, yt))):
        loss[i], std[i] = stdpt(stopper.bestmodel(xi) - yi, ys)
        print('%.5f %s %s' % (loss[i], std[i, :], labels[i]))
    scipy.io.savemat(pathr + name + '.mat', dict(bestepoch=stopper.bestloss, loss=loss, std=std, L=L, name=name))
    # files.download(pathr + name + '.mat')

    return np.concatenate(([stopper.bestloss], np.array(loss), np.array(std.ravel())))


H = [512, 64, 8, 2]


class WAVE(torch.nn.Module):
    def __init__(self, n):
        super(WAVE, self).__init__()
        self.fc0 = nn.Linear(n[0], n[1])
        self.fc1 = nn.Linear(n[1], n[2])
        self.fc2 = nn.Linear(n[2], n[3])
        self.out = nn.Linear(n[3], n[3])
        # for param in self.out.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x = torch.tanh(self.fc0(x))
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x  # self.out(x)


if __name__ == '__main__':
    _ = runexample(H, model=WAVE(H), str='.Tanh')

# 8 layers, 33376 parameters, 33370 gradients
#        epoch        time        loss   metric(s)
#            0    0.071015     0.96796      57.704      6.7323
#            1     0.14455     0.72525      57.944      4.6891
#            2     0.16589     0.60906      57.893      3.2648
#            3     0.14249     0.55574      57.307      2.5013
#            4     0.14867     0.52581       56.42      2.1762
#            5     0.14721     0.50423       55.48      2.0354
#            6     0.15886     0.48679      54.633      1.9413
#            7      0.1579     0.47237      53.931      1.8527
#            8      0.1649     0.46044      53.353      1.7714
#            9     0.13345     0.45037      52.845      1.7092
# WARNING: 3000 Patience not exceeded by epoch 9 (train longer).
# Finished 10 epochs in 1.435s (6.967 epochs/s). Best results:
#            9   0.0001719     0.45037      52.845      1.7092
# 0.45271 [     52.953      1.7229] train
# 0.45037 [     52.845      1.7092] validate
# 0.45150 [     52.891      1.7008] test
