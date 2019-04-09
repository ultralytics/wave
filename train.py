import argparse
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


def train(H, model, str, lr=0.001):
    data = 'wavedata25ns.mat'

    cuda = torch.cuda.is_available()
    os.makedirs(pathr + 'models', exist_ok=True)
    name = (data[:-4] + '%s%glr%s' % (H[:], lr, str)).replace(', ', '.').replace('[', '_').replace(']', '_')
    print('Running ' + name)

    device = select_device()

    if not os.path.isfile(pathd + data):
        os.system('wget -P data/ https://storage.googleapis.com/ultralytics/' + data)
    mat = scipy.io.loadmat(pathd + data)
    x = mat['inputs'][:]  # inputs (nx512) [waveform1 waveform2]
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

    # Loss criteria
    MSE = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9)

    # Scheduler
    stopper = patienceStopper(epochs=opt.epochs, patience=30, printerval=opt.printerval)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.1, min_lr=1E-5,
                                                           verbose=True)

    lossv = 1E6
    bs = opt.batch_size
    nb = int(np.ceil(x.shape[0] / bs))
    L = np.full((opt.epochs, 3), np.nan)
    model_info(model)
    for i in range(opt.epochs):
        scheduler.step(lossv)

        # Train
        model.train()
        for bi in range(nb):
            j = range(bi * bs, min((bi + 1) * bs, x.shape[0]))
            if ONNX_EXPORT:
                _ = torch.onnx._export(model, x, 'model.onnx', verbose=True)
                return

            loss = MSE(model(x[j]), y[j])
            L[i, 0] = loss.item()  # train

            # Zero gradients, backward pass, update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test
        model.eval()
        with torch.no_grad():
            yv_ = model(xv)
            lossv = MSE(yv_, yv)
            L[i, 1] = lossv.item()  # validate

        if i % opt.printerval == 0:
            std = (yv_ - yv).std(0).detach().cpu().numpy() * ys

        if stopper.step(lossv, model=None, metrics=std):
            break

    # Print and save final results
    # torch.save(stopper.bestmodel.state_dict(), pathr + 'models/' + name + '.pt')
    stopper.bestmodel.eval()
    loss, std = np.zeros(3), np.zeros((3, ny))
    for i, (xi, yi) in enumerate(((x, y), (xv, yv), (xt, yt))):
        with torch.no_grad():
            r = stopper.bestmodel(xi) - yi  # residuals, ().detach?
            loss[i] = (r ** 2).mean().cpu().item()
            std[i] = r.std(0).cpu().numpy() * ys
        print('%.5f %s %s' % (loss[i], std[i, :], labels[i]))

    scipy.io.savemat(pathr + name + '.mat', dict(bestepoch=stopper.bestloss, loss=loss, std=std, L=L, name=name))
    # files.download(pathr + name + '.mat')

    return np.concatenate(([stopper.bestloss], np.array(loss), np.array(std.ravel())))


#       400  5.1498e-05    0.023752      12.484     0.15728  # var 0
class WAVE(torch.nn.Module):
    def __init__(self, n=(512, 64, 8, 2)):
        super(WAVE, self).__init__()
        self.fc0 = nn.Linear(n[0], n[1])
        self.fc1 = nn.Linear(n[1], n[2])
        self.fc2 = nn.Linear(n[2], n[3])

    def forward(self, x):  # x.shape = [bs, 512]
        x = torch.tanh(self.fc0(x))  # [bs, 64]
        x = torch.tanh(self.fc1(x))  # [bs, 8]
        return self.fc2(x)  # [bs, 2]


# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/02-intermediate
#       121  2.6941e-05    0.021642      11.923     0.14201  # var 1
class WAVE4(nn.Module):
    def __init__(self, n_out=2):
        super(WAVE4, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 9), stride=(1, 2), padding=(0, 4), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1))
        # nn.MaxPool2d(kernel_size=(1, 2), stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 9), stride=(1, 2), padding=(0, 4), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1))
        # nn.MaxPool2d(kernel_size=(1, 2), stride=1))
        self.layer3 = nn.Conv2d(64, n_out, kernel_size=(2, 64), stride=(1, 1), padding=(0, 0))

    def forward(self, x):  # x.shape = [bs, 512]
        x = x.view((-1, 2, 256))  # [bs, 2, 256]
        x = x.unsqueeze(1)  # [bs, 1, 2, 256] =  = [N, C, H, W]
        x = self.layer1(x)  # [bs, 32, 1, 128]
        x = self.layer2(x)  # [bs, 64, 1, 64]
        x = self.layer3(x)
        return x.reshape(x.size(0), -1)  # [bs, 64*64]


#       121  2.6941e-05    0.021642      11.923     0.14201  # var 1
class WAVE3(nn.Module):
    def __init__(self, n_out=2):
        super(WAVE3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1))
        self.layer4 = nn.Conv2d(128, n_out, kernel_size=(1, 32), stride=1, padding=0)

    def forward(self, x):  # x.shape = [bs, 512]
        x = x.view((-1, 2, 256))  # [bs, 2, 256]
        x = x.unsqueeze(2)  # [bs, 2, 1, 256] = [N, C, H, W]

        x = self.layer1(x)  # [bs, 32, 1, 128]
        # print(x.shape)
        x = self.layer2(x)  # [bs, 64, 1, 64]
        # print(x.shape)
        x = self.layer3(x)  # [bs, 128, 1, 32]
        # print(x.shape)

        x = self.layer4(x)
        return x.reshape(x.size(0), -1)  # [bs, 64*64]


class WAVE2(nn.Module):
    def __init__(self, n_out=2):
        super(WAVE2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(2, 30), stride=(1, 2), padding=(1, 15), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(1, 2), stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(2, 30), stride=(1, 2), padding=(0, 15), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(1, 2), stride=1))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, n_out, kernel_size=(2, 64), stride=(1, 1), padding=(0, 0)))

    def forward(self, x):  # x.shape = [bs, 512]
        x = x.view((-1, 2, 256))  # [bs, 2, 256]
        x = x.unsqueeze(1)  # [bs, 1, 2, 256]
        x = self.layer1(x)  # [bs, 32, 1, 128]
        x = self.layer2(x)  # [bs, 64, 1, 64]
        x = self.layer3(x)
        return x.reshape(x.size(0), -1)  # [bs, 64*64]


H = [512, 64, 8, 2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=5000, help='size of each image batch')
    parser.add_argument('--printerval', type=int, default=1, help='print results interval')
    parser.add_argument('--var', nargs='+', default=[0], help='debug list')
    opt = parser.parse_args()
    opt.var = [float(x) for x in opt.var]
    print(opt, end='\n\n')

    init_seeds()

    if opt.var[0] == 0:
        _ = train(H, model=WAVE(), str='.Tanh')
    elif opt.var[0] == 2:
        _ = train(H, model=WAVE2(), str='.Tanh')
    elif opt.var[0] == 3:
        _ = train(H, model=WAVE3(), str='.Tanh')
    elif opt.var[0] == 4:
        _ = train(H, model=WAVE4(), str='.Tanh')

# 100K SET ---------------------------------------------------------------------
# Model Summary: 8 layers, 33376 parameters, 33376 gradients
#        epoch        time        loss   metric(s)
#            0     0.23533     0.72525      57.944      4.6891
#         1000      6.1377    0.027707      13.409     0.23723
#         2000      6.1811    0.025165       12.82     0.19568
#         3000      6.1135    0.024321      12.614     0.18148
#         4000      6.1703    0.023974      12.528     0.17578
#         5000      6.0297    0.023792       12.48     0.17282
#         6000       6.044    0.023641      12.443     0.17017
#         7000       6.022    0.025316       12.86     0.16977
#         8000      6.0789    0.023559      12.424     0.16832
#         9000      6.0554    0.023912      12.464     0.16599
#        10000      6.0805     0.02347      12.403     0.16509
#        11000      6.1321    0.024346      12.579     0.16366
#        12000      6.0378    0.025261      12.618     0.16218
#        13000       6.003    0.023413      12.391     0.16071
#        14000      6.0259    0.023771       12.46     0.15963
#        15000      6.0809    0.023371      12.382       0.158
#        16000      6.0842     0.02339      12.389     0.15699
# 3000 Patience exceeded at epoch 16857.
# Finished 50000 epochs in 102.663s (487.032 epochs/s). Best results:
#        13856      5.1492    0.023221      12.391     0.16071
# 0.01641 [     10.358     0.15294] train
# 0.02322 [      12.34     0.15902] validate
# 0.02316 [     12.328     0.15611] test

# BS 2K
# 100 Patience exceeded at epoch 510.
# Finished 1000 epochs in 27.223s (36.733 epochs/s). Best results:
#          409  5.7936e-05     0.02456       12.69     0.15899
# 0.01756 [     10.706     0.15338] train
# 0.02456 [      12.69     0.15899] validate
# 0.02457 [     12.687     0.15632] test

#       400  5.1498e-05    0.023752      12.484     0.15728  # var 0
#       121  2.6941e-05    0.021642      11.923     0.14201  # var 1


# 10K TEST SET
# 3000 Patience exceeded at epoch 4162.
# Finished 50000 epochs in 8.108s (6166.670 epochs/s). Best results:
#         1161   0.0035503    0.035007      15.125     0.25265
# 0.01647 [     10.276      0.2399] train
# 0.03501 [     15.104     0.25241] validate
# 0.04057 [     16.274     0.26408] test


# BASELINE TRAIN ON FIRST 10K
# 100 Patience exceeded at epoch 301.
# Finished 1000 epochs in 279.341s (3.580 epochs/s). Best results:
#          200  2.5511e-05    0.027798      13.435     0.21111
# 0.01846 [     10.901     0.17024] train
# 0.02752 [      13.41     0.18784] validate
# 0.03360 [     14.818     0.19295] test
