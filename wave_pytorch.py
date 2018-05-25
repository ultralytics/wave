import copy
import os
import time

import scipy.io
import torch

from functions import *

# set printoptions
torch.set_printoptions(linewidth=320, precision=8)
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

pathd = 'data/'
pathr = 'results/'
labels = ['train', 'validate', 'test']
torch.manual_seed(1)


class patienceTerminator(object):
    def __init__(self, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):
        #super(patienceTerminator, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = float('inf')
        self.num_bad_epochs = None
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        # self._init_is_better(mode=mode, threshold=threshold,
        #                     threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = float('inf')
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = metrics.item()
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if current < self.best:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            print('\n%g patience exceeded at epoch %g.' % (self.patience, epoch))
            return True
        else:
            return False


def runexample(H, model, str, lr=0.005, amsgrad=False):
    epochs = 20
    validations = 2000
    printInterval = 1
    data = 'wavedata25ns.mat'

    cuda = torch.cuda.is_available()
    os.makedirs(pathr + 'models', exist_ok=True)
    name = (data[:-4] + '%s%glr%s' % (H[:], lr, str)).replace(', ', '.').replace('[', '_').replace(']', '_')

    tica = time.time()
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Running %s on %s\n%s' %
          (name, device.type, torch.cuda.get_device_properties(0) if cuda else ''))

    if not os.path.isfile(pathd + data):
        os.system('wget -P data/ https://storage.googleapis.com/ultralytics/' + data)
    mat = scipy.io.loadmat(pathd + data)
    x = mat['inputs']  # inputs (nx512) [waveform1 waveform2]
    y = mat['outputs'][:, 1:2]  # outputs (nx4) [position(mm), time(ns), PE, E(MeV)]
    nz, nx = x.shape
    ny = y.shape[1]

    x, _, _ = normalize(x, 1)  # normalize each input row
    y, ymu, ys = normalize(y, 0)  # normalize each output column
    x, y = torch.Tensor(x), torch.Tensor(y)
    x, y, xv, yv, xt, yt = splitdata(x, y, train=0.70, validate=0.15, test=0.15, shuffle=False)

    # train_dataset = torch.utils.data.TensorDataset(x, y)
    # test_dataset = torch.utils.data.TensorDataset(xt, yt)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print(model)
    if cuda:
        x, xv, xt = x.cuda(), xv.cuda(), xt.cuda()
        y, yv, yt = y.cuda(), yv.cuda(), yt.cuda()
        model = model.cuda()

    # criteria and optimizer
    criteria = torch.nn.MSELoss(size_average=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=amsgrad)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1000, factor=0.66, min_lr=1E-4, verbose=True)
    stopper = patienceTerminator(patience=1000)

    model.train()
    modelb = copy.deepcopy(model)
    modeltwice = copy.deepcopy(model)
    modelhalf = copy.deepcopy(model)

    ticb = time.time()
    L = np.full((epochs, 3), np.nan)
    best = (0, 1E16)  # best (epoch, validation loss)
    for i in range(epochs):
        # for j, (xj, yj) in enumerate(train_loader):
        #    print(xj.shape,time.time() - tic)

        if i > 0:
            pb = []
            for param in model.parameters():
                pb.append(copy.deepcopy(param.data))

            with torch.no_grad():
                for j, param in enumerate(modeltwice.parameters()):
                    param.data = pa[j] + (pb[j] - pa[j]) * 1.5
                for j, param in enumerate(modelhalf.parameters()):
                    param.data = pa[j] + (pb[j] - pa[j]) / 1.5

        loss = criteria(model(x), y).item()
        losstwice = criteria(modeltwice(x), y).item()
        losshalf = criteria(modelhalf(x), y).item()
        # print(loss, losstwice, losshalf)

        if (losstwice < loss) & (losstwice < losshalf):
            model.load_state_dict(modeltwice.state_dict())
        elif losshalf < loss:
            model.load_state_dict(modelhalf.state_dict())

        y_pred = model(x)
        y_predv = model(xv)

        # loss
        loss = criteria(y_pred, y)
        lossv = criteria(y_predv, yv)
        L[i, 0] = loss.item()  # train
        L[i, 1] = lossv.item()  # validate
        # scheduler.step(lossv)
        stopper.step(lossv)

        if i > validations:  # validation checks
            if L[i, 1] < best[1]:
                modelb.load_state_dict(model.state_dict())
                best = (i, L[i, 1])
            if (i - best[0]) > validations:
                print('\n%g validation checks exceeded at epoch %g.' % (validations, i))
                break

        if i % printInterval == 0:
            std = (y_predv - yv).std(0).detach().item() * ys
            print('%.3fs' % (time.time() - ticb), i, L[i, 0:2], std)
            ticb = time.time()

        # Zero gradients, perform a backward pass, update parameters
        pa = []
        for param in model.parameters():
            pa.append(copy.deepcopy(param.data))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        print('WARNING: Validation loss still decreasing after %g epochs (train longer).' % (i + 1))
    # torch.save(best[2], pathr + 'models/' + name + '.pt')
    dt = time.time() - tica
    epochs = i

    modelb.eval()
    print('\nFinished %g epochs in %.3fs (%.3f epochs/s)\nBest results from epoch %g:' % (i + 1, dt, i / dt, best[0]))
    loss, std = np.zeros(3), np.zeros((3, ny))
    for i, (xi, yi) in enumerate(((x, y), (xv, yv), (xt, yt))):
        loss[i], std[i] = stdpt(modelb(xi) - yi, ys)
        print('%.5f %s %s' % (loss[i], std[i, :], labels[i]))
    scipy.io.savemat(pathr + name + '.mat', dict(bestepoch=best[0], loss=loss, std=std, L=L, name=name))
    # files.download(pathr + name + '.mat')

    return np.concatenate(([best[0]], np.array(loss), np.array(std.ravel())))


H = [512, 64, 8, 1]


class LinearAct(torch.nn.Module):
    def __init__(self, nx, ny):
        super(LinearAct, self).__init__()
        self.Linear1 = torch.nn.Linear(nx, ny)
        self.act = eval('torch.nn.' + 'Tanh' + '()')

    def forward(self, x):
        return self.act(self.Linear1(x))


class WAVE(torch.nn.Module):
    def __init__(self, n):
        super(WAVE, self).__init__()
        self.fc0 = LinearAct(n[0], n[1])
        self.fc1 = LinearAct(n[1], n[2])
        self.fc2 = torch.nn.Linear(n[2], n[3])

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return self.fc2(x)


if __name__ == '__main__':
    _ = runexample(H, model=WAVE(H), str='.Tanh')
