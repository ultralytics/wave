import os
import time
import copy

import scipy.io
import torch
from plotly.offline import plot
import plotly.graph_objs as go

from functions import *

# set printoptions
torch.set_printoptions(linewidth=320, precision=8)
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5


def runexample(H, model, str):
    lr = 0.002
    eps = 0.001
    epochs = 300
    validations = 5000
    printInterval = 10
    # batch_size = 10000
    data = 'wavedata25ns.mat'

    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    path = 'data/'
    os.makedirs(path + 'models', exist_ok=True)
    device = torch.device('cuda:0' if cuda else 'cpu')

    tica = time.time()
    if not os.path.isfile(path + data):
        import subprocess
        subprocess.call('wget -P data/ https://storage.googleapis.com/ultralytics/' + data, shell=True)
    mat = scipy.io.loadmat(path + data)
    x = mat['inputs']  # inputs (nx512) [waveform1 waveform2]
    y = mat['outputs'][:, 0:2]  # outputs (nx4) [position(mm), time(ns), PE, E(MeV)]
    nz, nx = x.shape
    ny = y.shape[1]

    name = (data[:-4] + '%s%glr%geps%s' % (H[:], lr, eps, str)).replace(', ', '.').replace('[', '_').replace(']', '_')
    print('Running %s on %s\n%s' % (name, device.type, torch.cuda.get_device_properties(0) if cuda else ''))

    class LinearTanh(torch.nn.Module):
        def __init__(self, nx, ny):
            super(LinearTanh, self).__init__()
            self.Linear1 = torch.nn.Linear(nx, ny)
            self.Tanh = torch.nn.Tanh()

        def forward(self, x):
            return self.Tanh(self.Linear1(x))

    class WAVE(torch.nn.Module):
        def __init__(self, nx, H):  # 512 in, 512 out
            super(WAVE, self).__init__()
            # H = [76, 23, 7]
            self.fc0 = LinearTanh(nx, H[0])
            self.fc1 = LinearTanh(H[0], H[1])
            self.fc2 = LinearTanh(H[1], H[2])
            self.fc3 = torch.nn.Linear(H[2], 2)

        def forward(self, x):
            return self.fc3(self.fc2(self.fc1(self.fc0(x))))

    if model is None:
        model = WAVE(512, H)

    x, _, _ = normalize(x, 1)  # normalize each input row
    y, ymu, ys = normalize(y, 0)  # normalize each output column
    x, y = torch.Tensor(x), torch.Tensor(y)
    x, y, xv, yv, xt, yt = splitdata(x, y, train=0.70, validate=0.15, test=0.15, shuffle=False)
    labels = ['train', 'validate', 'test']

    # train_dataset = data_utils.TensorDataset(x, y)
    # test_dataset = data_utils.TensorDataset(xt, yt)
    # train_loader = data_utils.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # test_loader = data_utils.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print(model)
    if cuda:
        x, xv, xt = x.cuda(), xv.cuda(), xt.cuda()
        y, yv, yt = y.cuda(), yv.cuda(), yt.cuda()
        model = model.cuda()

    # criteria and optimizer
    criteria = torch.nn.MSELoss(size_average=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps, amsgrad=True)

    ticb = time.time()
    L = np.full((epochs, 3), np.nan)
    best = (0, 1E6, model.state_dict())  # best (epoch, validation loss, model)
    for i in range(epochs):
        # for j, (xj, yj) in enumerate(train_loader):
        #    print(xj.shape,time.time() - tic)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)
        y_predv = model(xv)

        # Compute and print loss
        loss = criteria(y_pred, y)
        L[i, 0] = loss.item()  # / y.numel()  # train
        L[i, 1] = criteria(y_predv, yv).item()  # / yv.numel()  # validate
        # L[i, 2] = criteria(model(xt), yt).item() #/ yv.numel()  # test

        if i > validations:  # validation checks
            if L[i, 1] < best[1]:
                best = (i, L[i, 1], copy.deepcopy(model.state_dict()))
            if (i - best[0]) > validations:
                print('\n%g validation checks exceeded at epoch %g.' % (validations, i))
                break

        if i % printInterval == 0:  # print and save progress
            scipy.io.savemat(path + name + '.mat', dict(bestepoch=best[0], loss=L[best[0]], L=L, name=name))
            _, std = stdpt(y_predv - yv, ys)
            print('%.3fs' % (time.time() - ticb), i, L[i], std)
            ticb = time.time()

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        print('WARNING: Validation loss still decreasing after %g epochs (train longer).' % (i + 1))
    torch.save(best[2], path + 'models/' + name + '.pt')
    model.load_state_dict(best[2])
    dt = time.time() - tica

    print('\nFinished %g epochs in %.3fs (%.3f epochs/s)\nBest results from epoch %g:' % (i + 1, dt, i / dt, best[0]))
    loss, std = np.zeros(3), np.zeros((3, ny))
    for i, (xi, yi) in enumerate(((x, y), (xv, yv), (xt, yt))):
        loss[i], std[i] = stdpt(model(xi) - yi, ys)
        print('%.5f %s %s' % (loss[i], std[i, :], labels[i]))
    scipy.io.savemat(path + name + '.mat', dict(bestepoch=best[0], loss=loss, std=std, L=L, name=name))
    # files.download(path + name + '.mat')

    data = []
    for i, s in enumerate(labels):
        data.append(go.Scatter(x=np.arange(epochs), y=L[:, i], mode='markers+lines', name=s))
    layout = go.Layout(xaxis=dict(type='linear', autorange=True),
                       yaxis=dict(type='log', autorange=True))
    # configure_plotly_browser_state()
    plot(go.Figure(data=data, layout=layout))


class waveconv(torch.nn.Module):
    def __init__(self):
        super(waveconv, self).__init__()
        self.AP = torch.nn.AvgPool1d(4)
        self.MP = torch.nn.MaxPool1d(4)
        self.C1 = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=15, stride=1, groups=1)
        self.C2 = torch.nn.Conv1d(in_channels=3, out_channels=8, kernel_size=5, stride=1, groups=1)

    def forward(self, x):
        r, c = x.size()
        x = x.view((r, 1, c))
        y = self.C1(x)
        # y = self.MP(y)
        # y = self.C2(y)
        # y = self.AP(y)
        return y.view((r, y.numel() / r))


if __name__ == '__main__':
    # model = torch.nn.Sequential(
    #        waveconv(),
    #        torch.nn.Linear(498, H[0]), torch.nn.Tanh(),
    #        torch.nn.Linear(H[0], H[1]), torch.nn.Tanh(),
    #        torch.nn.Linear(H[1], H[2]), torch.nn.Tanh(),
    #        torch.nn.Linear(H[2], 2))
    H = [128, 32, 8]
    for i in range(1):
        runexample(H, None, '.' + str(i))
