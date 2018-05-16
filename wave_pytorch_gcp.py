import copy
import os
import time

import scipy.io
import torch

from functions import *

# set printoptions
torch.set_printoptions(linewidth=320, precision=8)
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

path = 'data/'


def runexample(H, model, str, lr=0.001):
    epochs = 200000
    validations = 5000
    printInterval = 1000
    # batch_size = 10000
    data = 'wavedata25ns.mat'

    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    os.makedirs(path + 'models', exist_ok=True)
    name = (data[:-4] + '%s%glr%s' % (H[:], lr, str)).replace(', ', '.').replace('[', '_').replace(']', '_')

    tica = time.time()
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Running %s on %s\n%s' %
          (name, device.type, torch.cuda.get_device_properties(0) if cuda else ''))

    if not os.path.isfile(path + data):
        import subprocess
        subprocess.call('wget -P data/ https://storage.googleapis.com/ultralytics/' + data, shell=True)
    mat = scipy.io.loadmat(path + data)
    x = mat['inputs']  # inputs (nx512) [waveform1 waveform2]
    y = mat['outputs'][:, 1:2]  # outputs (nx4) [position(mm), time(ns), PE, E(MeV)]
    nz, nx = x.shape
    ny = y.shape[1]

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)

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
            # scipy.io.savemat(path + name + '.mat', dict(bestepoch=best[0], loss=L[best[0]], L=L, name=name))
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
    # scipy.io.savemat(path + name + '.mat', dict(bestepoch=best[0], loss=loss, std=std, L=L, name=name))
    # files.download(path + name + '.mat')

    # data = []
    # for i, s in enumerate(labels):
    #   data.append(go.Scatter(x=np.arange(epochs), y=L[:, i], mode='markers+lines', name=s))
    # layout = go.Layout(xaxis=dict(type='linear', autorange=True),
    #                  yaxis=dict(type='log', autorange=True))
    # configure_plotly_browser_state()
    # iplot(go.Figure(data=data, layout=layout))

    return np.concatenate(([best[0]], np.array(loss), np.array(std.ravel())))


def tslr():
    # if __name__ == '__main__':
    H = [512, 108, 23, 5, 1]
    # tsv = ['Tanh', 'LogSigmoid', 'Softsign', 'ELU']
    tsv = np.logspace(-4, -2, 11)
    tsy = []

    for a in tsv:
        class LinearAct(torch.nn.Module):
            def __init__(self, nx, ny):
                super(LinearAct, self).__init__()
                self.Linear1 = torch.nn.Linear(nx, ny)
                self.act = eval('torch.nn.' + 'Tanh' + '()')

            def forward(self, x):
                return self.act(self.Linear1(x))

        class WAVE(torch.nn.Module):
            def __init__(self, n):  # n = [512, 108, 23, 5, 1]
                super(WAVE, self).__init__()
                self.fc0 = LinearAct(n[0], n[1])
                self.fc1 = LinearAct(n[1], n[2])
                self.fc2 = LinearAct(n[2], n[3])
                self.fc3 = torch.nn.Linear(n[3], n[4])

            def forward(self, x):
                return self.fc3(self.fc2(self.fc1(self.fc0(x))))

        y = runexample(H, model=WAVE(H), str=('.' + 'Tanh'), lr=a)
        tsy.append(y)
    tsy = np.array(tsy)
    scipy.io.savemat(path + 'TS.lr' + '.mat', dict(tsv=tsv, tsy=tsy))


if __name__ == '__main__':
    # def tsshape():
    class LinearAct(torch.nn.Module):
        def __init__(self, nx, ny):
            super(LinearAct, self).__init__()
            self.Linear1 = torch.nn.Linear(nx, ny)
            self.act = eval('torch.nn.' + 'Tanh' + '()')

        def forward(self, x):
            return self.act(self.Linear1(x))


    # H = [32] # 512 inputs, 2 outputs structures:
    # H = [81, 13]
    # H = [128, 32, 8]
    # H = [169, 56, 18, 6]

    # H = [23] # 512 inputs, 2 outputs structures:
    # H = [64, 8]
    # H = [108, 23, 5]
    # H = [147, 42, 12, 3]
    # H = [181, 64, 23, 8, 3]
    # H = [512, 108, 23, 5, 1]

    # tsv = ['Tanh', 'LogSigmoid', 'Softsign', 'ELU']
    # tsv = np.logspace(-4, -2, 11)
    tsv = [[512, 23, 1], [512, 64, 8, 1], [512, 108, 23, 5, 1], [512, 147, 42, 12, 3, 1], [512, 181, 64, 23, 8, 3, 1]]
    tsy = []

    H = tsv[0]
    class WAVE(torch.nn.Module):
        def __init__(self, n):  # n = [512, 108, 23, 5, 1]
            super(WAVE, self).__init__()
            self.fc0 = LinearAct(n[0], n[1])
            self.fc1 = torch.nn.Linear(n[1], n[2])
        def forward(self, x):
            return self.fc1(self.fc0(x))
    tsy.append(runexample(H, model=WAVE(H), str=('.' + 'Tanh')))

    H = tsv[1]
    class WAVE(torch.nn.Module):
        def __init__(self, n):  # n = [512, 108, 23, 5, 1]
            super(WAVE, self).__init__()
            self.fc0 = LinearAct(n[0], n[1])
            self.fc1 = LinearAct(n[1], n[2])
            self.fc2 = torch.nn.Linear(n[2], n[3])
        def forward(self, x):
            return self.fc2(self.fc1(self.fc0(x)))
    tsy.append(runexample(H, model=WAVE(H), str=('.' + 'Tanh')))

    H = tsv[2]
    class WAVE(torch.nn.Module):
        def __init__(self, n):  # n = [512, 108, 23, 5, 1]
            super(WAVE, self).__init__()
            self.fc0 = LinearAct(n[0], n[1])
            self.fc1 = LinearAct(n[1], n[2])
            self.fc2 = LinearAct(n[2], n[3])
            self.fc3 = torch.nn.Linear(n[3], n[4])
        def forward(self, x):
            return self.fc3(self.fc2(self.fc1(self.fc0(x))))
    tsy.append(runexample(H, model=WAVE(H), str=('.' + 'Tanh')))

    H = tsv[3]
    class WAVE(torch.nn.Module):
        def __init__(self, n):  # n = [512, 108, 23, 5, 1]
            super(WAVE, self).__init__()
            self.fc0 = LinearAct(n[0], n[1])
            self.fc1 = LinearAct(n[1], n[2])
            self.fc2 = LinearAct(n[2], n[3])
            self.fc3 = LinearAct(n[3], n[4])
            self.fc4 = torch.nn.Linear(n[4], n[5])
        def forward(self, x):
            return self.fc4(self.fc3(self.fc2(self.fc1(self.fc0(x)))))
    tsy.append(runexample(H, model=WAVE(H), str=('.' + 'Tanh')))

    H = tsv[4]
    class WAVE(torch.nn.Module):
        def __init__(self, n):  # n = [512, 108, 23, 5, 1]
            super(WAVE, self).__init__()
            self.fc0 = LinearAct(n[0], n[1])
            self.fc1 = LinearAct(n[1], n[2])
            self.fc2 = LinearAct(n[2], n[3])
            self.fc3 = LinearAct(n[3], n[4])
            self.fc4 = LinearAct(n[4], n[5])
            self.fc5 = torch.nn.Linear(n[5], n[6])
        def forward(self, x):
            return self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(self.fc0(x))))))
    tsy.append(runexample(H, model=WAVE(H), str=('.' + 'Tanh')))

    scipy.io.savemat(path + 'TS.shape' + '.mat', dict(tsv=tsv, tsy=np.array(tsy)))
