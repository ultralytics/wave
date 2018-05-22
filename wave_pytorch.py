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
torch.manual_seed(1)


def runexample(H, model, str, lr=0.001, amsgrad=False):
    epochs = 200000
    validations = 5000
    printInterval = 1000
    # batch_size = 10000
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
    labels = ['train', 'validate', 'test']

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

    model.train()
    ticb = time.time()
    L = np.full((epochs, 3), np.nan)
    best = (0, 1E6, model.state_dict())  # best (epoch, validation loss, model)
    for i in range(epochs):
        # for j, (xj, yj) in enumerate(train_loader):
        #    print(xj.shape,time.time() - tic)

        y_pred = model(x)
        y_predv = model(xv)

        # loss
        loss = criteria(y_pred, y)  # / y.numel()
        L[i, 0] = loss.item()  # / y.numel()  # train
        L[i, 1] = criteria(y_predv, yv).item()  # / yv.numel()  # validate
        # L[i, 2] = criteria(model(xt), yt).item() / yv.numel()  # test

        if i > validations:  # validation checks
            if L[i, 1] < best[1]:
                best = (i, L[i, 1], copy.deepcopy(model.state_dict()))
            if (i - best[0]) > validations:
                print('\n%g validation checks exceeded at epoch %g.' % (validations, i))
                break

        if i % printInterval == 0:
            # scipy.io.savemat(pathr + name + '.mat', dict(bestepoch=best[0], loss=L[best[0]], L=L, name=name))
            _, std = stdpt(y_predv - yv, ys)
            print('%.3fs' % (time.time() - ticb), i, L[i], std)
            ticb = time.time()

        # Zero gradients, perform a backward pass, update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        print('WARNING: Validation loss still decreasing after %g epochs (train longer).' % (i + 1))
    # torch.save(best[2], pathr + 'models/' + name + '.pt')
    model.load_state_dict(best[2])
    dt = time.time() - tica

    model.eval()
    print('\nFinished %g epochs in %.3fs (%.3f epochs/s)\nBest results from epoch %g:' % (i + 1, dt, i / dt, best[0]))
    loss, std = np.zeros(3), np.zeros((3, ny))
    for i, (xi, yi) in enumerate(((x, y), (xv, yv), (xt, yt))):
        loss[i], std[i] = stdpt(model(xi) - yi, ys)
        print('%.5f %s %s' % (loss[i], std[i, :], labels[i]))
    scipy.io.savemat(pathr + name + '.mat', dict(bestepoch=best[0], loss=loss, std=std, L=L, name=name))
    # files.download(pathr + name + '.mat')

    # data = []
    # for i, s in enumerate(labels):
    #   data.append(go.Scatter(x=np.arange(epochs), y=L[:, i], mode='markers+lines', name=s))
    # layout = go.Layout(xaxis=dict(type='linear', autorange=True),
    #                  yaxis=dict(type='log', autorange=True))
    # configure_plotly_browser_state()
    # iplot(go.Figure(data=data, layout=layout))

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
