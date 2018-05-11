import copy
import time

import numpy as np
import scipy.io
import torch

torch.set_printoptions(linewidth=320, precision=8)
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

path = '/Users/glennjocher/Google Drive/data/'


def normalize(x, axis=None):
    # normalize NN inputs and outputs by column (axis=0)
    if axis is None:
        mu, sigma = x.mean(), x.std()
    elif axis == 0:
        mu, sigma = x.mean(0), x.std(0)
    elif axis == 1:
        mu, sigma = x.mean(1).reshape(x.shape[0], 1), x.std(1).reshape(x.shape[0], 1)
    return (x - mu) / sigma, mu, sigma


def shuffledata(x, y):
    i = np.arange(x.shape[0])
    np.random.shuffle(i)
    return x[i], y[i]


def splitdata(x, y, train=0.7, validate=0.15, test=0.15, shuffle=False):
    n = x.shape[0]
    if shuffle:
        x, y = shuffledata(x, y)
    i = round(n * train)
    j = round(n * validate) + i
    k = round(n * test) + j
    return x[:i], y[:i], x[i:j], y[i:j], x[j:k], y[j:k]  # xy train, xy validate, xy test


def nnstd(r, ys):
    r = r.detach()
    loss = (r ** 2).mean().cpu().numpy()
    std = r.std(0).cpu().numpy() * ys
    return loss, std


def runexample(H, model):
    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    # !mkdir - p
    # drive / data / models

    lr = 0.002
    eps = 0.001
    batch_size = 10000
    epochs = 500000
    validation_checks = 5000
    name = 'nn%s%glr%geps25nsconv' % (H[:], lr, eps)

    tica = time.time()
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Running on %s\n%s' % (device.type, torch.cuda.get_device_properties(0) if cuda else ''))

    # if not os.path.isfile(path + 'wavedata25ns.mat'):
    # !wget -P drive/data/ https://storage.googleapis.com/ultralytics/wavedata25ns.mat
    mat = scipy.io.loadmat(path + 'wavedata25ns.mat')
    x = mat['inputs']  # network inputs (nx512)
    y = mat['outputs']  # network outputs (nx2)
    nb, D_in = x.shape
    D_out = y.shape[1]

    class LinearTanh(torch.nn.Module):
        def __init__(self, D_in, D_out):
            super(LinearTanh, self).__init__()
            self.linear1 = torch.nn.Linear(D_in, D_out)
            self.Tanh = torch.nn.Tanh()

        def forward(self, x):
            return self.Tanh(self.linear1(x))

    class coarseTime(torch.nn.Module):
        def __init__(self, D_in):  # 512 in, 512 out
            super(coarseTime, self).__init__()
            H = [76, 23, 7]
            self.l0 = LinearTanh(D_in, H[0])
            self.l1 = LinearTanh(H[0], H[1])
            self.l2 = LinearTanh(H[1], H[2])
            self.l3 = torch.nn.Linear(H[2], 1)
            self.p0 = LinearTanh(D_in, H[0])
            self.p1 = LinearTanh(H[0], H[1])
            self.p2 = LinearTanh(H[1], H[2])
            self.p3 = torch.nn.Linear(H[2], 1)

        def forward(self, x):
            p = self.p3(self.p2(self.p1(self.p0(x))))
            t = self.l3(self.l2(self.l1(self.l0(x))))
            return torch.cat((p, t), 1)

    if model is None:
        model = coarseTime(512)

    x, _, _ = normalize(x, 1)  # normalize each input row
    y, ymu, ys = normalize(y, 0)  # normalize each output column
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    x, y, xv, yv, xt, yt = splitdata(x, y, train=0.70, validate=0.15, test=0.15, shuffle=False)
    labels = ['train', 'validate', 'test']

    # SubsetRandomSampler
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

        if i > 2000:  # validation checks
            if L[i, 1] < best[1]:
                best = (i, L[i, 1], copy.deepcopy(model.state_dict()))
            if (i - best[0]) > validation_checks:
                print('\n%g validation checks exceeded at epoch %g.' % (validation_checks, i))
                break

        if i % 10 == 0:  # print and save progress
            # scipy.io.savemat(path + name + '.mat', dict(bestepoch=best[0], loss=L[best[0]], L=L, name=name))
            _, std = nnstd(y_predv - yv, ys)
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
    loss, std = np.zeros(3), np.zeros((3, D_out))
    for i, (xi, yi) in enumerate(((x, y), (xv, yv), (xt, yt))):
        loss[i], std[i] = nnstd(model(xi) - yi, ys)
        print('%.5f %s %s' % (loss[i], std[i, :], labels[i]))
    scipy.io.savemat(path + name + '.mat', dict(bestepoch=best[0], loss=loss, std=std, L=L, name=name))
    files.download(path + name + '.mat')

    # data = []
    # for i, s in enumerate(labels):
    #    data.append(go.Scatter(x=np.arange(epochs), y=L[:, i], mode='markers+lines', name=s))
    # layout = go.Layout(xaxis=dict(type='log', autorange=True),
    #                   yaxis=dict(type='log', autorange=True))
    # configure_plotly_browser_state()
    # iplot(go.Figure(data=data, layout=layout))


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


H = [76, 23, 7]
# model = torch.nn.Sequential(
#        waveconv(),
#        torch.nn.Linear(498, H[0]), torch.nn.Tanh(),
#        torch.nn.Linear(H[0], H[1]), torch.nn.Tanh(),
#        torch.nn.Linear(H[1], H[2]), torch.nn.Tanh(),
#        torch.nn.Linear(H[2], 2))
runexample(H, None)
