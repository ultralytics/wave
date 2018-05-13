import numpy as np


def normalize(x, axis=None):  # normalize x mean and std by axis
    if axis is None:
        mu, sigma = x.mean(), x.std()
    elif axis == 0:
        mu, sigma = x.mean(0), x.std(0)
    elif axis == 1:
        mu, sigma = x.mean(1).reshape(x.shape[0], 1), x.std(1).reshape(x.shape[0], 1)
    return (x - mu) / sigma, mu, sigma


def shuffledata(x, y):  # randomly shuffle x and y by same axis=0 indices
    i = np.arange(x.shape[0])
    np.random.shuffle(i)
    return x[i], y[i]


def splitdata(x, y, train=0.7, validate=0.15, test=0.15, shuffle=False):  # split training data
    n = x.shape[0]
    if shuffle:
        x, y = shuffledata(x, y)
    i = round(n * train)  # train
    j = round(n * validate) + i  # validate
    k = round(n * test) + j  # test
    return x[:i], y[:i], x[i:j], y[i:j], x[j:k], y[j:k]  # xy train, xy validate, xy test


def stdpt(r, ys):  # MSE loss + standard deviation (pytorch)
    r = r.detach()
    loss = (r ** 2).mean().cpu().item()
    std = r.std(0).cpu().numpy() * ys
    return loss, std


def stdtf(r, ys):  # MSE loss + standard deviation (tf eager)
    r = r.numpy()
    loss = (r ** 2).mean()
    std = r.std(0) * ys
    return loss, std
