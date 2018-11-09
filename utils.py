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


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    nP = sum(x.numel() for x in model.parameters())  # number parameters
    nG = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%4s %70s %9s %12s %20s %12s %12s' % ('', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%4g %70s %9s %12g %20s %12g %12g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('\n%g layers, %g parameters, %g gradients' % (i + 1, nP, nG))
