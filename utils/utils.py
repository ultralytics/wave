import copy
import time

import numpy as np
import torch

# Set printoptions
torch.set_printoptions(linewidth=1320, precision=5, profile="long")
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})  # format short g, %precision=5


def normalize(x, axis=None):  # normalize x mean and std by axis
    """Normalize an array 'x' by its mean and standard deviation along the specified axis."""
    if axis is None:
        mu, sigma = x.mean(), x.std()
    elif axis == 0:
        mu, sigma = x.mean(0), x.std(0)
    elif axis == 1:
        s = (x.shape[0], 1)
        mu, sigma = x.mean(1).reshape(s), x.std(1).reshape(s)
    return (x - mu) / sigma, mu, sigma


def shuffledata(x, y):  # randomly shuffle x and y by same axis=0 indices
    """Randomly shuffles arrays x and y along the same axis=0 indices."""
    i = np.arange(x.shape[0])
    np.random.shuffle(i)
    return x[i], y[i]


def splitdata(x, y, train=0.7, validate=0.15, test=0.15, shuffle=False):  # split training data
    """Splits data arrays x and y into training, validation, and test sets with optional shuffling."""
    n = x.shape[0]
    if shuffle:
        x, y = shuffledata(x, y)
    i = round(n * train)  # train
    j = round(n * validate) + i  # validate
    k = round(n * test) + j  # test
    return x[:i], y[:i], x[i:j], y[i:j], x[j:k], y[j:k]  # xy train, xy validate, xy test


def stdpt(r, ys):  # MSE loss + standard deviation (pytorch)
    """Calculate Mean Squared Error loss and standard deviation of tensor r, scaled by ys."""
    r = r.detach()
    loss = (r**2).mean().cpu().item()
    std = r.std(0).cpu().numpy() * ys
    return loss, std


def stdtf(r, ys):  # MSE loss + standard deviation (tf eager)
    """Calculate Mean Squared Error loss and standard deviation of tensor r using TensorFlow, scaled by ys."""
    r = r.numpy()
    loss = (r**2).mean()
    std = r.std(0) * ys
    return loss, std


def model_info(model):
    """Print a detailed line-by-line summary of a PyTorch model including layer name, gradient status, parameters,
    shape, mean, and std.
    """
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print("\n%5s %40s %9s %12s %20s %10s %10s" % ("layer", "name", "gradient", "parameters", "shape", "mu", "sigma"))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace("module_list.", "")
        print(
            "%5g %40s %9s %12g %20s %10.3g %10.3g"
            % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std())
        )
    print(f"Model Summary: {i + 1:g} layers, {n_p:g} parameters, {n_g:g} gradients")


class patienceStopper:
    """Implements early stopping mechanism for training models based on validation loss and patience criteria."""

    def __init__(self, patience=10, verbose=True, epochs=1000, printerval=10):
        """Initialize a patience stopper with given parameters for early stopping in training."""
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
        """Resets tracking metrics to initial states for the training process."""
        self.bestloss = float("inf")
        self.bestmetrics = None
        self.num_bad_epochs = 0

    def step(self, loss, metrics=None, model=None):
        """Updates internal state for each training epoch, tracking loss and metrics, and printing progress
        periodically.
        """
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
            self.final(f"{self.patience:g} Patience exceeded at epoch {self.epoch:g}.")
            return True
        elif self.epoch >= self.epochs:
            self.final(f"WARNING: {self.patience:g} Patience not exceeded by epoch {self.epoch:g} (train longer).")
            return True
        else:
            return False

    def first(self, model):
        """Prints a formatted header for model training details including epoch, time, loss, and metrics."""
        s = ("epoch", "time", "loss", "metric(s)")
        print("%12s" * len(s) % s)

    def printepoch(self, epoch, loss, metrics):
        """Prints the epoch number, elapsed time, loss, and optional metrics for model training."""
        s = (epoch, time.time() - self.t, loss)
        if metrics is not None:
            for i in range(len(metrics)):
                s += (metrics[i],)
        print("%12.5g" * len(s) % s)
        self.t = time.time()

    def final(self, msg):
        """Print final results and completion message with total elapsed time and best training epoch details."""
        dt = time.time() - self.t0
        print(
            f"{msg}\nFinished {self.epochs + 1:g} epochs in {dt:.3f}s ({(self.epochs + 1) / dt:.3f} epochs/s). Best results:"
        )
        self.printepoch(self.bestepoch, self.bestloss, self.bestmetrics)
