import os
import time

import plotly.graph_objs as go
import scipy.io
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from plotly.offline import plot
from utils.utils import *

tf.enable_eager_execution()


def runexample(H, model, str):
    lr = 0.002
    eps = 0.001
    epochs = 50000
    validations = 5000
    printInterval = 1000
    # batch_size = 10000
    data = "wavedata25ns.mat"

    cuda = tf.test.is_gpu_available()
    tf.set_random_seed(1)
    path = "data/"
    os.makedirs(f"{path}models", exist_ok=True)
    name = (data[:-4] + "%s%glr%geps%s" % (H[:], lr, eps, str)).replace(", ", "_").replace("[", "_").replace("]", "_")

    tica = time.time()
    device = "/gpu:0" if cuda else "/cpu:0"
    print(f"Running {name} on {device}")

    if not os.path.isfile(path + data):
        os.system(f"wget -P data/ https://storage.googleapis.com/ultralytics/{data}")
    mat = scipy.io.loadmat(path + data)
    x = mat["inputs"]  # inputs (nx512) [waveform1 waveform2]
    y = mat["outputs"][:, 0:2]  # outputs (nx4) [position(mm), time(ns), PE, E(MeV)]
    nz, nx = x.shape
    ny = y.shape[1]

    if model is None:
        # model = WAVE(nx, ny, H)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(H[0], activation=tf.tanh, input_shape=(512,)),  # must declare input shape
                tf.keras.layers.Dense(H[1], activation=tf.tanh),
                tf.keras.layers.Dense(
                    H[2],
                    activation=tf.tanh,
                ),
                tf.keras.layers.Dense(ny),
            ]
        )

    x, _, _ = normalize(x, 1)  # normalize each input row
    y, ymu, ys = normalize(y, 0)  # normalize each output column
    x, y, xv, yv, xt, yt = splitdata(x, y, train=0.70, validate=0.15, test=0.15, shuffle=False)
    labels = ["train", "validate", "test"]

    print(model)

    # if cuda:
    #    x, xv, xt = tf.convert_to_tensor(x).gpu(), tf.convert_to_tensor(xv).gpu(), tf.convert_to_tensor(xt).gpu()
    #    y, yv, yt = tf.convert_to_tensor(y).gpu(), tf.convert_to_tensor(yv).gpu(), tf.convert_to_tensor(yt).gpu()
    # model = model.gpu()

    # criteria and optimizer
    def criteria(y_pred, y):  # MSE
        return tf.reduce_mean(tf.square(y_pred - y))

    optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=eps)

    ticb = time.time()
    L = np.full((epochs, 3), np.nan)
    best = (0, 1e6, None)  # best (epoch, validation loss, model)
    with tf.device(device):
        for i in range(epochs):
            # Calculate derivatives of the input function with respect to its parameters.
            with tf.GradientTape() as tape:
                y_pred = model(x)
                loss = criteria(y_pred, y)
            grads = tape.gradient(loss, model.variables)  # DO NOT INDENT, not inside tf.GradientTape context manager
            y_predv = model(xv)

            # Compute and print loss
            L[i, 0] = loss.numpy()  # / y.numel()  # train
            L[i, 1] = criteria(y_predv, yv).numpy()  # / yv.numel()  # validate
            # L[i, 2] = criteria(model(xt), yt).numpy() #/ yv.numel()  # test

            if i > validations:  # validation checks
                if L[i, 1] < best[1]:
                    best = (i, L[i, 1], None)
                if (i - best[0]) > validations:
                    print("\n%g validation checks exceeded at epoch %g." % (validations, i))
                    break

            if i % printInterval == 0:  # print and save progress
                # scipy.io.savemat(path + name + '.mat', dict(bestepoch=best[0], loss=L[best[0]], L=L, name=name))
                _, std = stdtf(y_predv - yv, ys)
                print("%.3fs" % (time.time() - ticb), i, L[i], std)
                ticb = time.time()

            # Apply the gradient to the model
            optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())
        else:
            print("WARNING: Validation loss still decreasing after %g epochs (train longer)." % (i + 1))
        # torch.save(best[2], path + 'models/' + name + '.pt')
        # model.load_state_dict(best[2])
        dt = time.time() - tica

    print("\nFinished %g epochs in %.3fs (%.3f epochs/s)\nBest results from epoch %g:" % (i + 1, dt, i / dt, best[0]))
    loss, std = np.zeros(3), np.zeros((3, ny))
    for i, (xi, yi) in enumerate(((x, y), (xv, yv), (xt, yt))):
        loss[i], std[i] = stdtf(model(xi) - yi, ys)
        print("%.5f %s %s" % (loss[i], std[i, :], labels[i]))
    # scipy.io.savemat(path + name + '.mat', dict(bestepoch=best[0], loss=loss, std=std, L=L, name=name))
    # files.download(path + name + '.mat')

    data = []
    for i, s in enumerate(labels):
        data.append(go.Scatter(x=np.arange(epochs), y=L[:, i], mode="markers+lines", name=s))
    layout = go.Layout(xaxis=dict(type="linear", autorange=True), yaxis=dict(type="log", autorange=True))
    # configure_plotly_browser_state()
    plot(go.Figure(data=data, layout=layout))


if __name__ == "__main__":
    H = [128, 32, 8]
    for i in range(1):
        runexample(H, None, f".{str(i)}")
