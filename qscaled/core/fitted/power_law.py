import numpy as np
import scipy


def power_law_with_const(x, a, b, c):
    return c + (x / (b + 1e-12)) ** (-a)


def power_law_with_const_objective(args, *params):
    x, y = params
    a, b, c = args
    return ((power_law_with_const(x, a, b, c) - y) ** 2).mean()


def fit_powerlaw(xs, y, top_k=500):
    optim_f = power_law_with_const_objective
    init_grid = [slice(-2, 2, 0.8), slice(0, 1, 0.2), slice(-2, 2, 0.8)]
    _, _, brute_xs, brute_ys = scipy.optimize.brute(
        optim_f,
        init_grid,
        args=(xs, y),
        full_output=True,
        finish=None,
        Ns=1,
        workers=-1,
    )

    brute_xs = brute_xs.reshape(brute_xs.shape[0], -1)
    brute_ys = brute_ys.reshape(-1)

    top_idxs = np.argsort(brute_ys)[:top_k]
    top_xs = brute_xs[:, top_idxs]
    preds = []
    for i in range(top_xs.shape[1]):
        pred = scipy.optimize.minimize(optim_f, top_xs[:, i], args=(xs, y), method='L-BFGS-B').x
        loss = optim_f(pred, xs, y)
        preds.append((pred, loss))
    return sorted(preds, key=lambda x: x[1])[0][0]
