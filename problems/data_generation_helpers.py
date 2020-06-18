import numpy as np
from scipy.io import loadmat


def boundary_X_2d_separate(low, high, n):
    vary_0 = np.linspace(low[0], high[0], n)[:, None]
    vary_1 = np.linspace(low[1], high[1], n)[:, None]

    const_low_0 = np.ones((n, 1)) * low[0]
    const_low_1 = np.ones((n, 1)) * low[1]

    const_high_0 = np.ones((n, 1)) * high[0]
    const_high_1 = np.ones((n, 1)) * high[1]

    c0_low = np.hstack([const_low_0, vary_1])
    c0_high = np.hstack([const_high_0, vary_1])
    c1_low = np.hstack([vary_0, const_low_1])
    c1_high = np.hstack([vary_0, const_high_1])

    return c0_low, c0_high, c1_low, c1_high


def boundary_X_2d(low, high, n):
    c0_low, c0_high, c1_low, c1_high = boundary_X_2d_separate(low, high, n)

    return np.vstack([c0_low, c0_high, c1_low, c1_high])


def grid_X_2d(low, high, n):
    X = np.empty((n*n, 2))

    for i, x in enumerate(np.linspace(low[0], high[0], n)):
        for j, y in enumerate(np.linspace(low[1], high[1], n)):
            idx = i*n + j
            X[idx, 0] = x
            X[idx, 1] = y

    return X


def load_mat_2d(filename, x1_key, x2_key, u_key):
    data = loadmat(filename)

    x1 = data[x1_key]
    x2 = data[x2_key]
    u = np.real(data[u_key])

    return x1, x2, u


def ungrid_u_2d(x1, x2, u):
    n_total = x1.shape[0] * x2.shape[0]

    X = np.zeros((n_total, 2))
    U = np.zeros((n_total, 1))

    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            idx = i*x2.shape[0] + j
            X[idx, 0] = x1[i]
            X[idx, 1] = x2[j]
            U[idx, 0] = u[i, j]

    return X, U


def get_u_const_idx_2d(x1, x2, u, idx_1=None, idx_2=None):

    if idx_1 is not None:
        U = np.zeros((x2.shape[0], 1))
        X = np.zeros((x2.shape[0], 2))
        i = idx_1
        for j in range(x2.shape[0]):
            U[j, 0] = u[i, j]
            X[j, 0], X[j, 1] = x1[i], x2[j]

        return X, U
    elif idx_2 is not None:
        U = np.zeros((x1.shape[0], 1))
        X = np.zeros((x1.shape[0], 2))
        j = idx_2
        for i in range(x1.shape[0]):
            U[i, 0] = u[i, j]
            X[i, 0], X[i, 1] = x1[i], x2[j]

        return X, U
    else:
        return None


def random_choice(X, size=10000):
    idx = np.random.choice(list(range(X.shape[0])), size=size)
    return X[idx, :]
