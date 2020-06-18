import numpy as np
import tensorflow as tf

import problems.poisson
import problems.linear_advection
import problems.burgers
import problems.full_pendulum

tf.config.set_visible_devices([], 'GPU')


def get_test(test_name, *args):
    return dispatch[test_name](*args)


def format_training_data(bcs, n):
    x_bcs = np.asarray(
        bcs[0], dtype=np.float64).reshape((-1, 1))
    u_bcs = np.asarray(bcs[1], dtype=np.float64).reshape((-1, 1))

    x_df = np.linspace(
        bcs[0][0], bcs[0][1], n).reshape((-1, 1))[1:-1]

    return x_bcs, u_bcs, x_df


def test_constant(c):
    def odefun(x, u, g):
        u_x = g.gradient(u, x)
        if u_x is not None:
            return c-u_x[:, 0]
        else:
            return tf.zeros_like(u)

    # bcs : ((xmin, xmax), (ymin, ymax))
    bcs = ((0.0, 10.0), (0.0, 20.0))
    return odefun, bcs


def test_linear(m, b):
    def odefun(x, u, g):
        u_x = g.gradient(u, x)
        if u_x is not None:
            return m*x[:, 0] + b - u_x[:, 0]
        else:
            return m*x[:, 0] + b
    # bcs : ((xmin, xmax), (ymin, ymax))
    bcs = ((0.0, 10.0), (0.0, 100.0))
    return odefun, bcs


def test_trig():
    def odefun(x, u, g):

        u_x = g.gradient(u, x)

        if u_x is not None:
            return tf.cos(x)[:, 0] - u_x[:, 0]
        else:
            return tf.cos(x)[:, 0]

    # bcs : ((xmin, xmax), (ymin, ymax))
    bcs = ((0.0, np.pi), (0.0, 0.0))
    return odefun, bcs


def test_exp(k):

    def odefun(x, u, g):
        u_x = g.gradient(u, x)

        if u_x is not None:
            return (u_x[:, 0] - u*k)
        else:
            return tf.ones_like(u) * np.inf

    # bcs : ((xmin, xmax), (ymin, ymax))
    bcs = ((-1, 1), (np.exp(-1*k), np.exp(1*k)))

    return odefun, bcs


def test_shm(omega):

    def odefun(x, u, g):
        u_x = g.gradient(u, x)

        if u_x is not None:
            u_xx = g.gradient(u_x[:, 0], x)

            if u_xx is not None:
                return (u_xx[:, 0] + omega**2 * u)

        return tf.ones_like(u) * np.inf  # Disallow constants

    X = np.array([0.05, 0.1])[:, None]
    U = np.sin(omega * X)
    X_df = np.linspace(0, 1, 128)[:, None]

    return X, U, X_df, odefun, 2


def test_shm_const(omega):

    def odefun(x, u, g):
        u_x = g.gradient(u, x)

        if u_x is not None:
            u_xx = g.gradient(u_x[:, 0], x)

            if u_xx is not None:
                return (u_xx[:, 0] + omega**2 * u)

        return tf.ones_like(u) * np.inf

    X = np.array([0.05, 0.1])[:, None]
    U = np.sin(omega * X)
    X_const = np.ones((2, 1)) * omega
    X = np.hstack([X, X_const])
    X_df = np.linspace(0, 1, 128)[:, None]
    X_df_const = np.ones((128, 1)) * omega
    X_df = np.hstack([X_df, X_df_const])

    return X, U, X_df, odefun, 2


def test_shm_dense(omega):

    def odefun(x, u, g):
        u_x = g.gradient(u, x)

        if u_x is not None:
            u_xx = g.gradient(u_x[:, 0], x)

            if u_xx is not None:
                return (u_xx[:, 0] + omega**2 * u)

        return tf.ones_like(u) * np.inf

    X = np.linspace(0, 1, 128)[:, None]
    U = np.sin(omega * X)
    X = np.hstack([X, np.ones((128, 1))*omega])
    X_df = np.linspace(0, 1, 1)[:, None]
    X_df = np.hstack([X_df,  np.ones((1, 1))*omega])

    return X, U, X_df, odefun, 2


def linear_advection(v):
    ''' 1D transport equation with initial condion sin(x), periodic boundary conditions'''

    X_boundary, U_boundary, X_df = problems.linear_advection.gen_training_data(
        v)

    pdefn = problems.linear_advection.get_pdefn(v)

    return X_boundary, U_boundary, X_df, pdefn, 1


def test_burgers():

    X, U, X_df = problems.burgers.get_training_data()

    pdefn = problems.burgers.get_pdefn()

    return X, U, X_df, pdefn, 2


def test_burgers_dense(n=1000):

    X_true, U_true = problems.burgers.get_test_data()

    idx = np.random.choice(list(range(X_true.shape[0])), size=n)

    pdefn = problems.burgers.get_pdefn()

    X = X_true[idx, :]
    U = U_true[idx, :]
    X_df = X_true[idx, :]

    return X, U, X_df, pdefn, 2


def test_burgers_slice():

    x, t, u = problems.burgers.load_data()

    t_idx = int(t.shape[0] * .75)  # Find the index for t

    X = np.empty((x.shape[0], 2))
    U = np.empty((x.shape[0], 1))
    for i, x in enumerate(x):
        X[i, 0] = x
        X[i, 1] = t[t_idx]
        U[i, 0] = u[i, t_idx]

    pdefn = problems.burgers.get_pdefn()

    return X, U, X, pdefn, 2


def test_pendulum(omega):

    X, U, X_df = problems.full_pendulum.get_training_data(omega)

    pdefn = problems.full_pendulum.get_pdefn(omega)

    return X, U, X_df, pdefn, 2


def test_poisson(k):
    '''Poission equation with solutions sin(kx)*sin(ky)'''

    X, U, X_df = problems.poisson.gen_training_data(k)
    pdefn = problems.poisson.get_pdefn(k)

    return X, U, X_df, pdefn, 2


dispatch = {
    "shm": test_shm,
    "linear_advection": linear_advection,
    "burgers": test_burgers,
    "burgers_dense": test_burgers_dense,
    "burgers_slice": test_burgers_slice,
    "pendulum": test_pendulum,
    "poisson": test_poisson,
}
