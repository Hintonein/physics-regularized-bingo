import numpy as np
import tensorflow as tf

import problems.poisson
import problems.linear_advection
import problems.burgers
import problems.full_pendulum
import problems.simple as simple

tf.config.set_visible_devices([], 'GPU')


def get_test(test_name, *args):
    return dispatch[test_name](*args)


def test_linear(m, b):
    X, U, X_df = simple.linear.get_training_data(m, b)

    pdefn = simple.linear.get_pdefn(m, b)

    return X, U, X_df, pdefn, 1


def test_trig():
    X, U, X_df = simple.trig.get_training_data()

    pdefn = simple.trig.get_pdefn()

    return X, U, X_df, pdefn, 1


def test_exp(k):
    X, U, X_df = simple.exp.get_training_data(k)

    pdefn = simple.exp.get_pdefn(k)

    return X, U, X_df, pdefn, 1


def test_shm(omega):

    X, U, X_df = simple.shm.get_training_data(omega)

    pdefn = simple.shm.get_pdefn(omega)

    return X, U, X_df, pdefn, 2


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
    "linear": test_linear,
    "trig": test_trig,
    "exp": test_exp,
    "shm": test_shm,
    "linear_advection": linear_advection,
    "burgers": test_burgers,
    "burgers_dense": test_burgers_dense,
    "burgers_slice": test_burgers_slice,
    "pendulum": test_pendulum,
    "poisson": test_poisson,
}
