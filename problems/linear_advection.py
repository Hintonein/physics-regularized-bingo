import numpy as np
import tensorflow as tf
import problems.data_generation_helpers as util


def analytic_solution(X, v):
    return np.sin(X[:, 0] - v * X[:, 1])


def get_pdefn(v):

    def pdefn(X, U, g):
        g.__exit__(None, None, None)
        u_x = g.gradient(U, X[0])
        u_t = g.gradient(U, X[1])

        if u_x is not None and u_t is not None:
            return (u_t + v * u_x)
        else:
            return tf.ones_like(U)*np.inf

    return pdefn


def gen_training_data(v, low=[0, 0], high=[2*np.pi, 1], n_b=256, n_df=10000):

    top, bottom, initial, _ = util.boundary_X_2d_separate(
        low, high, n_b)

    X_boundary = np.vstack([initial, top, bottom])
    U_boundary = analytic_solution(X_boundary, v)

    X_df = np.random.uniform(low=low, high=high, size=(n_df, 2))

    return X_boundary, U_boundary, X_df


def gen_testing_data(v, low=[0, 0], high=[2*np.pi, 1], n=256):
    X = util.grid_X_2d(low, high, n)
    U = analytic_solution(X, v)[:, None]

    return X, U
