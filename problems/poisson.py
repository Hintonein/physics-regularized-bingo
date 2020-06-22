import numpy as np
import tensorflow as tf
import problems.data_generation_helpers as util


def analytic_solution(X, k):
    return np.sin(k*X[:, 0]) * np.sin(k*X[:, 1])


def get_pdefn(k):

    def pdefn(X, U, g):
        u_x = g.gradient(U, X[0])
        u_y = g.gradient(U, X[1])
        # Manually exit so that these ops don't get added to the tape
        # This results in a 10-25% speedup
        g.__exit__(None, None, None)
        if u_x is not None and u_y is not None:

            u_xx = g.gradient(u_x, X[0])
            u_yy = g.gradient(u_y, X[1])

            if u_xx is not None and u_yy is not None:

                return u_xx + u_yy + 2*(k**2)*tf.sin(X[0])*tf.sin(X[1])

        return tf.ones_like(U) * np.inf

    return pdefn


def gen_training_data(k, low=[0, 0], high=[1, 1], n_b=64, n_df=5000):

    X_boundary = util.boundary_X_2d(low, high, n_b)
    U_boundary = analytic_solution(X_boundary, k)[:, None]

    X_df = np.random.uniform(low=low, high=high, size=(n_df, 2))

    return X_boundary, U_boundary, X_df


def gen_testing_data(k, low=[0, 0], high=[1, 1], n=256):
    X = util.grid_X_2d(low, high, n)
    U = analytic_solution(X, k)[:, None]

    return X, U
