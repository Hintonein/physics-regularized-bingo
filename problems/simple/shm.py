import numpy as np
import tensorflow as tf


def analytic_solution(X, omega):
    return np.sin(X * omega)


def get_pdefn(omega):

    def pdefn(X, U, g):

        u_t = g.gradient(U, X[0])
        g.__exit__(None, None, None)
        if u_t is not None:
            u_tt = g.gradient(u_t, X[0])

            if u_tt is not None:
                return u_tt + omega**2 * U

        return tf.ones_like(U) * np.inf

    return pdefn


def get_training_data(omega, low=0, high=1, n_df=128):

    X_boundary = np.array([low, high])[:, None]
    U_boundary = analytic_solution(X_boundary, omega)

    X_df = np.linspace(low, high, n_df)[:, None]

    return X_boundary, U_boundary, X_df


def get_testing_data(omega, low=0, high=1, n=256):

    X = np.linspace(low, high, n)[:, None]
    U = analytic_solution(X, omega)

    return X, U
