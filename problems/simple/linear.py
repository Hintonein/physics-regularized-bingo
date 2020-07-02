import numpy as np
import tensorflow as tf


def analytic_solution(X, m, b):
    return (X**2) * m + b


def get_pdefn(m, b):

    def pdefn(X, U, g):
        g.__exit__(None, None, None)
        u_x = g.gradient(U, X[0])

        if u_x is not None:
            return m * X[0] + b - u_x
        else:
            return tf.ones_like(U) * np.inf

    return pdefn


def get_training_data(m, b, low=0, high=10, n_df=64):

    X_boundary = np.array([low, high])[:, None]
    U_boundary = analytic_solution(X_boundary, m, b)

    X_df = np.linspace(low, high, n_df)[:, None]

    return X_boundary, U_boundary, X_df


def get_testing_data(m, b, low=0, high=10, n=256):
    X = np.linspace(low, high, n)[:, None]
    U = analytic_solution(X, m, b)

    return X, U
