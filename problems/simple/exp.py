import numpy as np
import tensorflow as tf


def analytic_solution(X, k):
    return np.exp(k * X)


def get_pdefn(k):

    def pdefn(X, U, g):

        g.__exit__(None, None, None)
        u_x = g.gradient(U, X[0])

        if u_x is not None:
            return u_x - U * k
        else:
            return tf.ones_like(U) * np.inf

    return pdefn


def get_training_data(k, low=-1, high=1, n_df=64):
    X_boundary = np.array([low, high])[:, None]
    U_boundary = analytic_solution(X_boundary, k)

    X_df = np.linspace(low, high, n_df)[:, None]

    return X_boundary, U_boundary, X_df


def get_testing_data(k, low, high, n=128):

    X = np.linspace(low, high, n)[:, None]
    U = analytic_solution(X, k)

    return X, U
