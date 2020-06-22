import numpy as np
import tensorflow as tf


def analytic_solution(X):
    return np.sin(X[:, 0])


def get_pdefn():

    def pdefn(X, U, g):

        g.__exit__(None, None, None)
        u_x = g.gradient(U, X[0])

        if u_x is not None:
            return tf.cos(X[0]) - u_x
        else:
            return tf.ones_like(U) * np.inf

    return pdefn


def get_training_data(low=0, high=np.pi, n_df=64):

    X_boundary = np.array([low, high])[:, None]
    U_boundary = analytic_solution(X_boundary)

    X_df = np.linspace(low, high, n_df)[:, None]

    return X_boundary, U_boundary, X_df


def get_testing_data(low=0, high=np.pi, n=128):

    X = np.linspace(low, high, n)[:, None]
    U = analytic_solution(X)

    return X, U
