import numpy as np
from problems.data_generation_helpers import load_mat_2d, ungrid_u_2d, get_u_const_idx_2d, random_choice
import os
import tensorflow as tf


def load_data():
    path = os.path.dirname(os.path.abspath(__file__))
    filename = f"{path}/../data/burgers_shock.mat"
    return load_mat_2d(filename, "x", "t", "usol")


def get_training_data(n_df=10000):

    x, t, u = load_data()

    X_initial, U_initial = get_u_const_idx_2d(x, t, u, idx_2=0)
    X_bottom, U_bottom = get_u_const_idx_2d(x, t, u, idx_1=0)
    X_top, U_top = get_u_const_idx_2d(x, t, u, idx_1=x.shape[0]-1)

    X_boundary = np.vstack([X_initial, X_bottom, X_top])
    U_boundary = np.vstack([U_initial, U_bottom, U_top])

    X, _ = ungrid_u_2d(x, t, u)

    X_df = random_choice(X, n_df)

    return X_boundary, U_boundary, X_df


def get_test_data():

    x, t, u = load_data()

    X, U = ungrid_u_2d(x, t, u)

    return X, U


def get_pdefn():
    nu = 0.01 / np.pi

    def pdefn(X, U, g):

        U_1 = g.gradient(U, X)
        if U_1 is not None:

            u_x = U_1[:, 0]
            u_t = U_1[:, 1]

            U_xx = g.gradient(u_x, X)
            if U_xx is not None:
                u_xx = U_xx[:, 0]
                return u_t + U*u_x - nu*u_xx

        return tf.ones_like(U) * np.inf

    return pdefn
