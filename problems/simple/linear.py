import numpy as np
import torch


def analytic_solution(X, m, b):
    return (X**2) * m + b


def get_pdefn(m, b):

    def pdefn(X, U):
        if U.grad_fn is not None:

            u_x = torch.autograd.grad(U.sum(), X[0])[0]

            if u_x is not None:
                return m * 2 * X[0] + b - u_x

        return torch.ones_like(U) * np.inf

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
