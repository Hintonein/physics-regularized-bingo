import numpy as np
import torch


def analytic_solution(X, omega):
    return np.sin(X * omega)


def get_pdefn(omega):

    def pdefn(X, U):

        if U.grad_fn is not None:
            u_t = torch.autograd.grad(U.sum(), X[0], create_graph=True)[0]
            if u_t is not None and u_t.grad_fn is not None:
                u_tt = torch.autograd.grad(
                    u_t.sum(), X[0], allow_unused=True)[0]

                if u_tt is not None:
                    return u_tt + omega**2 * U

        return torch.ones_like(U) * np.inf

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
