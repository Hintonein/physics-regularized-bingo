import numpy as np
import torch
import problems.data_generation_helpers as util


def analytic_solution(X, v):
    return np.sin(X[:, 0] - v * X[:, 1])


def get_pdefn(v):

    def pdefn(X, U):
        if U.grad_fn is not None:

            u_x = torch.autograd.grad(
                U, X[0], create_graph=True, allow_unused=True)[0]
            u_t = torch.autograd.grad(
                U, X[1], create_graph=True, allow_unused=True)[0]

            if u_x is not None and u_t is not None:
                return (u_t + v * u_x)

        return torch.ones_like(U) * np.inf

    return pdefn


def gen_training_data(v, low=[0, 0], high=[2 * np.pi, 1], n_b=256, n_df=10000):

    top, bottom, initial, _ = util.boundary_X_2d_separate(
        low, high, n_b)

    X_boundary = np.vstack([initial, top, bottom])
    U_boundary = analytic_solution(X_boundary, v)

    X_df = np.random.uniform(low=low, high=high, size=(n_df, 2))

    return X_boundary, U_boundary, X_df


def gen_testing_data(v, low=[0, 0], high=[2 * np.pi, 1], n=256):
    X = util.grid_X_2d(low, high, n)
    U = analytic_solution(X, v)[:, None]

    return X, U
