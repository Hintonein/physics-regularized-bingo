import numpy as np
from scipy.integrate import solve_ivp
import torch


def get_pdefn(omega):

    def pdefn(X, U):

        if U.grad_fn is not None:

            u_t = torch.autograd.grad(U, X[0], create_graph=True)[0]
            if u_t is not None:
                u_tt = torch.autograd.grad(u_t, X[0])[0]
                if u_tt is not None:
                    return u_tt + omega**2 * torch.sin(U)

        return torch.ones_like(U) * np.inf

    return pdefn


def get_test_data(omega, t_start=0, t_end=2, n=128):
    solution = get_integral_solution(omega, t_start, t_end, n)

    X = solution.t[:, None]
    U = (solution.y[0, :])[:, None]

    return X, U


def get_training_data(omega, t_start=0, t_end=2, n=128):

    solution = get_integral_solution(omega, t_start, t_end, n)

    X = np.array([0.05, 0.1])[:, None]
    U = (solution.sol(X[:, 0])[0, :])[:, None]

    X_df = np.linspace(t_start, t_end, n)[:, None]

    return X, U, X_df


def get_integral_solution(omega, t_start=0, t_end=2, n=128):

    def pendulum_system(t, xv):
        x, v = xv
        dx_dt = v
        dv_dt = -omega**2 * np.sin(x)

        return [dx_dt, dv_dt]

    initial_conditions = np.array([-np.pi / 2, 0])

    solution = solve_ivp(pendulum_system, (0, 2),
                         initial_conditions, max_step=0.001, dense_output=True)

    return solution
