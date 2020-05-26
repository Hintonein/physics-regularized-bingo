import data.util as util
import os


def load_burgers_grid():
    path = os.path.dirname(os.path.abspath(__file__))
    filename = f"{path}/burgers_shock.mat"
    return util.load_mat_2d(filename, "x", "t", "usol")

# In a format ready to be used by the PINN input


def load_burgers_flat():
    x, t, u = load_burgers_grid()

    X, U = util.ungrid_u_2d(x, t, u)
    return X, U, [x, t, u]


def load_burgers_bounds():
    X, U, [x, t, u] = load_burgers_flat()

    X_initial, U_initial = util.get_u_const_idx_2d(x, t, u, idx_2=0)
    X_bottom, U_bottom = util.get_u_const_idx_2d(x, t, u, idx_1=0)
    X_top, U_top = util.get_u_const_idx_2d(x, t, u, idx_1=x.shape[0]-1)

    return X, U, [X_initial, X_bottom, X_top], [U_initial, U_bottom, U_top], [x, t, u]
