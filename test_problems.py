import numpy as np
import tensorflow as tf
import data.load
from scipy.integrate import solve_ivp


def get_test(test_name, *args):
    return dispatch[test_name](*args)


def format_training_data(bcs, n):
    x_bcs = np.asarray(
        bcs[0], dtype=np.float64).reshape((-1, 1))
    u_bcs = np.asarray(bcs[1], dtype=np.float64).reshape((-1, 1))

    x_df = np.linspace(
        bcs[0][0], bcs[0][1], n).reshape((-1, 1))[1:-1]

    return x_bcs, u_bcs, x_df


def test_constant(c):
    def odefun(x, u, g):
        u_x = g.gradient(u, x)
        if u_x is not None:
            return c-u_x[:, 0]
        else:
            return tf.zeros_like(u)

    # bcs : ((xmin, xmax), (ymin, ymax))
    bcs = ((0.0, 10.0), (0.0, 20.0))
    return odefun, bcs


def test_linear(m, b):
    def odefun(x, u, g):
        u_x = g.gradient(u, x)
        if u_x is not None:
            return m*x[:, 0] + b - u_x[:, 0]
        else:
            return m*x[:, 0] + b
    # bcs : ((xmin, xmax), (ymin, ymax))
    bcs = ((0.0, 10.0), (0.0, 100.0))
    return odefun, bcs


def test_trig():
    def odefun(x, u, g):

        u_x = g.gradient(u, x)

        if u_x is not None:
            return tf.cos(x)[:, 0] - u_x[:, 0]
        else:
            return tf.cos(x)[:, 0]

    # bcs : ((xmin, xmax), (ymin, ymax))
    bcs = ((0.0, np.pi), (0.0, 0.0))
    return odefun, bcs


def test_exp(k):

    def odefun(x, u, g):
        u_x = g.gradient(u, x)

        if u_x is not None:
            return (u_x[:, 0] - u*k)
        else:
            return tf.ones_like(u) * np.inf

    # bcs : ((xmin, xmax), (ymin, ymax))
    bcs = ((-1, 1), (np.exp(-1*k), np.exp(1*k)))

    return odefun, bcs


def test_shm(omega):

    def odefun(x, u, g):
        u_x = g.gradient(u, x)

        if u_x is not None:
            u_xx = g.gradient(u_x[:, 0], x)

            if u_xx is not None:
                return (u_xx[:, 0] + omega**2 * u)

        return tf.ones_like(u) * np.inf  # Disallow constants

    X = np.array([0.05, 0.1])[:, None]
    U = np.sin(omega * X)
    X_df = np.linspace(0, 1, 128)[:, None]

    return X, U, X_df, odefun, 2


def test_shm_const(omega):

    def odefun(x, u, g):
        u_x = g.gradient(u, x)

        if u_x is not None:
            u_xx = g.gradient(u_x[:, 0], x)

            if u_xx is not None:
                return (u_xx[:, 0] + omega**2 * u)

        return tf.ones_like(u) * np.inf

    X = np.array([0.05, 0.1])[:, None]
    U = np.sin(omega * X)
    X_const = np.ones((2, 1)) * omega
    X = np.hstack([X, X_const])
    X_df = np.linspace(0, 1, 128)[:, None]
    X_df_const = np.ones((128, 1)) * omega
    X_df = np.hstack([X_df, X_df_const])

    return X, U, X_df, odefun, 2


def test_shm_dense(omega):

    def odefun(x, u, g):
        u_x = g.gradient(u, x)

        if u_x is not None:
            u_xx = g.gradient(u_x[:, 0], x)

            if u_xx is not None:
                return (u_xx[:, 0] + omega**2 * u)

        return tf.ones_like(u) * np.inf

    X = np.linspace(0, 1, 128)[:, None]
    U = np.sin(omega * X)
    X = np.hstack([X, np.ones((128, 1))*omega])
    X_df = np.linspace(0, 1, 1)[:, None]
    X_df = np.hstack([X_df,  np.ones((1, 1))*omega])

    return X, U, X_df, odefun, 2


def test_transport(v, n_x, n_t):
    ''' 1D transport equation with initial condion sin(x), periodic boundary condition'''

    def pdefun(X, U, g):
        U_1 = g.gradient(U, X)

        if U_1 is not None:

            u_x = U_1[:, 0]
            u_t = U_1[:, 1]

            return (u_t + v * u_x)
        else:
            return tf.ones_like(U)*np.inf

    def solution_true(X):
        return np.sin(X[:, 0] - X[:, 1]*v).reshape((X.shape[0], 1))

    X_init_x = np.linspace(0, 2*np.pi, n_x).reshape((n_x, 1))
    X_init = np.hstack([X_init_x, np.zeros((n_x, 1))])

    X_min_x = np.zeros((n_t, 1))
    X_min_t = np.linspace(0, 1, n_t).reshape((n_t, 1))
    X_min = np.hstack([X_min_x, X_min_t])

    X_max_x = np.ones((n_t, 1)) * 2 * np.pi
    X_max_t = np.linspace(0, 1, n_t).reshape((n_t, 1))
    X_max = np.hstack([X_max_x, X_max_t])

    X_boundary = np.vstack([X_init, X_min, X_max])
    U_boundary = solution_true(X_boundary)

    X_df = np.empty((n_x*n_t, 2))
    for i, x in enumerate(np.linspace(0, 2*np.pi, n_x)):
        for j, t in enumerate(np.linspace(0, 1, n_t)):
            idx = i*n_t + j
            X_df[idx, 0], X_df[idx, 1] = x, t

    return X_boundary, U_boundary, X_df, pdefun, 1


def test_burgers():

    nu = 0.01 / np.pi

    def pdefun(X, U, g):

        U_1 = g.gradient(U, X)
        if U_1 is not None:

            u_x = U_1[:, 0]
            u_t = U_1[:, 1]

            U_xx = g.gradient(u_x, X)
            if U_xx is not None:
                u_xx = U_xx[:, 0]
                return u_t + U*u_x - nu*u_xx

        return tf.ones_like(U) * np.inf

    X_true, U_true, X_bounds, U_bounds, _ = data.load.load_burgers_bounds()

    X = np.vstack(X_bounds)
    U = np.vstack(U_bounds)

    X_df = np.random.uniform(low=[-1, 0], high=[1, 1], size=(5000, 2))

    return X, U, X_df, pdefun, 2


def test_burgers_dense(n=1000):

    nu = 0.01 / np.pi

    def pdefun(X, U, g):

        U_1 = g.gradient(U, X)
        if U_1 is not None:

            u_x = U_1[:, 0]
            u_t = U_1[:, 1]

            U_xx = g.gradient(u_x, X)
            if U_xx is not None:
                u_xx = U_xx[:, 0]
                return u_t + U*u_x - nu*u_xx

        return tf.ones_like(U) * np.inf

    X_true, U_true, _ = data.load.load_burgers_flat()

    idx = np.random.choice(list(range(X_true.shape[0])), size=n)

    X = X_true[idx, :]
    U = U_true[idx, :]
    X_df = X_true[idx, :]

    return X, U, X_df, pdefun, 2


def test_pendulum(omega):

    def odefun(X, U, g):
        U_1 = g.gradient(U, X)
        if U_1 is not None:
            U_2 = g.gradient(U_1[:, 0], X)
            if U_2 is not None:
                return U_2[:, 0] + omega**2 * tf.sin(U)

        return tf.ones_like(U)*np.inf

    def pendulum_system(t, xv):
        x, v = xv
        dx_dt = v
        dv_dt = -omega**2 * np.sin(x)

        return [dx_dt, dv_dt]

    initial_conditions = np.array([-np.pi/2, 0])

    solution = solve_ivp(pendulum_system, (0, 2),
                         initial_conditions, max_step=0.001, dense_output=True)

    X = np.array([0.05, 0.1])[:, None]
    U = (solution.sol(X[:, 0])[0, :])[:, None]
    print(U)

    X_true = solution.t[:, None]
    U_true = (solution.y[0, :])[:, None]

    X_df = np.linspace(0, 2, 128)[:, None]

    return X, U, X_df, odefun, 2


def test_poisson(k):
    '''Poission equation with solutions sin(kx)*sin(ky)'''

    def solution(X):
        return np.sin(k*X[:, 0])*np.sin(k*X[:, 1])[:, 1]

    def pdefun(X, U, g):
        U_1 = g.gradient(U, X)
        if U_1 is not None:
            u_x = U_1[:, 0]
            u_y = U_1[:, 0]

            U_xx = g.gradient(u_x, X)
            U_yy = g.gradient(u_y, X)

            if U_xx is not None and U_yy is not None:
                u_xx = U_xx[:, 0]
                u_yy = U_yy[:, 1]

                return u_xx + u_yy + 2*k**2*tf.sin(X[:, 0])*tf.sin(X[:, 1])

        return tf.ones_like(U) * np.inf

    n_b = 64
    l = np.linspace(0, 1, n_b)[:, None]
    X_left = np.hstack([np.zeros((n_b, 1)), l])
    X_right = np.hstack([np.ones((n_b, 1)), l])
    X_bottom = np.hstack([l, np.zeros((n_b, 1))])
    X_top = np.hstack([l, np.ones((n_b, 1))])

    X = np.vstack([X_left, X_right, X_bottom, X_top])
    U = np.zeros((n_b*4, 1))

    n_df = 5000
    X_df = np.random.uniform(low=[0, 0], high=[1, 1], size=(n_df, 2))

    return X, U, X_df, pdefun, 2


def test_burgers_slice():
    nu = 0.01 / np.pi

    def pdefun(X, U, g):

        U_1 = g.gradient(U, X)
        if U_1 is not None:

            u_x = U_1[:, 0]
            u_t = U_1[:, 1]

            U_xx = g.gradient(u_x, X)
            if U_xx is not None:
                u_xx = U_xx[:, 0]
                return u_t + U*u_x - nu*u_xx

        return tf.ones_like(U) * np.inf

    _, _, [x, t, u] = data.load.load_burgers_flat()

    t_idx = int(t.shape[0] * .75)  # Find the index for t

    X = np.empty((x.shape[0], 2))
    U = np.empty((x.shape[0], 1))
    for i, x in enumerate(x):
        X[i, 0] = x
        X[i, 1] = t[t_idx]
        U[i, 0] = u[i, t_idx]

    return X, U, X, pdefun, 2


def test_burgers_slice_nodf():
    nu = 0.01 / np.pi

    def pdefun(X, U, g):

        U_1 = g.gradient(U, X)
        if U_1 is not None:

            u_x = U_1[:, 0]
            u_t = U_1[:, 1]

            U_xx = g.gradient(u_x, X)
            if U_xx is not None:
                u_xx = U_xx[:, 0]
                return u_t + U*u_x - nu*u_xx

        return tf.ones_like(U) * np.inf

    _, _, [x, t, u] = data.load.load_burgers_flat()

    t_idx = int(t.shape[0] * .75)  # Find the index for t

    X = np.empty((x.shape[0], 2))
    U = np.empty((x.shape[0], 1))
    for i, x in enumerate(x):
        X[i, 0] = x
        X[i, 1] = t[t_idx]
        U[i, 0] = u[i, t_idx]

    return X, U, None, pdefun, 2


dispatch = {
    "shm": test_shm,
    "transport": test_transport,
    "burgers": test_burgers,
    "burgers_dense": test_burgers_dense,
    "burgers_slice": test_burgers_slice,
    "burgers_slice_nodf": test_burgers_slice_nodf,
    "pendulum": test_pendulum,
    "poisson": test_poisson,
}
