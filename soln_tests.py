# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import time
import numpy as np
import math as m
from mpi4py import MPI
import tensorflow as tf
import data.load

from helper_funcs import *

from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.symbolic_regression.agraph.generator import AGraphGenerator
from bingo.symbolic_regression.agraph.component_generator import ComponentGenerator

# define custom fitness
from differential_regression import DifferentialRegression_TF

from bingo.evolutionary_algorithms.deterministic_crowding import DeterministicCrowdingEA
from bingo.evolutionary_optimizers.parallel_archipelago import ParallelArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island
from bingo.local_optimizers.continuous_local_opt import ContinuousLocalOptimization
from bingo.stats.pareto_front import ParetoFront

POP_SIZE = 50
STACK_SIZE = 10
MAX_GENERATIONS = 50
FITNESS_THRESHOLD = 1e-6
STAGNATION_THRESHOLD = 100
CHECK_FREQUENCY = 1
MIN_GENERATIONS = 1


def format_training_data(bcs, n):
    x_bcs = np.asarray(
        bcs[0], dtype=np.float64).reshape((-1, 1))
    u_bcs = np.asarray(bcs[1], dtype=np.float64).reshape((-1, 1))

    x_df = np.linspace(
        bcs[0][0], bcs[0][1], n).reshape((-1, 1))[1:-1]

    return x_bcs, u_bcs, x_df


def agraph_similarity(ag_1, ag_2):
    return ag_1.fitness == ag_2.fitness and ag_1.get_complexity() == ag_2.get_complexity()


def solve_diffeq_gpsr(X, U, X_df, error_df_fn, df_order=1):
    # evolve the population

    communicator = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()

    # rank 0 generates the data to be fitted
    if rank == 0:
        # these are unused values, but required by the current bingo
        # implementation.  set to arbitrary empty values
        x_discr = np.vstack([X, X_df])
        y_vals = np.empty((x_discr.shape[0], 1))
        dummy = np.hstack([x_discr, y_vals])

    # TODO: Update for MPI, need to broadcase X,U,X_df
    dummy = MPI.COMM_WORLD.bcast(dummy, root=0)

    # tell bingo which mathematical building blocks may be used
    component_generator = ComponentGenerator(X.shape[1])
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")
    component_generator.add_operator("sin")
    component_generator.add_operator("cos")
    component_generator.add_operator("exp")

    crossover = AGraphCrossover(component_generator)
    mutation = AGraphMutation(component_generator)
    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator)

    # tell bingo how fitness is defined
    fitness = DifferentialRegression_TF(
        X, U, X_df, error_df_fn, df_order, metric="rmse")

    # tell bingo how to calibrate any coefficients
    local_opt_fitness = ContinuousLocalOptimization(
        fitness, algorithm='Nelder-Mead')
    evaluator = Evaluation(local_opt_fitness)

    ea = DeterministicCrowdingEA(evaluator, crossover, mutation, 0.4, 0.4)

    island = Island(ea, agraph_generator, POP_SIZE)

    pareto_front = ParetoFront(secondary_key=lambda ag: ag.get_complexity(),
                               similarity_function=agraph_similarity)

    archipelago = ParallelArchipelago(island, hall_of_fame=pareto_front)

    print("Starting evolution...")
    # go do the evolution and send back the best equations
    optim_result = archipelago.evolve_until_convergence(MAX_GENERATIONS, FITNESS_THRESHOLD,
                                                        convergence_check_frequency=CHECK_FREQUENCY, min_generations=MIN_GENERATIONS,
                                                        stagnation_generations=STAGNATION_THRESHOLD, checkpoint_base_name='checkpoint', num_checkpoints=0)

    if rank == 0:
        print(optim_result)
        print("Generation: ", archipelago.generational_age)
        print_pareto_front(pareto_front)
        return archipelago, pareto_front


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
    bcs = ((0.0, m.pi), (0.0, 0.0))
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

    return X, U, X_df, odefun


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

    return X, U, X_df, odefun


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

    return X, U, X_df, odefun


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

    return X_boundary, U_boundary, X_df, pdefun


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

    return X, U, X_df, pdefun


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

    return X, U, X_df, pdefun


def main():

    # discretization points
    n = 21

    # # define ODE and solve with GPSR
    # print("Solving constant ODE function...")
    # odefun, bcs = test_constant(2.0)
    # agraph, pareto_front = solve_ode_gpsr(odefun, bcs, n)
    # plot_pareto_front(pareto_front, 'pareto_constant_ode')
    # #plot_data_and_model(bcs, agraph, 'soln_constant_ode')

    # # define ODE and solve with GPSR
    # print("Solving linear ODE function...")
    # odefun, bcs = test_linear(2.0, 0.0)
    # agraph, pareto_front = solve_ode_gpsr(odefun, bcs, n)
    # plot_pareto_front(pareto_front, 'pareto_linear_ode')
    # #plot_data_and_model(bcs, agraph, 'soln_linear_ode')

    # # define ODE and solve with GPSR
    # print("Solving trig. ODE function...")
    # odefun, bcs = test_trig()
    # agraph, pareto_front = solve_ode_gpsr(odefun, bcs, n)
    # plot_pareto_front(pareto_front, 'pareto_trig_ode')
    # #plot_data_and_model(bcs, agraph, 'soln_trig_ode')

    # print("Solving exponential function")
    # odefun, bcs = test_exp(1)
    # agraph, pareto_front = solve_ode_gpsr(odefun, bcs, n)
    # plot_pareto_front(pareto_front, 'pareto_trig_exp')
    # #plot_data_and_model(bcs, agraph, 'soln_trig_ode')

    # print("Solving simple harmonic motion")
    # X, U, X_df, odefun = test_shm(2*np.pi)
    # agraph, pareto_front = solve_diffeq_gpsr(X, U, X_df, odefun, 2)

    # X, U, X_df, odefun = test_shm_const(2*np.pi)
    # agraph, pareto_front = solve_diffeq_gpsr(X, U, X_df, odefun, 2)

    # print("Solving the transport equation")
    # X, U, X_df, pdefun = test_transport(1, 20, 20)
    # agraph, pareto_front = solve_diffeq_gpsr(X, U, X_df, pdefun, 1)

    print("Solving the burgers equation")
    X, U, X_df, pdefun = test_burgers()
    agraph, pareto_front = solve_diffeq_gpsr(X, U, X_df, pdefun, 2)


if __name__ == '__main__':
    main()
