# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import time
import numpy as np
import math as m
from mpi4py import MPI
import tensorflow as tf

from helper_funcs import *

from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.symbolic_regression.agraph.generator import AGraphGenerator
from bingo.symbolic_regression.agraph.component_generator import ComponentGenerator

from bingo.symbolic_regression.implicit_regression import ImplicitTrainingData

# define custom fitness
from differential_regression import ImplicitRegression_TF

from bingo.evolutionary_algorithms.deterministic_crowding import DeterministicCrowdingEA
from bingo.evolutionary_optimizers.parallel_archipelago import ParallelArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island
from bingo.local_optimizers.continuous_local_opt import ContinuousLocalOptimization
from bingo.stats.pareto_front import ParetoFront

POP_SIZE = 50
STACK_SIZE = 10
MAX_GENERATIONS = 25
FITNESS_THRESHOLD = 1e-6
STAGNATION_THRESHOLD = 100
CHECK_FREQUENCY = 1
MIN_GENERATIONS = 1


def constant_ode(x, c):
    return c


def linear_ode(x, m, b):
    return m * x + b


def trig_ode(x):
    return np.cos(x)


def agraph_similarity(ag_1, ag_2):
    return ag_1.fitness == ag_2.fitness and ag_1.get_complexity() == ag_2.get_complexity()


def solve_ode_gpsr(odefun, bcs, n, persistent=False):
    # evolve the population

    communicator = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()

    # rank 0 generates the data to be fitted
    if rank == 0:
        # these are unused values, but required by the current bingo
        # implementation.  set to arbitrary empty values
        x_discr = np.linspace(bcs[0][0], bcs[0][1], n)
        y_vals = np.empty_like(x_discr)
        dummy = np.empty([len(x_discr), 2], dtype=float)
        dummy[:, 0] = x_discr
        dummy[:, 1] = y_vals

    dummy = MPI.COMM_WORLD.bcast(dummy, root=0)

    # not used
    training_data = ImplicitTrainingData(dummy)

    # tell bingo which mathematical building blocks may be used
    component_generator = ComponentGenerator(1)
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
    fitness = ImplicitRegression_TF(training_data, odefun, bcs, n, persistent)

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
            return (-u*k)

    # bcs : ((xmin, xmax), (ymin, ymax))
    bcs = ((-1, 1), (np.exp(-1*k), np.exp(1*k)))

    return odefun, bcs


def test_shm(omega):

    def odefun(x, u, g):
        u_x = g.gradient(u, x)

        if u_x is not None:
            u_xx = g.gradient(u_x, x)

            if u_xx is not None:
                return (u_xx[:, 0] + omega**2 * u)

        return (omega**2 * u)

    bcs = ((0, 0.75), (np.sin(0), np.sin(0.75*omega)))

    return odefun, bcs


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

    print("Solving simple harmonic motion")
    odefun, bcs = test_shm(2*np.pi)
    agraph, pareto_front = solve_ode_gpsr(odefun, bcs, n, True)
    plot_pareto_front(pareto_front, 'pareto_shm_ode')
    #plot_data_and_model(bcs, agraph, 'soln_trig_ode')


if __name__ == '__main__':
    main()
