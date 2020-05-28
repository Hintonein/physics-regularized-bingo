# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import time
import numpy as np
import math as m
from mpi4py import MPI
import tensorflow as tf
from test_problems import get_test

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


def agraph_similarity(ag_1, ag_2):
    return ag_1.fitness == ag_2.fitness and ag_1.get_complexity() == ag_2.get_complexity()


def solve_diffeq_gpsr(test_name, operators, hyperparams, *args):
    # evolve the population

    pop_size = hyperparams["pop_size"]
    stack_size = hyperparams["stack_size"]
    max_generations = hyperparams["max_generations"]
    fitness_threshold = hyperparams["fitness_threshold"]
    stagnation_threshold = hyperparams["stagnation_threshold"]
    check_frequency = hyperparams["check_frequency"]
    min_generations = 1

    communicator = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()

    # rank 0 generates the data to be fitted
    if rank == 0:
        X, U, X_df, error_df_fn, df_order = get_test(test_name, *args)

    # TODO: Update for MPI, need to broadcase X,U,X_df
    dummy = MPI.COMM_WORLD.bcast((X, U, X_df), root=0)

    # tell bingo which mathematical building blocks may be used
    component_generator = ComponentGenerator(X.shape[1])
    for opp in operators:
        component_generator.add_operator(opp)

    crossover = AGraphCrossover(component_generator)
    mutation = AGraphMutation(component_generator)
    agraph_generator = AGraphGenerator(stack_size, component_generator)

    # tell bingo how fitness is defined
    fitness = DifferentialRegression_TF(
        X, U, X_df, error_df_fn, df_order, metric="rmse")

    # tell bingo how to calibrate any coefficients
    local_opt_fitness = ContinuousLocalOptimization(
        fitness, algorithm='Nelder-Mead')
    evaluator = Evaluation(local_opt_fitness)

    ea = DeterministicCrowdingEA(evaluator, crossover, mutation, 0.4, 0.4)

    island = Island(ea, agraph_generator, pop_size)

    pareto_front = ParetoFront(secondary_key=lambda ag: ag.get_complexity(),
                               similarity_function=agraph_similarity)

    archipelago = ParallelArchipelago(island, hall_of_fame=pareto_front)

    print("Starting evolution...")
    # go do the evolution and send back the best equations
    optim_result = archipelago.evolve_until_convergence(max_generations, fitness_threshold,
                                                        convergence_check_frequency=check_frequency, min_generations=min_generations,
                                                        stagnation_generations=stagnation_threshold, checkpoint_base_name='checkpoint', num_checkpoints=0)

    if rank == 0:
        print(optim_result)
        print("Generation: ", archipelago.generational_age)
        print_pareto_front(pareto_front)
        return archipelago, pareto_front


def main():
    hyperparams = {
        "pop_size": 50,
        "stack_size": 10,
        "max_generations": 50,
        "fitness_threshold": 1e-6,
        "stagnation_threshold": 100,
        "check_frequency": 1,
        "min_generations": 1
    }

    operators = ["+", "-", "sin", "cos", "exp"]

    solve_diffeq_gpsr("pendulum", operators, hyperparams, 2*np.pi)


if __name__ == '__main__':
    main()
