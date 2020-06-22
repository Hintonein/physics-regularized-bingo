# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import time
import numpy as np
import math as m
from mpi4py import MPI
from test_problems import get_test
import sys
import json

from helper_funcs import *

from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.symbolic_regression.agraph.generator import AGraphGenerator
from bingo.symbolic_regression.agraph.component_generator import ComponentGenerator

# define custom fitness
from differential_regression import DifferentialRegression_TF

from bingo.evolutionary_algorithms.deterministic_crowding import DeterministicCrowdingEA
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
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
    differential_weight = hyperparams["differential_weight"]
    crossover_rate = hyperparams["crossover_rate"]
    mutation_rate = hyperparams["mutation_rate"]
    evolution_algorithm = hyperparams["evolution_algorithm"]
    min_generations = hyperparams["min_generations"]

    communicator = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()

    # rank 0 generates the data to be fitted
    X, U, X_df, error_df_fn, df_order = get_test(test_name, *args)

    # tell bingo which mathematical building blocks may be used
    component_generator = ComponentGenerator(X.shape[1])
    for opp in operators:
        component_generator.add_operator(opp)

    crossover = AGraphCrossover(component_generator)
    mutation = AGraphMutation(component_generator)
    agraph_generator = AGraphGenerator(stack_size, component_generator)

    # tell bingo how fitness is defined
    fitness = DifferentialRegression_TF(
        X, U, X_df, error_df_fn, df_order,
        metric="rmse", differential_weight=differential_weight)

    # tell bingo how to calibrate any coefficients
    local_opt_fitness = ContinuousLocalOptimization(
        fitness, algorithm='Nelder-Mead')
    evaluator = Evaluation(local_opt_fitness)

    if evolution_algorithm == "DeterministicCrowding":
        ea = DeterministicCrowdingEA(
            evaluator, crossover, mutation, crossover_rate, mutation_rate)
    elif evolution_algorithm == "AgeFitness":
        ea = AgeFitnessEA(evaluator, agraph_generator, crossover, mutation,
                          crossover_rate, mutation_rate, pop_size)

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
        return archipelago, pareto_front, optim_result
    else:
        return None, None, None


def main(experiment_params):
    log_file = experiment_params["log_file"]
    problem = experiment_params["problem"]
    hyperparams = experiment_params["hyperparams"]
    operators = experiment_params["operators"]
    problem_args = experiment_params["problem_args"]

    _, pareto_front, result = solve_diffeq_gpsr(
        problem, operators, hyperparams, *problem_args)

    if pareto_front:
        log_trial(log_file, problem, operators,
                  problem_args, hyperparams, pareto_front, result)


if __name__ == '__main__':
    experiment_file = sys.argv[1]

    with open(experiment_file) as f:
        setup_json = json.load(f)

    if type(setup_json) is list:
        if len(sys.argv) > 2:
            idx = int(sys.argv[2])
            main(setup_json[idx])
        else:
            for setup in setup_json:
                main(setup)
    else:
        main(setup_json)
