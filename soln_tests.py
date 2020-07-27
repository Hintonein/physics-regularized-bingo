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
import logging

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

from bingo.evolutionary_optimizers.evolutionary_optimizer import load_evolutionary_optimizer_from_file
import argparse


def agraph_similarity(ag_1, ag_2):
    return ag_1.fitness == ag_2.fitness and ag_1.get_complexity() == ag_2.get_complexity()


def create_evolutionary_optimizer(test_name, operators, hyperparams, checkpoint_file, use_df, *args):

    pop_size = hyperparams["pop_size"]
    stack_size = hyperparams["stack_size"]
    differential_weight = hyperparams["differential_weight"]
    crossover_rate = hyperparams["crossover_rate"]
    mutation_rate = hyperparams["mutation_rate"]
    evolution_algorithm = hyperparams["evolution_algorithm"]

    rank = MPI.COMM_WORLD.Get_rank()

    # rank 0 generates the data to be fitted
    X, U, X_df, error_df_fn, df_order = get_test(test_name, *args)

    # If we want to fall back to regular regression without changing the interface
    if not use_df:
        X_df = None

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

    return ParallelArchipelago(island, hall_of_fame=pareto_front)


def main(experiment_params, checkpoint=None):
    rank = MPI.COMM_WORLD.Get_rank()

    log_file = experiment_params["log_file"]
    result_file = experiment_params["result_file"]
    checkpoint_file = experiment_params["checkpoint_file"]

    logging.basicConfig(filename=f"{log_file}_{rank}.log",
                        filemode="a", level=logging.INFO)

    if checkpoint is None:

        problem = experiment_params["problem"]
        hyperparams = experiment_params["hyperparams"]
        operators = experiment_params["operators"]
        problem_args = experiment_params["problem_args"]

        if "use_df" in experiment_params:
            use_df = experiment_params["use_df"]
        else:
            use_df = True

        optimizer = create_evolutionary_optimizer(
            problem, operators, hyperparams, checkpoint_file, use_df, *problem_args)

    else:
        optimizer = checkpoint

    max_generations = hyperparams["max_generations"]
    min_generations = hyperparams["min_generations"]
    fitness_threshold = hyperparams["fitness_threshold"]
    stagnation_threshold = hyperparams["stagnation_threshold"]
    check_frequency = hyperparams["check_frequency"]

    print("Starting evolution...")
    # go do the evolution and send back the best equations
    optim_result = optimizer.evolve_until_convergence(max_generations, fitness_threshold,
                                                      convergence_check_frequency=check_frequency, min_generations=min_generations,
                                                      stagnation_generations=stagnation_threshold, checkpoint_base_name=checkpoint_file, num_checkpoints=2)

    if rank == 0:
        pareto_front = optimizer.hall_of_fame
        print(optim_result)
        print("Generation: ", optimizer.generational_age)
        print_pareto_front(pareto_front)
        log_trial(result_file, problem, operators,
                  problem_args, hyperparams, pareto_front, optim_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", dest="experiment_file",
                        type=str, help="Experiment file to run", required=True)
    parser.add_argument("-n", required=False, dest="experiment_idx",
                        help="Index of experiment in give experiment file", type=int)
    parser.add_argument(
        "-k", help="Checkpoint file to resume from", type=str, dest="checkpoint_file")

    args = parser.parse_args()

    if args.checkpoint_file is not None:
        checkpoint = load_evolutionary_optimizer_from_file(
            args.checkpoint_file)[0]
    else:
        checkpoint = None

    with open(args.experiment_file) as f:
        setup_json = json.load(f)

    if type(setup_json) is list:
        if args.experiment_idx is None:
            for setup in setup_json:
                main(setup)  # TODO: Deal with checkpointing in the list case
        else:
            main(setup_json[args.experiment_idx], checkpoint)
    else:
        main(setup_json, checkpoint)
