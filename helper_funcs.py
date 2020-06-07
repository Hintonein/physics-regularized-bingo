import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def pareto_front_to_json(pareto_front):
    pareto_front_list = []
    for member in pareto_front:
        pareto_front_list.append(
            {
                "complexity": member.get_complexity(),
                "fitness": member.fitness,
                "equation": member.__str__(),
            })

    return pareto_front_list


def log_trial(fname, problem, operators, problem_args, hyperparams, pareto_front):

    if os.path.exists(fname):
        with open(fname, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(
        {
            "problem": problem,
            "problem_args": problem_args,
            "hyperparams": hyperparams,
            "operators": operators,
            "pareto_front": pareto_front_to_json(pareto_front)
        })

    with open(fname, "w+") as f:
        json.dump(data, f)


def print_pareto_front(hall_of_fame):
    print("  FITNESS    COMPLEXITY    EQUATION")
    for member in hall_of_fame:
        fit, compl, mem = member.fitness, member.get_complexity(), member
#        print(member.get_stack_string())
        print(f"{fit}, {compl}, {mem}")


def plot_pareto_front(hall_of_fame, fname):
    fitness_vals = []
    complexity_vals = []
    for member in hall_of_fame:
        fitness_vals.append(member.fitness)
        complexity_vals.append(member.get_complexity())
    plt.figure()
    plt.step(complexity_vals, fitness_vals, 'k', where='post')
    plt.plot(complexity_vals, fitness_vals, 'or')
    plt.xlabel('Complexity')
    plt.ylabel('Fitness')
    plt.savefig(fname + ".pdf")


def plot_data_and_model(bcs, agraph, fname):
    x = np.linspace(bcs[0][0], bcs[0][1], 100)
    best_individual = agraph.get_best_individual()
    best_model_y = best_individual.evaluate_equation_at(x)

    plt.figure()
    plt.plot(bcs[0], bcs[1], 'ro', label='BCs')
    plt.plot(x, best_model_y, 'b-', label='Best Individual')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(fname + ".pdf")
