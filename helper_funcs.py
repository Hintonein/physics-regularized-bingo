import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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

