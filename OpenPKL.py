from bingo.evolutionary_optimizers.parallel_archipelago import load_parallel_archipelago_from_file
import sympy

def OpenPKL(file):
    archipelago = load_parallel_archipelago_from_file(file)
    hofs = archipelago.hall_of_fame
    for i,hof in enumerate(hofs):
        print('##### Equation - {} #####'.format(len(hofs)-1-i))
        exp = str(hof)
        exp = exp.replace(')(',')*(')
        print('f(x) = {}'.format(sympy.expand(exp)))
        print('Fitness = {}\n'.format(hof.fitness))

if __name__=="__main__":
    file = '/mnt/c/Users/hongs/Downloads/bingo_diffeq_pt/checkpoints/poisson_simple_13.pkl'
    OpenPKL(file)
