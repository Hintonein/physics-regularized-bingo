# Physics Regularized Bingo (PR-Bingo)
## Introduction
PR-GPSR is a interpretable machine learning package for the discovery of analytic solutions to differential equations. If you want to understand the theory of physic regularized technique with GPSR, you can read our [paper](https://arxiv.org/abs/2302.03175).

## Install Bingo
To install `Bingo`, simply use `pip` or `pip3`. Detail discriptions of installation are [Here](https://nightdr.github.io/bingo/installation.html).
```
pip install bingo-nasa
```

## Install PR-Bingo
`RP-Bingo` is simply installed by `git clone`, after installing `Bingo`.
```
git clone https://github.com/jhochhal/bingo_diffeq_pt
```
## Tutorials
Two numerical experiments based on boundary-value
problems were selected to verify that physics-regularized
GPSR can evolve known solutions to differential equations. The first experiment verifies the solution to a linear,
fourth-order ODE (`Beam bending problem`). The second experiment then verifies a
linear, second-order PDE (`Poisson's equation`). 

## Implement code with MPI
### Beam bending problem
First, you need to select the hyperparameters at `beam_bending.json` file in `experiment_files` folder.
Then, you can implement at `Ubuntu` or `Windows terminal`:
```
mpiexec -n <the number of cpu (int)> python soln_tests.py -e <Address of json file>/beam_bending.json
```
Where you need to fill `<>`. For example, it can be 
```
mpiexec -n 10 python soln_tests.py -e /mnt/c/Users/jake/bingo_diffeq_pt/experiment_files/beam_bending.json
```
Then, the progress of the evolution is written at `log file` in `logs folder`. The name of log file is `beam_bending_<cpu-rank>`. As the evolution is terminated, the obtained equation is written at the log file. Or you can open `pkl` file in `checkpoints folder` to check the evolution process. You need write the name of `pkl` file with address in `OpenPKL.py` and implement code
```
python OpenPKL.py
```
then, you can check the obtaine equation at the specific generation. 
