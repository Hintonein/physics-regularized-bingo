{
  "problem": "poisson",
  "operators": ["+","-","*","/","sin","cos"],
  "problem_args": [3.141592653589793],
  
  "hyperparams": {
    "pop_size": 150,
    "stack_size": 20,
    "max_generations": 100,
    "fitness_threshold": 1e-2,
    "stagnation_threshold": 100,
    "differential_weight": 1.0,
    "check_frequency": 1,
    "min_generations": 1,
    "crossover_rate": 0.5,
    "mutation_rate": 0.5,
    "evolution_algorithm": "DeterministicCrowding"
    },
  
  "result_file": "results/poisson_p3_13.json",
  "log_file": "logs_noise/poisson_p3_13",
  "checkpoint_file": "checkpoints_noise/poisson_p3_13"
}