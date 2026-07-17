import numpy as np

# ALGORITHMS = ["MOEAD", "NSGA3"]
ALGORITHMS = ["MOEAD"]

PROBLEMS = ["dtlz1", 
            "dtlz2",
            "dtlz3",
            "dtlz4",
            "dtlz5",
            # "dtlz6",
            # "dtlz7",
            "wfg1",
            "wfg2",            
            "wfg3",
            "wfg4",
            "wfg5",
            "wfg6",
            # "wfg7",
            # "wfg8",
            # "wfg9"
            # tirando as três mais complicadas pro piloto
]

N_OBJECTIVES = [2,3]
N_VAR = [10,20]

# indexed by (n_objectives, n_var)
N_INDIVIDUALS_MAP = {
    (2,N_VAR[0]): 150, (2,N_VAR[1]): 250,
    (3,N_VAR[0]): 200, (3,N_VAR[1]): 300,
}

# indexed by (n_objectives, n_var)
N_GENERATIONS_MAP = {
    (2,N_VAR[0]): 1500, (2,N_VAR[1]): 1700,
    (3,N_VAR[0]): 1600, (3,N_VAR[1]): 1800,
}

SMAC_TOTAL_TRIALS = 220
MAX_TRIALS_PER_CONFIG = 3
N_PARALLEL_WORKERS = 16

ALGORITHM_SPECS = {
    "MOEAD": {
        "static": {
            "crossover_operator": "SBX",
            "mutation_operator": "PolynomialMutation",
            "decomposition": "pbi", 
            "crossover_eta": 20.0,
            "crossover_prob": 0.9, 
            "n_neighbors": 20
        },
        "hyperparameters": {
            "mutation_eta": {"type": "float", "bounds": (2.0, 40.0), "default": 20},
            "pbi_theta": {"type": "float", "bounds": (1.0, 20.0), "default": 5.0}
        }
    }
}