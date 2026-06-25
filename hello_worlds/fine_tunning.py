import json
import numpy as np
import optuna

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.parameters import get_params, flatten, set_params, hierarchical

from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation

n_evals = 200000
algorithm = NSGA2()
problem = get_problem("dtlz1", n_var = 60, n_obj=3)
ref_point = np.array([1.0,1.0,1.0])

def objective(trial):
    
    params = {
        "pop_size": trial.suggest_int("pop_size", 100, 500),
        "crossover": SBX(
            prob = trial.suggest_float("crossover_prob", 0.6, 1.0),
            eta = trial.suggest_float("crossover_eta", 10, 40)
        ),
        "mutation": PolynomialMutation(
            prob = trial.suggest_float("mutation_prob", 0.001, 0.2),
            eta = trial.suggest_float("mutation_eta", 5, 50)
        ),
        "eliminate_duplicates": True
    }

    print(f"\n[INICIANDO TRIAL {trial.number}]")

    hv = []
    n_runs = 5
    n_gen = n_evals // params["pop_size"]
    
    for i in range(n_runs):
        algorithm = NSGA2(**params)

        res = minimize(
            problem,
            algorithm, 
            termination = ('n_gen', n_gen),
            seed = trial.number * n_runs + i, 
            verbose = False
        )
        
        if res.F is not None and len(res.F) > 0:
            current_hv = HV(ref_point=ref_point).do(res.F)
            hv.append(current_hv)
        else:
            hv.append(0.0)
    
    return np.mean(hv)

study = optuna.create_study(
    direction="maximize", 
    sampler=optuna.samplers.TPESampler()
)

study.optimize(objective, n_trials=30, n_jobs=-1)

print("-" * 30)
print(f"Melhor valor de HV: {study.best_value}")
print(f"Melhores parâmetros: {study.best_params}")