import json
import os
import numpy as np
import optuna
import optunahub

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.parameters import get_params, flatten, set_params, hierarchical

from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize

from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.termination import get_termination
from pymoo.termination.collection import TerminationCollection

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation

from joblib import Parallel, delayed

n_evals = 250000
problem = get_problem("dtlz1", n_var = 60, n_obj=3)
ref_point = np.array([1.1,1.1,1.1])

module = optunahub.load_module("samplers/smac_sampler")
SMACSampler = module.SMACSampler

n_runs = 5

def run_MOEA(params, seed):
    algorithm = NSGA2(**params)
    termination = DefaultMultiObjectiveTermination(
        ftol=0.001,    
        period=40,    
        xtol=1e-8,
        n_max_evals=n_evals,
        n_skip=5
    )

    res = minimize(
        problem,
        algorithm,
        termination = termination,
        seed = seed,
        verbose = False
    )
    
    if res.F is not None and len(res.F) > 0:
        # 1. Tenta calcular o HV real
        hv_value = HV(ref_point=ref_point).do(res.F)
        
        if hv_value > 0:
            return hv_value
        
        # 2. Se HV for 0, calcula a "distância de falha"
        # Medimos o quanto os pontos ultrapassaram o ref_point
        # np.maximum(0, res.F - ref_point) captura apenas o excesso
        dist_relativa = np.mean(np.maximum(0, (res.F - ref_point) / ref_point))
        
        # Retornamos um valor negativo penalizado pelo log
        # O "-1.0" garante que qualquer falha seja pior que qualquer HV positivo
        return -1.0 - np.log1p(dist_relativa)

    # 3. Caso o algoritmo nem tenha retornado pontos (erro catastrófico)
    return -10.0

def objective(trial):
    params = {
        "pop_size": trial.suggest_int("pop_size", 100, 600),
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
    
    hv_results = Parallel(n_jobs=-1)(
        delayed(run_MOEA)(params, trial.number * n_runs + i)
        for i in range (n_runs)
    )
    
    return np.median(hv_results)

n_trials = 250
sampler = module.SMACSampler(
    {
        "pop_size": optuna.distributions.IntDistribution(100,500),
        "crossover_prob": optuna.distributions.FloatDistribution(0.6,1.0),
        "crossover_eta": optuna.distributions.FloatDistribution(10,40),
        "mutation_prob": optuna.distributions.FloatDistribution(0.001, 0.2),
        "mutation_eta": optuna.distributions.FloatDistribution(5, 50)
    }, 
    n_trials = n_trials
)

study = optuna.create_study(
    direction="maximize", 
    sampler=sampler
)

study.optimize(objective, n_trials=n_trials)

print("-" * 30)
print(f"Melhor valor de HV: {study.best_trial}")
print(f"Melhores parâmetros: {study.best_params}")

fig = optuna.visualization.plot_optimization_history(study)
fig.write_image("smac_optimization_history.png")