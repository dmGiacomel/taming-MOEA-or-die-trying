import os
# Trava de threads para garantir que cada worker do Dask use apenas 1 núcleo
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from ConfigSpace import ConfigurationSpace, Integer, Float
from smac import BlackBoxFacade, Scenario
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination.default import DefaultMultiObjectiveTermination

class NSGA2Tuner:
    def __init__(self):
        # 1. Configurações do Problema
        self.problem = get_problem("dtlz1", n_var=60, n_obj=3)
        self.n_evals = 400000 
        self.ref_point = np.array([1.1, 1.1, 1.1])
        
        # 2. ConfigSpace
        cs = ConfigurationSpace(seed=42)
        cs.add([
            Integer("pop_size", (100, 600), default=450),
            Integer("n_offsprings", (50, 600), default=450),
            Float("crossover_prob", (0.6, 1.0), default=0.9),
            Float("crossover_eta", (10.0, 50.0), default=20.0),
            Float("mutation_prob", (0.001, 0.2), default=0.02),
            Float("mutation_eta", (5.0, 50.0), default=15.0)
        ])
        self.cs = cs

    def train(self, config, seed: int = 0) -> float:
        # Algoritmo
        algorithm = NSGA2(
            pop_size=config["pop_size"],
            n_offsprings=config["n_offsprings"],
            crossover=SBX(prob=config["crossover_prob"], eta=config["crossover_eta"]),
            mutation=PolynomialMutation(prob=config["mutation_prob"], eta=config["mutation_eta"]),
            eliminate_duplicates=True
        )

        # REINTEGRADO: Seu critério original com tolerâncias
        termination = DefaultMultiObjectiveTermination(
            ftol=0.001,    
            period=40,    
            xtol=1e-8,
            n_max_evals=self.n_evals,
            n_skip=5
        )

        # Execução
        res = minimize(
            self.problem,
            algorithm,
            termination,
            seed=seed,
            verbose=False
        )

        # Lógica de Custo
        if res.F is not None and len(res.F) > 0:
            hv_value = HV(ref_point=self.ref_point).do(res.F)
            if hv_value > 0:
                return float(1.0 - hv_value)
            
            dist_relativa = np.mean(np.maximum(0, (res.F - self.ref_point) / self.ref_point))
            return float(1.0 + np.log1p(dist_relativa))

        return 10.0

if __name__ == "__main__":
    tuner = NSGA2Tuner()

    scenario = Scenario(
        tuner.cs,
        deterministic=False, 
        n_trials=250,
        n_workers=4,
        output_directory="resultados_nsga2_dtlz1_400k_numpy_ilimitada"
    )

    smac = BlackBoxFacade(scenario, tuner.train, overwrite=True)
    
    print(f"Iniciando SMAC3 Paralelo (Dask) - Budget Máximo: {tuner.n_evals}")
    incumbent = smac.optimize()

    print("-" * 30)
    print(f"Melhor Configuração: {incumbent}")