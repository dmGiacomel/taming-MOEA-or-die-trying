import os
import json
import glob
import numpy as np
from ConfigSpace import ConfigurationSpace, Integer, Float
from smac import Scenario
from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade

# Imports do Pymoo
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.decomposition.pbi import PBI
from pymoo.termination import get_termination

# Importação da especificação experimental declarativa
from experimental_space import (
    ALGORITHMS, PROBLEMS, N_OBJECTIVES, N_VAR, 
    N_INDIVIDUALS_MAP, N_GENERATIONS_MAP, 
    SMAC_TOTAL_TRIALS, MAX_TRIALS_PER_CONFIG, N_PARALLEL_WORKERS,
    ALGORITHM_SPECS
)

def build_config_space(algorithm_name: str, seed: int = 42) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=seed)
    
    spec = ALGORITHM_SPECS.get(algorithm_name, {})
    hp_dict = spec.get("hyperparameters", {})
    
    for name, params in hp_dict.items():
        if params["type"] == "float":
            cs.add(Float(name, params["bounds"], default=params["default"]))
        elif params["type"] == "int":
            cs.add(Integer(name, params["bounds"], default=params["default"]))
            
    return cs

def is_scenario_finished(output_dir: str) -> bool:
   
    if not os.path.exists(output_dir):
        print ("aqui")
        return False
        
    json_files = glob.glob(os.path.join(output_dir, "**", "optimization.json"), recursive=True)
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if data.get("finished", False):
                    return True
        except (json.JSONDecodeError, OSError):
            continue
            
    return False

class EvolutionaryTuner:
    def __init__(self, algorithm_name: str, problem_name: str, n_obj: int, n_var: int):
        self.algorithm_name = algorithm_name
        self.problem_name = problem_name
        self.n_obj = n_obj
        self.n_var = n_var
        self.problem = get_problem(problem_name, n_var=n_var, n_obj=n_obj)
        
        self.pop_size = N_INDIVIDUALS_MAP[(n_obj, n_var)]
        max_gen = N_GENERATIONS_MAP[(n_obj, n_var)]
        self.n_evals = self.pop_size * max_gen
        
        self.cs = build_config_space(algorithm_name)

    def _get_decomposition_func(self, static_cfg, config):
        decomp_name = static_cfg.get("decomposition", "pbi").lower()
        if decomp_name == "pbi":
            theta_val = config.get("pbi_theta", 5.0)
            return PBI(theta=theta_val, normalize=True)
        else:
            raise ValueError(f"Escalarização desconhecida: {decomp_name}")

    def train(self, config, seed: int) -> float:
        static_cfg = ALGORITHM_SPECS[self.algorithm_name]["static"]
        
        crossover_op = SBX(prob=static_cfg["crossover_prob"], eta=static_cfg["crossover_eta"])
        
        mut_prob = 1.0 / self.n_var
        mutation_op = PolynomialMutation(prob=mut_prob, eta=config["mutation_eta"])

        ref_dirs = get_reference_directions("energy", self.n_obj, n_points=self.pop_size)
        
        if self.algorithm_name == "MOEAD":
            algorithm = MOEAD(
                ref_dirs=ref_dirs,
                n_neighbors=static_cfg["n_neighbors"],
                decomposition=self._get_decomposition_func(static_cfg, config),
                crossover=crossover_op,
                mutation=mutation_op
            )
        elif self.algorithm_name == "NSGA3":
            algorithm = NSGA3(
                ref_dirs=ref_dirs,
                crossover=crossover_op,
                mutation=mutation_op,
                eliminate_duplicates=True
            )
        else:
            raise ValueError(f"Algoritmo desconhecido: {self.algorithm_name}")

        termination = get_termination("n_eval", self.n_evals)

        res = minimize(
            self.problem,
            algorithm,
            termination,
            seed=seed,
            verbose=False
        )

        if res.F is not None and len(res.F) > 0:
            if "wfg" in self.problem_name.lower():
                max_teorico = np.array([2.0 * (m + 1) for m in range(self.n_obj)])
            else:
                max_teorico = np.array([1.0] * self.n_obj)
                
            F_norm = res.F / max_teorico
            ref_point_fixo = np.array([1.1] * self.n_obj)
            
            hv_value = HV(ref_point=ref_point_fixo).do(F_norm)
            if hv_value > 0:
                return float(- hv_value)
            
            dist_relativa = np.mean(np.maximum(0, (F_norm - ref_point_fixo) / ref_point_fixo))
            return float(np.log1p(dist_relativa))

        return 10.0

if __name__ == "__main__":
    for algo in ALGORITHMS:
        for prob in PROBLEMS:
            for obj in N_OBJECTIVES:
                for var in N_VAR:
                    if (obj, var) not in N_INDIVIDUALS_MAP:
                        continue
                        
                    output_dir = f"resultados_{algo.lower()}_{prob}_{obj}obj_{var}var"

                    if is_scenario_finished(output_dir):
                        print(f"[SKIP] Cenário concluído previamente: {algo} | {prob} | {obj} Objetivos | {var} Variáveis")
                        continue

                    print(f"\n[START] Sintonizando: {algo} | {prob} | {obj} Objetivos | {var} Variáveis")
                    print(f"Diretório de Saída: {output_dir}")
                    
                    tuner = EvolutionaryTuner(algo, prob, obj, var)
                    
                    scenario = Scenario(
                        tuner.cs,
                        deterministic=False, 
                        n_trials=SMAC_TOTAL_TRIALS,
                        max_budget=MAX_TRIALS_PER_CONFIG,
                        n_workers=N_PARALLEL_WORKERS,
                        output_directory=output_dir
                    )
                    smac = HyperparameterOptimizationFacade(scenario, tuner.train, overwrite=True)
                    incumbent = smac.optimize()

                    print(f"[SUCCESS] Melhor configuração obtida:")
                    print(incumbent)
                    print("-" * 50)