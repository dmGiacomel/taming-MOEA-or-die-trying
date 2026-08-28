import time
import os
import json
import csv
import glob
import traceback
import numpy as np
import pandas as pd
import concurrent.futures
from pymoo.problems import get_problem
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.decomposition.pbi import PBI
from pymoo.core.callback import Callback
from pymoo.termination import get_termination

PROBLEMS = [f"dtlz{i}" for i in range(1, 6)] + [f"wfg{i}" for i in range(1, 7)]
OBJECTIVES = [2, 3]
VARIABLES = [10, 20]
SEEDS = list(range(42, 72))
THREADS = 14
OUTPUT_FILE = "resultados_estudo_piloto.csv"

def get_dimensions(M, n):
    if M == 2 and n == 10: return 150, 1500
    if M == 2 and n == 20: return 250, 1700
    if M == 3 and n == 10: return 200, 1600
    if M == 3 and n == 20: return 300, 1800
    return 150, 1500

class DynamicRandomCallback(Callback):
    def __init__(self, interval=20):
        super().__init__()
        self.interval = interval

    def __call__(self, algorithm, **kwargs):
        self.notify(algorithm)

    def notify(self, algorithm):
        if algorithm.n_gen % self.interval == 0:
            if hasattr(algorithm, "mutation"):
                algorithm.mutation.eta = float(np.random.uniform(2.0, 40.0))
            if hasattr(algorithm, "decomposition"):
                algorithm.decomposition.theta = float(np.random.uniform(1.0, 20.0))

def load_smac_parameters(base_dir="."):
    smac_params = {}
    search_pattern = os.path.join(base_dir, "resultados_moead_*")
    folder_paths = glob.glob(search_pattern)
    
    for folder_path in folder_paths:
        folder_name = os.path.basename(folder_path)
        folder_parts = folder_name.split('_')
        
        if len(folder_parts) < 5:
            continue
            
        prob = folder_parts[2]
        M_str = folder_parts[3].replace('obj', '')
        n_str = folder_parts[4].replace('var', '')
        
        if not (M_str.isdigit() and n_str.isdigit()):
            continue
            
        M = int(M_str)
        n = int(n_str)
        
        intensifier_file = glob.glob(os.path.join(folder_path, "**", "intensifier.json"), recursive=True)
        runhistory_file = glob.glob(os.path.join(folder_path, "**", "runhistory.json"), recursive=True)
        
        if not intensifier_file or not runhistory_file:
            continue
            
        try:
            with open(intensifier_file[0], 'r') as f:
                intensifier_data = json.load(f)
            
            incumbent_ids = intensifier_data.get("incumbent_ids", [])
            if not incumbent_ids:
                continue
            
            best_config_id = str(incumbent_ids[0])
            
            with open(runhistory_file[0], 'r') as f:
                runhistory_data = json.load(f)
                
            configs_dict = runhistory_data.get("configs", {})
            
            if best_config_id in configs_dict:
                best_params = configs_dict[best_config_id]
                eta_m = best_params.get("mutation_eta", 20.0)
                theta = best_params.get("pbi_theta", 5.0)
                
                config_key = f"{prob}_{M}_{n}"
                smac_params[config_key] = {"eta_m": eta_m, "theta": theta}
                
        except (json.JSONDecodeError, OSError, IndexError, ValueError, KeyError):
            pass
            
    return smac_params

def run_experiment(config):
    problem_name, M, n, seed, group, smac_params = config
    pop_size, n_gen = get_dimensions(M, n)
    
    problem = get_problem(problem_name, n_var=n, n_obj=M)
    ref_dirs = get_reference_directions("energy", M, n_points=pop_size)
    
    p_c, eta_c = 0.9, 20.0
    p_m = 1.0 / n
    n_neighbors = int(0.2 * pop_size) if group == "dynamic" else 20
    
    if group == "baseline":
        eta_m, theta = 20.0, 5.0
        callback = Callback()
    elif group == "smac3":
        config_key = f"{problem_name}_{M}_{n}"
        params = smac_params.get(config_key, {})
        eta_m = params.get("eta_m", 20.0)
        theta = params.get("theta", 5.0)
        callback = Callback()
    elif group == "dynamic":
        eta_m, theta = 20.0, 5.0 
        callback = DynamicRandomCallback(interval=20)
        
    crossover = SBX(prob=p_c, eta=eta_c)
    mutation = PolynomialMutation(prob=p_m, eta=eta_m)
    decomp = PBI(theta=theta, normalize=True)
    
    algorithm = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=n_neighbors,
        decomposition=decomp,
        crossover=crossover,
        mutation=mutation,
    )
    
    termination = get_termination("n_gen", n_gen)
    
    start_time = time.time()
    res = minimize(problem, algorithm, termination, seed=seed, callback=callback, verbose=False)
    exec_time = time.time() - start_time
    
    pf_exact = None
    if hasattr(problem, "pareto_front"):
        if callable(problem.pareto_front):
            try:
                pf_exact = problem.pareto_front(ref_dirs)
            except Exception:
                try:
                    pf_exact = problem.pareto_front()
                except Exception:
                    pass
        else:
            pf_exact = problem.pareto_front
            
    if pf_exact is None:
        pf_exact = res.F if res.F is not None else np.zeros((1, M))

    final_igd = IGDPlus(pf_exact).do(res.F) if res.F is not None else 10.0

    if res.F is not None and len(res.F) > 0:
        if "wfg" in problem_name.lower():
            max_teorico = np.array([2.0 * (m + 1) for m in range(M)])
        else:
            max_teorico = np.array([1.0] * M)
            
        F_norm = res.F / max_teorico
        final_hv = HV(ref_point=np.array([1.1] * M)).do(F_norm)
    else:
        final_hv = 0.0
    
    return {
        "problem": problem_name, "M": M, "n": n, "seed": seed, "group": group,
        "HV": final_hv, "IGD+": final_igd, "runtime": exec_time
    }

def main():
    smac_params = load_smac_parameters()
    print(f"Parâmetros do SMAC carregados para {len(smac_params)} cenários.")
    
    completed_configs = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            df_existing = pd.read_csv(OUTPUT_FILE)
            for _, row in df_existing.iterrows():
                completed_configs.add((row['problem'], int(row['M']), int(row['n']), int(row['seed']), row['group']))
        except Exception:
            pass

    configs_to_run = []
    for p in PROBLEMS:
        for M in OBJECTIVES:
            for n in VARIABLES:
                for seed in SEEDS:
                    for g in ["baseline", "smac3", "dynamic"]:
                        if (p, M, n, seed, g) not in completed_configs:
                            configs_to_run.append((p, M, n, seed, g, smac_params))
                        
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["problem", "M", "n", "seed", "group", "HV", "IGD+", "runtime"])

    print(f"Iniciando a execução de {len(configs_to_run)} tarefas pendentes em {THREADS} threads...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=THREADS) as executor:
        futures = {executor.submit(run_experiment, c): c for c in configs_to_run}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                with open(OUTPUT_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([res["problem"], res["M"], res["n"], res["seed"], res["group"], res["HV"], res["IGD+"], res["runtime"]])
            except Exception as e:
                print(f"Erro na execução de uma das tarefas:")
                traceback.print_exc()

if __name__ == "__main__":
    main()