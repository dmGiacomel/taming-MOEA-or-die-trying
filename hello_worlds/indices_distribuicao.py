import matplotlib.pyplot as plt
import numpy as np
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

def plot_sbx_pm_comparison(eta1, eta2):
    problem = Problem(n_var=1, xl=0.0, xu=1.0)
    
    # --- Configuração SBX (Simulated Binary Crossover) ---
    a_sbx = Individual(X=np.array([0.2]))
    b_sbx = Individual(X=np.array([0.8]))
    parents_sbx = [[a_sbx, b_sbx] for _ in range(100000)]

    off_sbx1 = SBX(prob=1.0, prob_var=1.0, eta=eta1).do(problem, parents_sbx)
    X_sbx1 = off_sbx1.get("X")

    off_sbx2 = SBX(prob=1.0, prob_var=1.0, eta=eta2).do(problem, parents_sbx)
    X_sbx2 = off_sbx2.get("X")

    # --- Configuração PM (Polynomial Mutation) ---
    # Correção: Criar populações completamente separadas na memória
    pop_pm1 = Population.new(X=np.full((100000, 1), 0.5))
    pop_pm2 = Population.new(X=np.full((100000, 1), 0.5))

    off_pm1 = PM(prob=1.0, prob_var=1.0, eta=eta1).do(problem, pop_pm1)
    X_pm1 = off_pm1.get("X")

    off_pm2 = PM(prob=1.0, prob_var=1.0, eta=eta2).do(problem, pop_pm2)
    X_pm2 = off_pm2.get("X")

    # --- Plotagem ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfico SBX
    ax1.hist(X_sbx1, range=(0, 1), bins=500, density=True, color="blue", alpha=0.6, label=f"eta = {eta1}")
    ax1.hist(X_sbx2, range=(0, 1), bins=500, density=True, color="red", alpha=0.6, label=f"eta = {eta2}")
    ax1.set_title("SBX (Crossover)\nPais em 0.2 e 0.8")
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("Valor da Variável")
    ax1.set_ylabel("Densidade")
    ax1.legend(loc='upper center')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Gráfico PM
    ax2.hist(X_pm1, range=(0, 1), bins=500, density=True, color="blue", alpha=0.6, label=f"eta = {eta1}")
    ax2.hist(X_pm2, range=(0, 1), bins=500, density=True, color="red", alpha=0.6, label=f"eta = {eta2}")
    ax2.set_title("PM (Mutação)\nIndivíduo original em 0.5")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Valor da Variável")
    ax2.set_ylabel("Densidade")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

plot_sbx_pm_comparison(2.0, 40.0)