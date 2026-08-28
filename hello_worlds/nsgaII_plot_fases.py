import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize

# 1. Definição do problema WFG1 (2 objetivos para visualização)
problem = get_problem("dtlz4", n_var=50, n_obj=2)

# 2. Configuração do NSGA-II com 150 indivíduos
algorithm = NSGA2(pop_size=150)

# 3. Execução da otimização salvando o histórico completo
n_gen = 350
res = minimize(problem,
               algorithm,
               ('n_gen', n_gen),
               seed=42,
               save_history=True)

# 4. Obtenção do front de Pareto teórico para referência visual
pf = problem.pareto_front()

# 5. Configuração da figura com 3 subplots lado a lado
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Índices das gerações que queremos visualizar (Início, Meio, Fim)
indices_geracoes = [0, 50, n_gen - 1]

for i, gen_idx in enumerate(indices_geracoes):
    ax = axes[i]
    
    # Extrai a geração correspondente do histórico
    history_gen = res.history[gen_idx]
    F = history_gen.pop.get("F") # Valores de todos os indivíduos
    num_geracao = history_gen.n_gen
    
    # Plot do Front de Pareto Teórico
    ax.scatter(pf[:, 0], pf[:, 1], color="red", s=10, label="Front de Pareto")
    
    # Plot de TODOS os indivíduos da população atual
    ax.scatter(F[:, 0], F[:, 1], facecolor="none", edgecolor="blue", 
               label=f"População (Ger. {num_geracao})")
    
    # Formatação de cada subplot
    ax.set_title(f"Geração {num_geracao}")
    # Parâmetro labelpad adicionado para aproximar os rótulos dos eixos
    ax.set_xlabel("Objetivo 1", labelpad=2)
    ax.set_ylabel("Objetivo 2", labelpad=2)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

plt.tight_layout()
plt.show()