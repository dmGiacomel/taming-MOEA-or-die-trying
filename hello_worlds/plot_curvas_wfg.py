import matplotlib.pyplot as plt
import numpy as np
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions

fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
axes = axes.flatten()

plots_info = [
    ('wfg1', 'WFG1'),
    ('wfg2', 'WFG2 (Filtrado)'),
    ('wfg3', 'WFG3'),
    ('wfg4', 'WFG4 a WFG9')
]

# Força a geração de 5000 direções (pontos) para 2 objetivos
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=1000)

for i, (prob_name, title) in enumerate(plots_info):
    problem = get_problem(prob_name, n_obj=2, n_var=30)
    
    # Ao invés de n_pareto_points, enfiamos o ref_dirs goela abaixo da pymoo
    pf_bruto = problem.pareto_front(ref_dirs=ref_dirs)
    
    pf_bruto = pf_bruto[pf_bruto[:, 0].argsort()]
    
    pf_filtrado = []
    min_f2 = float('inf')
    
    for ponto in pf_bruto:
        if ponto[1] < min_f2:
            pf_filtrado.append(ponto)
            min_f2 = ponto[1]
            
    pf = np.array(pf_filtrado)
    
    ax = axes[i]
    # Tamanho do ponto reduzido para 1 (s=1) para a alta densidade ficar fluida
    ax.scatter(pf[:, 0], pf[:, 1], color='red', s=1)
    
    ax.set_title(title, pad=10, fontsize=12)
    ax.set_xlabel('$f_1$')
    ax.set_ylabel('$f_2$')
    ax.grid(True, linestyle='--', alpha=0.7)

plt.show()