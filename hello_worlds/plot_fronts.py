import numpy as np
import matplotlib.pyplot as plt
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions

# Gera uma malha muito mais densa para o 3D (n_partitions=60 resulta em 1891 pontos)
ref_dirs_3d = get_reference_directions("das-dennis", 3, n_partitions=60)

# Instancia os problemas
prob_dtlz = get_problem("dtlz2", n_var=10, n_obj=2)
prob_wfg = get_problem("wfg2", n_var=10, n_obj=3)

# Obtém os fronts de Pareto teóricos
pf_2d = prob_dtlz.pareto_front()
pf_3d = prob_wfg.pareto_front(ref_dirs_3d)

# Configura a figura
fig = plt.figure(figsize=(14, 6))

# Plot da esquerda: Front 2D (Contínuo)
ax1 = fig.add_subplot(121)
idx = np.argsort(pf_2d[:, 0])
pf_2d_sorted = pf_2d[idx]
ax1.plot(pf_2d_sorted[:, 0], pf_2d_sorted[:, 1], color='dodgerblue', linewidth=2.5)
ax1.set_title("Fronteira de Pareto - DTLZ2 (2D)")
ax1.set_xlabel("Objetivo 1")
ax1.set_ylabel("Objetivo 2")
ax1.grid(True, linestyle='--', alpha=0.5)

# Plot da direita: Front 3D (WFG2 - Scatter com Alta Densidade)
ax2 = fig.add_subplot(122, projection='3d')

# Tamanho do ponto reduzido (s=10) para evitar que o excesso de pontos esconda os buracos
scatter = ax2.scatter(pf_3d[:, 0], pf_3d[:, 1], pf_3d[:, 2], 
                      c=pf_3d[:, 2], cmap='viridis', alpha=0.8, s=10)

ax2.set_title("Fronteira de Pareto - WFG2 (3D)")
ax2.set_xlabel("Objetivo 1")
ax2.set_ylabel("Objetivo 2")
ax2.set_zlabel("Objetivo 3")

# Ajusta levemente o ângulo de visão para favorecer a percepção das "ilhas" do front
ax2.view_init(elev=25, azim=45)

plt.tight_layout()
plt.savefig("pareto_fronts_alta_densidade.png", dpi=300, bbox_inches='tight')
plt.show()