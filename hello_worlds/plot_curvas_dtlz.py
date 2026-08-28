import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pymoo.problems import get_problem

# Dimensões reduzidas para forçar a proximidade
fig = plt.figure(figsize=(12, 10))

# wspace e hspace controlam o espaço em branco entre colunas e linhas (valores menores = mais perto)
gs = fig.add_gridspec(2, 2, wspace=0.05, hspace=0.15)

ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ax2 = fig.add_subplot(gs[0, 1], projection='3d')
ax3 = fig.add_subplot(gs[1, 0], projection='3d')
ax4 = fig.add_subplot(gs[1, 1], projection='3d')

plots_info = [
    (ax1, 'dtlz1', 'DTLZ1', 30, 45),
    (ax2, 'dtlz2', 'DTLZ2 / DTLZ3 / DTLZ4', 30, 45),
    (ax3, 'dtlz5', 'DTLZ5 / DTLZ6', 25, 120), 
    (ax4, 'dtlz7', 'DTLZ7', 30, 45)
]

for ax, prob_name, title, elev, azim in plots_info:
    problem = get_problem(prob_name, n_obj=3)
    pf = problem.pareto_front()
    
    ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], color='red', s=15)
    
    # Pad do título reduzido
    ax.set_title(title, pad=5, fontsize=14)
    
    # Pad dos labels reduzido
    ax.set_xlabel('$f_1$', labelpad=5)
    ax.set_ylabel('$f_2$', labelpad=5)
    ax.set_zlabel('$f_3$', labelpad=5)
    
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(3))
    
    # Pad dos valores dos eixos reduzido
    ax.tick_params(axis='x', pad=0)
    ax.tick_params(axis='y', pad=0)
    ax.tick_params(axis='z', pad=0)
    
    # Zoom interno aumentado para preencher melhor o espaço de cada célula
    ax.set_box_aspect(aspect=None, zoom=0.95)
    
    ax.view_init(elev=elev, azim=azim)

# Ajuste extra para cortar as margens globais externas
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

plt.show()