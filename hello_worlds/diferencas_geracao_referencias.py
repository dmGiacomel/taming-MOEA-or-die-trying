import matplotlib.pyplot as plt
from pymoo.util.ref_dirs import get_reference_directions

# Redução do número de partições para amostrar menos pontos e evidenciar a diferença geométrica
n_partitions = 8

# Geração dos pontos usando Das-Dennis
ref_dirs_dd_2d = get_reference_directions("das-dennis", 2, n_partitions=n_partitions)
ref_dirs_dd_3d = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)

# Contagem para manter o exato mesmo número de pontos no Riesz's Energy
n_points_2d = len(ref_dirs_dd_2d)
n_points_3d = len(ref_dirs_dd_3d)

# Geração dos pontos usando Riesz's Energy
ref_dirs_energy_2d = get_reference_directions("energy", 2, 15)
ref_dirs_energy_3d = get_reference_directions("energy", 3, 60)

# Configuração da figura
fig = plt.figure(figsize=(14, 10))

# 1. Das-Dennis 2D (Linha de Cima, Esquerda)
ax1 = fig.add_subplot(2, 2, 1)
ax1.scatter(ref_dirs_dd_2d[:, 0], ref_dirs_dd_2d[:, 1], color='blue', s=30)
ax1.set_title("Das-Dennis (2D)")
ax1.set_xlabel("f1")
ax1.set_ylabel("f2")
ax1.grid(True)
ax1.set_aspect('equal')

# 2. Das-Dennis 3D (Linha de Cima, Direita)
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.scatter(ref_dirs_dd_3d[:, 0], ref_dirs_dd_3d[:, 1], ref_dirs_dd_3d[:, 2], color='blue', s=30)
ax2.set_title("Das-Dennis (3D)")
ax2.set_xlabel("f1")
ax2.set_ylabel("f2")
ax2.set_zlabel("f3")
ax2.view_init(elev=30, azim=45)

# 3. Riesz's Energy 2D (Linha de Baixo, Esquerda)
ax3 = fig.add_subplot(2, 2, 3)
ax3.scatter(ref_dirs_energy_2d[:, 0], ref_dirs_energy_2d[:, 1], color='blue', s=30)
ax3.set_title("Riesz's Energy (2D)")
ax3.set_xlabel("f1")
ax3.set_ylabel("f2")
ax3.grid(True)
ax3.set_aspect('equal')

# 4. Riesz's Energy 3D (Linha de Baixo, Direita)
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.scatter(ref_dirs_energy_3d[:, 0], ref_dirs_energy_3d[:, 1], ref_dirs_energy_3d[:, 2], color='blue', s=30)
ax4.set_title("Riesz's Energy (3D)")
ax4.set_xlabel("f1")
ax4.set_ylabel("f2")
ax4.set_zlabel("f3")
ax4.view_init(elev=30, azim=45)

# Ajuste de layout com preenchimento extra para prevenir sobreposições de texto
plt.tight_layout(pad=3.0)
plt.show()