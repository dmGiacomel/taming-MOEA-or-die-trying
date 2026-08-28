import numpy as np
import matplotlib.pyplot as plt
from pymoo.decomposition.pbi import PBI

def plot_pbi_contours(theta1, theta2):
    # Definição do vetor de peso (direção de busca)
    weights = np.array([0.5, 0.5])

    # Malha de pontos no espaço de objetivos
    x = np.linspace(0, 1.2, 200)
    y = np.linspace(0, 1.2, 200)
    X, Y = np.meshgrid(x, y)
    F = np.column_stack([X.flatten(), Y.flatten()])

    # Instanciação dos métodos de decomposição
    pbi1 = PBI(theta=theta1)
    pbi2 = PBI(theta=theta2)

    # Cálculo dos valores PBI
    Z1 = pbi1.do(F, weights=weights).reshape(X.shape)
    Z2 = pbi2.do(F, weights=weights).reshape(X.shape)

    # Plotagem
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    levels = 12 # Quantidade de curvas de nível

    # Gráfico 1: theta = 1.0
    ax1.contour(X, Y, Z1, levels=levels, colors="black")
    ax1.plot([0, 1.2], [0, 1.2], color='red', linestyle='--', linewidth=2, label='Vetor de Peso')
    ax1.set_title(rf"Decomposição PBI ($\theta = {theta1}$)")
    ax1.set_xlim(0, 1.2)
    ax1.set_ylim(0, 1.2)
    ax1.set_xlabel(r"$f_1$")
    ax1.set_ylabel(r"$f_2$")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Gráfico 2: theta = 20.0
    ax2.contour(X, Y, Z2, levels=levels, colors="black")
    ax2.plot([0, 1.2], [0, 1.2], color='red', linestyle='--', linewidth=2, label='Vetor de Peso')
    ax2.set_title(rf"Decomposição PBI ($\theta = {theta2}$)")
    ax2.set_xlim(0, 1.2)
    ax2.set_ylim(0, 1.2)
    ax2.set_xlabel(r"$f_1$")
    ax2.set_ylabel(r"$f_2$")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

plot_pbi_contours(1.0, 20.0)