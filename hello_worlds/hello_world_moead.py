import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.decomposition.pbi import PBI

# ==========================================
# 1. CONFIGURAÇÃO DOS HIPERPARÂMETROS DA RUN
# ==========================================
# Definição do Problema
PROBLEM_NAME = "wfg1"
N_OBJECTIVES = 2
N_VARIABLES = 10

# Configurações de População e Parada
POP_SIZE = 150  # Deve ser compatível com as direções de referência para o n_obj
N_GENERATIONS = 1500
SEED = 42

# Hiperparâmetros do MOEAD
N_NEIGHBORS = 20
PBI_THETA = 5.0
CROSSOVER_PROB = 0.9
CROSSOVER_ETA = 20.0
MUTATION_PROB = 1.0 / N_VARIABLES
MUTATION_ETA = 20.0

# ==========================================
# 2. INICIALIZAÇÃO DOS COMPONENTES DO PYMOO
# ==========================================
# Instancia o problema
problem = get_problem(PROBLEM_NAME, n_var=N_VARIABLES, n_obj=N_OBJECTIVES)

# Gera as direções de referência (pesos) baseadas no tamanho da população
ref_dirs = get_reference_directions("energy", N_OBJECTIVES, n_points=POP_SIZE)

# Ajusta o tamanho real da população gerado pelo método das direções
actual_pop_size = len(ref_dirs)
print(f"Tamanho ajustado da população (ref_dirs): {actual_pop_size}")

# Define os operadores de variação genética
crossover_op = SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA)
mutation_op = PolynomialMutation(prob=MUTATION_PROB, eta=MUTATION_ETA)

# Define a função de escalarização (Decomposition)
decomposition_func = PBI(theta=PBI_THETA)

# Instancia o algoritmo MOEAD
algorithm = MOEAD(
    ref_dirs=ref_dirs,
    n_neighbors=N_NEIGHBORS,
    decomposition=decomposition_func,
    crossover=crossover_op,
    mutation=mutation_op,
)

# ==========================================
# 3. EXECUÇÃO DA OTIMIZAÇÃO
# ==========================================
print(f"Iniciando otimização do {PROBLEM_NAME} com MOEAD...")

res = minimize(
    problem,
    algorithm,
    termination=('n_gen', N_GENERATIONS),
    seed=SEED,
    verbose=False  # Mostra o progresso de cada geração no console
)

# ==========================================
# 4. AVALIAÇÃO DOS RESULTADOS
# ==========================================
print("\n" + "="*40)
print("EXECUÇÃO CONCLUÍDA")
print("="*40)

if res.F is not None and len(res.F) > 0:
    print(f"Número de soluções encontradas na Fronteira de Pareto: {len(res.F)}")
    
    # Cálculo do Hipervolume (usando ponto de referência ideal padrão de 1.1)
    ref_point = np.ones(N_OBJECTIVES) * 1.1
    hv_value = HV(ref_point=ref_point).do(res.F)
    
    print(f"Hipervolume calculado: {hv_value:.4f}")
    print("\nAlgumas soluções da fronteira (F):")
    print(res.F[:5])  # Exibe as 5 primeiras soluções
else:
    print("Nenhuma solução válida foi retornada.")