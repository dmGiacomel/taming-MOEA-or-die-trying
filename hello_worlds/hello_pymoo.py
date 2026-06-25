import numpy as np
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.visualization.fitness_landscape import FitnessLandscape
from pymoo.core.parameters import get_params, flatten, set_params, hierarchical

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.config import Config
print(Config.show_compile_hint)

problem = get_problem("rastrigin", n_var = 1000)

algorithm = GA(
    pop_size=10000,
    eliminate_duplicates=True,
)

termination = get_termination("n_gen", 1000000)

res = minimize(
    problem,
    algorithm,
    termination,
    seed = 42,
    verbose = True,
)

print(f'Best solution found: X:{res.X} \nF:{res.F} \nGEN:{res.algorithm.n_gen}')
print(res)
print(flatten(get_params(algorithm)))