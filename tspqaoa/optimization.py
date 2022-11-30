# optimization tools for qaoa varloop and qls

from scipy.optimize import minimize

from .qaoa import get_tsp_expectation_value_method


def get_optimized_angles(G, x0, pen, init_state=None, method='COBYLA'):
    E = get_tsp_expectation_value_method(G, pen, init_state)
    min_x = minimize(E, x0, method=method)
    return min_x