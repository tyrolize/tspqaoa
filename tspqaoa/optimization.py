from .qaoa import get_tsp_expectation_value
from scipy.optimize import minimize


def get_optimized_angles(G, x0, pen, method='COBYLA'):
    E = get_tsp_expectation_value(G, pen)
    res = minimize(E, x0, method=method)
    return res