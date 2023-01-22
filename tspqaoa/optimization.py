# optimization tools for qaoa varloop and qls

import networkx as nx
from networkx.algorithms import approximation as approx
#from networkx.algorithms.approximation import traveling_salesman_problem
import numpy as np
from scipy.optimize import minimize
from qiskit import (Aer, ClassicalRegister, QuantumCircuit, QuantumRegister,
                    execute)
from qiskit.providers.aer import AerSimulator, AerError
from qiskit.visualization import plot_histogram

from tspqaoa.qaoa_qiskit import get_tsp_expectation_value_method


def get_optimized_angles(G, x0, pen, i_n=[], method='COBYLA', device="GPU"):
    E = get_tsp_expectation_value_method(G, pen, i_n, device=device)
    min_x = minimize(E, x0, method=method)
    return min_x


def run_tsp_solver(G, init_state):
    return approx.greedy_tsp(G)[:-1]