# optimization tools for qaoa varloop and qls

import numpy as np
from scipy.optimize import minimize
from qiskit import (Aer, ClassicalRegister, QuantumCircuit, QuantumRegister,
                    execute)
from qiskit.providers.aer import AerSimulator

from .qaoa import get_tsp_expectation_value_method, get_tsp_qaoa_circuit


def get_optimized_angles(G, x0, pen, init_state=None, method='COBYLA'):
    E = get_tsp_expectation_value_method(G, pen, init_state)
    min_x = minimize(E, x0, method=method)
    return min_x


def run_qaoa(G, init_state):

    pen = G.number_of_nodes()*10

    x0 = np.ones(2) # p is inferred from len(x0)
    x = get_optimized_angles(G, x0, pen, init_state=init_state)
    x = x['x']
    p=len(x)
    beta = x[0:int(p/2)]
    gamma = x[int(p/2):p]
    qc = get_tsp_qaoa_circuit(G, beta, gamma, init_state=init_state, pen=5, T1=1, T2=1)
    qc.measure_all()

    aersim = AerSimulator(device="CPU")
    counts = execute(qc, aersim).result().get_counts()

    max_states = [key for key, value in counts.items() if value == max(counts.values())] # most likely states

    # add more statistics here (output error if not a sharp peak)

    assert len(max_states) == 1

    return max_states[0]