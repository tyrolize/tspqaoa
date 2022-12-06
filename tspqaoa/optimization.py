# optimization tools for qaoa varloop and qls

import networkx as nx
import numpy as np
from scipy.optimize import minimize
from qiskit import (Aer, ClassicalRegister, QuantumCircuit, QuantumRegister,
                    execute)
from qiskit.providers.aer import AerSimulator

from .qaoa import get_tsp_expectation_value_method, get_tsp_qaoa_circuit
from .utils import format_from_onehot, unformat_to_onehot


def get_optimized_angles(G, x0, pen, init_state=None, translate=None, method='COBYLA'):
    E = get_tsp_expectation_value_method(G, pen, init_state=init_state, translate=translate)
    min_x = minimize(E, x0, method=method)
    return min_x


def run_qaoa(G, init_state):
    """
    Run QAOA on the graph

    Parameters
    ----------
    G : networkx.Graph
        Graph to rewrite (and solve TSP on)
    init_state : list of int
        init state of G

    Returns
    -------
    output_state : List of integers (size n)
        updated state of G
    """
    G_workable = G.copy()

    translate_dictionary = {}
    key = 0
    for i in G_workable.nodes():
        translate_dictionary[key] = i
        key += 1
    untranslate_dictionary = {}
    for key, value in translate_dictionary.items():
        untranslate_dictionary[value] = key
    G_workable = nx.relabel_nodes(G_workable, untranslate_dictionary)
    # now G should have numerically numerated nodes

    init_state_translated = [x if x not in untranslate_dictionary
                             else untranslate_dictionary[x] for x in init_state]

    init_state_translated_onehot = unformat_to_onehot(init_state_translated)

    pen = G_workable.number_of_nodes()*10

    x0 = np.ones(2) # p is inferred from len(x0)
    x = get_optimized_angles(G_workable, x0, pen,
                             init_state=init_state_translated_onehot, translate=translate_dictionary)
    x = x['x']
    p=len(x)
    beta = x[0:int(p/2)]
    gamma = x[int(p/2):p]
    qc = get_tsp_qaoa_circuit(G_workable, beta, gamma, init_state=init_state_translated_onehot,
                              pen=5, T1=1, T2=1, translate=translate_dictionary)
    qc.measure_all()

    aersim = AerSimulator(device="CPU")
    counts = execute(qc, aersim).result().get_counts()

    max_states = [key for key, value in counts.items() if value == max(counts.values())] # most likely states

    # add more statistics here (output error if not a sharp peak)

    assert len(max_states) == 1

    return max_states[0] # this a string