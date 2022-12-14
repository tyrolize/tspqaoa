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

from tspqaoa.qaoa import get_tsp_expectation_value_method, get_tsp_qaoa_circuit
from tspqaoa.utils import format_from_onehot, unformat_to_onehot


def get_optimized_angles(G, x0, pen, i_n=[],
                         init_state=None, translate=None, method='COBYLA', device="GPU"):
    E = get_tsp_expectation_value_method(G, pen, i_n,
                                         init_state=init_state,
                                         translate=translate,
                                         device=device)
    min_x = minimize(E, x0, method=method)
    return min_x


def run_qaoa(G, init_state, i_n, device="GPU"):
    """
    Run QAOA on the graph

    Parameters
    ----------
    G : networkx.Graph
        Graph to solve TSP on
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
    
    pen = G_workable.number_of_nodes()*10

    x0 = np.ones(4) # p is inferred from len(x0)
    x = get_optimized_angles(G_workable, x0, pen, i_n, init_state=init_state_translated, device=device)
    x = x['x']
    p=len(x)
    beta = x[0:int(p/2)]
    gamma = x[int(p/2):p]
    qc = get_tsp_qaoa_circuit(G_workable, beta, gamma, pen=5, init_state=init_state_translated, T1=1, T2=1)
    qc.measure_all()

    #plot_histogram(aersim.run(qc).result().get_counts(), figsize=(25,15))
    aersim = AerSimulator(device=device)

    counts = execute(qc, aersim).result().get_counts()

    max_states = [key for key, value in counts.items() if value == max(counts.values())] # most likely states

    # add more statistics here (output error if not a sharp peak)

    assert len(max_states) == 1

    output_state_untranslated = format_from_onehot(max_states[0])
    output_state = [x if x not in translate_dictionary
                    else translate_dictionary[x] for x in output_state_untranslated]
    return output_state


def run_tsp_solver(G, init_state):
    return approx.greedy_tsp(G)[:-1]