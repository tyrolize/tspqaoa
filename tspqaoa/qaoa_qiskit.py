# QAOA circuits

import math

import networkx as nx
import numpy as np
from qiskit import (Aer, ClassicalRegister, QuantumCircuit, QuantumRegister,
                    execute)
from qiskit.compiler import transpile
from qiskit.providers.aer import AerSimulator

from tspqaoa.graph_utils import misra_gries_edge_coloring
from tspqaoa.optimization import get_optimized_angles
from tspqaoa.utils import (compute_tsp_cost_expectation,
                           format_from_onehot, unformat_to_onehot)


def append_zz_term(qc, q1, q2, gamma):
    qc.cx(q1, q2)
    qc.rz(2 * gamma, q2)
    qc.cx(q1, q2)


def append_x_term(qc, q1, beta):
    qc.h(q1)
    qc.rz(2 * beta, q1)
    qc.h(q1) # is this more efficient than rx?


def append_zzzz_term(qc, q1, q2, q3, q4, angle):
    qc.cx(q1,q4)
    qc.cx(q2,q4)
    qc.cx(q3,q4)
    qc.rz(2 * angle, q4)
    qc.cx(q3,q4)
    qc.cx(q2,q4)
    qc.cx(q1,q4)


def append_4_qubit_pauli_rotation_term(qc, q1, q2, q3, q4, beta, pauli="zzzz"):
    allowed_symbols = set("xyz")
    if set(pauli).issubset(allowed_symbols) and len(pauli) == 4:
        if pauli[0] == "x":
            qc.h(q1)
        elif pauli[0] == "y":
            qc.rx(-np.pi*.5,q1)
        if pauli[1] == "x":
            qc.h(q2)
        elif pauli[1] == "y":
            qc.rx(-np.pi*.5,q2)
        if pauli[2] == "x":
            qc.h(q3)
        elif pauli[2] == "y":
            qc.rx(-np.pi*.5,q3)
        if pauli[3] == "x":
            qc.h(q4)
        elif pauli[3] == "y":
            qc.rx(-np.pi*.5,q4)
        append_zzzz_term(qc, q1, q2, q3, q4, beta)
        if pauli[0] == "x":
            qc.h(q1)
        elif pauli[0] == "y":
            qc.rx(-np.pi*.5,q1)
        if pauli[1] == "x":
            qc.h(q2)
        elif pauli[1] == "y":
            qc.rx(-np.pi*.5,q2)
        if pauli[2] == "x":
            qc.h(q3)
        elif pauli[2] == "y":
            qc.rx(-np.pi*.5,q3)
        if pauli[3] == "x":
            qc.h(q4)
        elif pauli[3] == "y":
            qc.rx(-np.pi*.5,q4)
    else:
        raise ValueError("Not a valid Pauli gate or wrong locality")


def get_tsp_cost_operator_circuit(
    G, gamma, pen, encoding="onehot", structure="controlled z"):
    """
    Generates a circuit for the TSP phase unitary with optional penalty.

    Parameters
    ----------
    G : networkx.Graph
        Graph to solve TSP on
    gamma :
        QAOA parameter gamma
    pen :
        Penalty for edges with no roads
    encoding : string, default "onehot"
        Type of encoding for the city ordering
    translate :
        dictionary with city encoding (ascending numerical to problem encoding)

    Returns
    -------
    qc : qiskit.QuantumCircuit
        Quantum circuit implementing the TSP phase unitary
    """
    if encoding == "onehot" and structure == "zz rotation":
        N = G.number_of_nodes()
        if not nx.is_weighted(G):
            raise ValueError("Provided graph is not weighted")
        qc = QuantumCircuit(N**2)
        for n in range(N): # cycle over all cities in the input ordering
            for u in range(N):
                for v in range(N): #road from city v to city u
                    q1 = (n*N + u) % (N**2)
                    q2 = ((n+1)*N + v) % (N**2)
                    if G.has_edge(u, v):
                        append_zz_term(qc, q1, q2, gamma * G[u][v]["weight"])
                    else:
                        append_zz_term(qc, q1, q2, gamma * pen)
        return qc
    if encoding == "onehot" and structure == "controlled z":
        N = G.number_of_nodes()
        if not nx.is_weighted(G):
            raise ValueError("Provided graph is not weighted")
        qc = QuantumCircuit(N**2)
        for n in range(N): # cycle over all cities in the input ordering
            for u in range(N):
                for v in range(N): #road from city v to city u
                    q1 = (n*N + u) % (N**2)
                    q2 = ((n+1)*N + v) % (N**2)
                    if G.has_edge(u, v):
                        qc.crz(gamma * G[u][v]["weight"], q1, q2)
                    else:
                        qc.crz(gamma * pen, q1, q2)
        return qc


def get_ordering_swap_partial_mixing_circuit(
    G, i, j, u, v, beta, T, encoding="onehot", structure="pauli rotations"):
    """
    Generates an ordering swap partial mixer for the TSP mixing unitary.

    Parameters
    ----------
    G : networkx.Graph
        Graph to solve TSP on
    i, j :
        Positions in the ordering to be swapped
    u, v :
        Cities to be swapped
    beta :
        QAOA angle
    T :
        Number of Trotter steps
    encoding : string, default "onehot"
        Type of encoding for the city ordering

    Returns
    -------
    qc : qiskit.QuantumCircuit
        Quantum circuit implementing the TSP phase unitary
    """
    if encoding == "onehot" and structure == "pauli rotations":
        N = G.number_of_nodes()
        dt = beta/T
        qc = QuantumCircuit(N**2)
        qui = (N*i + u)
        qvj = (N*j + v)
        quj = (N*j + u)
        qvi = (N*i + v)
        for t in range(T):
            append_4_qubit_pauli_rotation_term(qc, qui, qvj, quj, qvi, dt, "xxxx")
            append_4_qubit_pauli_rotation_term(qc, qui, qvj, quj, qvi, -dt, "xxyy")
            append_4_qubit_pauli_rotation_term(qc, qui, qvj, quj, qvi, dt, "xyxy")
            append_4_qubit_pauli_rotation_term(qc, qui, qvj, quj, qvi, dt, "xyyx")
            append_4_qubit_pauli_rotation_term(qc, qui, qvj, quj, qvi, dt, "yxxy")
            append_4_qubit_pauli_rotation_term(qc, qui, qvj, quj, qvi, dt, "yxyx")
            append_4_qubit_pauli_rotation_term(qc, qui, qvj, quj, qvi, -dt, "yyxx")
            append_4_qubit_pauli_rotation_term(qc, qui, qvj, quj, qvi, dt, "yyyy")
        return qc


def get_color_parity_ordering_swap_mixer_circuit(G, beta, T1, T2, encoding="onehot"):
    if encoding == "onehot":
        N = G.number_of_nodes()
        dt = beta/T2
        qc = QuantumCircuit(N**2)
        G = misra_gries_edge_coloring(G)
        colors = nx.get_edge_attributes(G, "misra_gries_color").values()
        for c in colors:
            for t in range(T2):
                for i in range(0,N-1,2):
                    for u, v in G.edges:
                        if G[u][v]["misra_gries_color"] == c:
                            qc = qc.compose(get_ordering_swap_partial_mixing_circuit(
                                G, i, i+1, u, v, dt, T1, encoding="onehot"))                       
                for i in range(1,N-1,2):
                    for u, v in G.edges:
                        if G[u][v]["misra_gries_color"] == c:
                            qc = qc.compose(get_ordering_swap_partial_mixing_circuit(
                                G, i, i+1, u, v, dt, T1, encoding="onehot"))
                if N%2 == 1:
                    for u, v in G.edges:
                        if G[u][v]["misra_gries_color"] == c:
                            qc = qc.compose(get_ordering_swap_partial_mixing_circuit(
                                G, N-1, 0, u, v, dt, T1, encoding="onehot"))
        return qc


def get_simultaneous_ordering_swap_mixer(G, beta, T1, T2, encoding="onehot"):
    if encoding == "onehot":
        N = G.number_of_nodes()
        dt = beta/T2
        qc = QuantumCircuit(N**2)
        for t in range(T2):
            for i in range(N):
                for u, v in G.edges:
                    qc = qc.compose(get_ordering_swap_partial_mixing_circuit(
                                G, i, (i+1)%N, u, v, dt, T1, encoding="onehot"))
        return qc


def get_tsp_init_circuit(G, init_state=None, encoding="onehot"):
    """
    Generates an inti state.

    Parameters
    ----------
    G : networkx.Graph
        Graph to solve TSP on
    init_state : list of integers

    Returns
    -------
    qc : qiskit.QuantumCircuit
        Quantum circuit implementing the TSP phase unitary
    """
    if encoding == "onehot" and init_state:
        N = G.number_of_nodes()
        assert N == len(init_state)
        qc = QuantumCircuit(N**2)
        for i in range(N):
            qc.x(i*N + init_state[i])
        return qc
    elif encoding == "onehot":
        N = G.number_of_nodes()
        qc = QuantumCircuit(N**2)
        for i in range(N):
            qc.x(i*N+i)
        return qc


def get_tsp_qaoa_circuit(
    G, beta, gamma, T1=5, T2=5, pen=2,
    transpile_to_basis=True, save_state=True, encoding="onehot"
):
    if encoding == "onehot":
        assert len(beta) == len(gamma)
        p = len(beta)  # infering number of QAOA steps from the parameters passed
        N = G.number_of_nodes()
        qr = QuantumRegister(N**2)
        qc = QuantumCircuit(qr)
        # prepare the init state in onehot encoding
        qc = qc.compose(get_tsp_init_circuit(G, encoding="onehot"))
        # second, apply p alternating operators
        for i in range(p):
            qc = qc.compose(get_tsp_cost_operator_circuit(G, gamma[i], pen, encoding="onehot"))
            qc = qc.compose(get_simultaneous_ordering_swap_mixer(G, beta[i], T1, T2, encoding="onehot"))
        if transpile_to_basis:
            qc = transpile(qc, optimization_level=0, basis_gates=["u1", "u2", "u3", "cx"])
        if save_state:
            qc.save_state()
        return qc


def get_tsp_expectation_value_method(G, pen, i_n=[], device="GPU"):
    
    """
    Runs parametrized circuit
    
    Args:
        G: networkx graph
        pen: int
            penalty for wrong formatted paths
        init_state: string
            initial state in the onehot encoding
    
    Returns :
        execute_circ method
    """
    
    #backend = Aer.get_backend('qasm_simulator')
    aersim = AerSimulator(device=device)
    
    def execute_circ(angles):
        n = len(angles)
        assert n%2 == 0
        beta = angles[0:int(n/2)]
        gamma = angles[int(n/2):n]
        qc = get_tsp_qaoa_circuit(G, beta, gamma)
        qc.measure_all()
        #counts = backend.run(qc).result().get_counts()
        counts = execute(qc, aersim).result().get_counts()
        
        return compute_tsp_cost_expectation(counts, G, pen, i_n)
    
    return execute_circ


def run_qaoa(G, i_n=[], T_depth=5, device="GPU"):
    """
    Run QAOA on the graph

    Parameters
    ----------
    G : networkx.Graph
        Graph to solve TSP on

    Returns
    -------
    output_state : List of integers (size n)
        updated state of G
    """
    pen = G.number_of_nodes()*10

    x0 = np.ones(2) # p is inferred from len(x0)
    x = get_optimized_angles(G, x0, pen, i_n, device=device)
    x = x['x']
    p=len(x)
    beta = x[0:int(p/2)]
    gamma = x[int(p/2):p]
    qc = get_tsp_qaoa_circuit(G, beta, gamma, pen=5, T1=T_depth, T2=T_depth)
    qc.measure_all()

    #plot_histogram(aersim.run(qc).result().get_counts(), figsize=(25,15))
    aersim = AerSimulator(device=device)

    counts = execute(qc, aersim).result().get_counts()

    max_states = [key for key, value in counts.items() if value == max(counts.values())] # most likely states

    # add more statistics here (output error if not a sharp peak)

    assert len(max_states) == 1

    output_state_untranslated = format_from_onehot(max_states[0])
    return output_state_untranslated