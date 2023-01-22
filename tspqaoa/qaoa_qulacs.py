# QAOA circuits

import math

import networkx as nx
import numpy as np
from collections import Counter
from qulacs import QuantumCircuit, QuantumCircuitSimulator, ParametricQuantumCircuit, QuantumState

from tspqaoa.optimization import get_optimized_angles
from tspqaoa.utils import (compute_tsp_cost_expectation,
                           format_from_onehot, unformat_to_onehot)


def append_zz_term(qc, q1, q2, gamma):
    """
    Appends a ZZ term to qc

    Parameters
    ----------
    qc : qulacs QuantumCircuit
    q1 : int
        qubit 1
    q2 : int
        qubit 2
    gamma : float
        angle
    """
    qc.add_CNOT_gate(q1, q2)
    qc.add_RZ_gate(q2, 2*gamma)
    qc.add_CNOT_gate(q1, q2)


def append_x_term(qc, q1, beta):
    qc.add_H_gate(q1)
    qc.add_RZ_gate(q1, 2*beta)
    qc.add_H_gate(q1)


def append_zzzz_term(qc, q1, q2, q3, q4, angle):
    qc.add_CNOT_gate(q1,q4)
    qc.add_CNOT_gate(q2,q4)
    qc.add_CNOT_gate(q3,q4)
    qc.add_RZ_gate(q4, 2*angle)
    qc.add_CNOT_gate(q3,q4)
    qc.add_CNOT_gate(q2,q4)
    qc.add_CNOT_gate(q1,q4)


def append_4_qubit_pauli_rotation_term(qc, q1, q2, q3, q4, beta, pauli="zzzz"):
    allowed_symbols = set("xyz")
    if set(pauli).issubset(allowed_symbols) and len(pauli) == 4:
        if pauli[0] == "x":
            qc.add_H_gate(q1)
        elif pauli[0] == "y":
            qc.add_RX_gate(q1,-np.pi*.5)
        if pauli[1] == "x":
            qc.add_H_gate(q2)
        elif pauli[1] == "y":
            qc.add_RX_gate(q2,-np.pi*.5)
        if pauli[2] == "x":
            qc.add_H_gate(q3)
        elif pauli[2] == "y":
            qc.add_RX_gate(q3,-np.pi*.5)
        if pauli[3] == "x":
            qc.add_H_gate(q4)
        elif pauli[3] == "y":
            qc.add_RX_gate(q4,-np.pi*.5)
        append_zzzz_term(qc, q1, q2, q3, q4, beta)
        if pauli[0] == "x":
            qc.add_H_gate(q1)
        elif pauli[0] == "y":
            qc.add_RX_gate(q1,-np.pi*.5)
        if pauli[1] == "x":
            qc.add_H_gate(q2)
        elif pauli[1] == "y":
            qc.add_RX_gate(q2,-np.pi*.5)
        if pauli[2] == "x":
            qc.add_H_gate(q3)
        elif pauli[2] == "y":
            qc.add_RX_gate(q3,-np.pi*.5)
        if pauli[3] == "x":
            qc.add_H_gate(q4)
        elif pauli[3] == "y":
            qc.add_RX_gate(q4,-np.pi*.5)
    else:
        raise ValueError("Not a valid Pauli gate or wrong locality")


def get_tsp_cost_operator_circuit(
    G, gamma, pen, encoding="onehot", structure="controlled z"): # pafloxy
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
        qc = ParametricQuantumCircuit(N**2)
        for n in range(N): # cycle over all cities in the input ordering
            for u in range(N):
                for v in range(N): #road from city v to city u
                    q1 = (n*N + u) % (N**2)
                    q2 = ((n+1)*N + v) % (N**2)
                    if G.has_edge(u, v):
                        # append_zz_term(qc, q1, q2, gamma * G[u][v]["weight"])
                        qc.add_parametric_multi_Pauli_rotation_gate([q1,q2], [3,3],
                                                                    gamma * G[u][v]["weight"])
                    else:
                        # append_zz_term(qc, q1, q2, gamma * pen)
                        qc.add_parametric_multi_Pauli_rotation_gate([q1,q2], [3,3], gamma * pen)
        return qc


def get_ordering_swap_partial_mixing_circuit( 
    G, i, j, u, v, beta, T, encoding="onehot", structure="pauli rotations"): # pafloxy
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
        qc = ParametricQuantumCircuit(N**2)
        qui = (N*i + u)
        qvj = (N*j + v)
        quj = (N*j + u)
        qvi = (N*i + v)
        for t in range(T):
            # append_4_qubit_pauli_rotation_term(qc, qui, qvj, quj, qvi, dt, "xxxx")
            qc.add_parametric_multi_Pauli_rotation_gate([qui, qvj, quj, qvi], [1,1,1,1], dt)
            # ParametricPauliRotation([qui, qvj, quj, qvi], [1,1,1,1], dt).update_quantum_state(qstate)
            # append_4_qubit_pauli_rotation_term(qc, qui, qvj, quj, qvi, -dt, "xxyy")
            qc.add_parametric_multi_Pauli_rotation_gate([qui, qvj, quj, qvi], [1,1,2,2], -dt)
            # ParametricPauliRotation([qui, qvj, quj, qvi], [1,1,2,2], -dt).update_quantum_state(qstate)
            # append_4_qubit_pauli_rotation_term(qc, qui, qvj, quj, qvi, dt, "xyxy")
            qc.add_parametric_multi_Pauli_rotation_gate([qui, qvj, quj, qvi], [1,2,1,2], dt)
            # ParametricPauliRotation([qui, qvj, quj, qvi], [1,2,1,2], dt).update_quantum_state(qstate)
            # append_4_qubit_pauli_rotation_term(qc, qui, qvj, quj, qvi, dt, "xyyx")
            qc.add_parametric_multi_Pauli_rotation_gate([qui, qvj, quj, qvi], [1,2,2,1], dt)
            # ParametricPauliRotation([qui, qvj, quj, qvi], [1,2,2,1], dt).update_quantum_state(qstate)
            # append_4_qubit_pauli_rotation_term(qc, qui, qvj, quj, qvi, dt, "yxxy")
            qc.add_parametric_multi_Pauli_rotation_gate([qui, qvj, quj, qvi], [2,1,1,2], dt)
            # ParametricPauliRotation([qui, qvj, quj, qvi], [2,1,1,2], dt).update_quantum_state(qstate)
            # append_4_qubit_pauli_rotation_term(qc, qui, qvj, quj, qvi, dt, "yxyx")
            qc.add_parametric_multi_Pauli_rotation_gate([qui, qvj, quj, qvi], [2,1,2,1], dt)
            # ParametricPauliRotation([qui, qvj, quj, qvi], [2,1,2,1], dt).update_quantum_state(qstate)
            # append_4_qubit_pauli_rotation_term(qc, qui, qvj, quj, qvi, -dt, "yyxx")
            qc.add_parametric_multi_Pauli_rotation_gate([qui, qvj, quj, qvi], [2,2,1,1], -dt)
            # ParametricPauliRotation([qui, qvj, quj, qvi], [2,2,1,1], -dt).update_quantum_state(qstate)
            # append_4_qubit_pauli_rotation_term(qc, qui, qvj, quj, qvi, dt, "yyyy")
            qc.add_parametric_multi_Pauli_rotation_gate([qui, qvj, quj, qvi], [2,2,2,2], dt)
            # ParametricPauliRotation([qui, qvj, quj, qvi], [2,2,2,2], dt).update_quantum_state(qstate)
        return qc


def get_simultaneous_ordering_swap_mixer(G, beta, T1, T2, encoding="onehot"):
    if encoding == "onehot":
        N = G.number_of_nodes()
        dt = beta/T2
        qc = ParametricQuantumCircuit(N**2)
        for t in range(T2):
            for i in range(N):
                for u, v in G.edges:
                    qc.merge_circuit(get_ordering_swap_partial_mixing_circuit(
                                G, i, (i+1)%N, u, v, dt, T1, encoding="onehot"))
        return qc


def get_tsp_init_circuit(G: nx.Graph , init_state=None, encoding="onehot"): # pafloxy
    """
    Generates an inti state.

    Parameters
    ----------
    G : networkx.Graph
        Graph to solve TSP on
    init_state : list of integers

    Returns
    -------
    qc : ParametricQuantumCircuit
        Quantum circuit implementing the TSP phase unitary
    """
    if encoding == "onehot" and init_state:
        N = G.number_of_nodes()
        assert N == len(init_state)
        qc = ParametricQuantumCircuit(N**2)
        for i in range(N):
            qc.add_X_gate(i*N + init_state[i])
        return qc

    elif encoding == "onehot":
        N = G.number_of_nodes()
        qc = ParametricQuantumCircuit(N**2)
        for i in range(N):
            qc.add_X_gate(i*N+i)
        return qc


def get_tsp_qaoa_circuit(
    G, beta, gamma, T1=5, T2=5, pen=2,
    transpile_to_basis=True, save_state=True, encoding="onehot"
): # pafloxy
    if encoding == "onehot":
        
        assert len(beta) == len(gamma)
        p = len(beta)  # infering number of QAOA steps from the parameters passed
        N = G.number_of_nodes()
    
        
        # prepare the init state in onehot encoding
        qc = get_tsp_init_circuit(G, encoding="onehot")

        # second, apply p alternating operators
        for i in range(p):
            qc.merge_circuit(get_tsp_cost_operator_circuit(G, gamma[i], pen, encoding="onehot"))
            qc.merge_circuit(get_simultaneous_ordering_swap_mixer(G, beta[i], T1, T2, encoding="onehot"))
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
    def execute_circ(angles):
        
        n = len(angles)
        assert n%2 == 0
        beta = angles[0:int(n/2)]
        gamma = angles[int(n/2):n]
        
        qc = get_tsp_qaoa_circuit(G, beta, gamma)
        
        qstate = QuantumState( qc.get_qubit_count() )
        qc.update_quantum_state(qstate)
        
        ## sample from state ##
        num_samples = 1024
        samples = qstate.sampling(num_samples)
        counts = dict([(bin(item)[2:].zfill(len) ,counts) for item, counts in Counter(samples).items()])

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
    n = G.number_of_nodes()
    pen = n*10
    nshots = 1024

    x0 = np.ones(2) # p is inferred from len(x0)
    x = get_optimized_angles(G, x0, pen, i_n, device=device)
    x = x['x']
    p=len(x)
    beta = x[0:int(p/2)]
    gamma = x[int(p/2):p]
    qstate = QuantumState(n**2)
    tsp_circuit = get_tsp_qaoa_circuit(G, beta, gamma, pen=5, T1=T_depth, T2=T_depth)

    tsp_circuit.update_quantum_state(qstate)
    counts=dict([(int_to_binstr(item, n**2) ,counts) for item, counts in Counter(qstate.sampling(nshots)).items()])

    max_states = [key for key, value in counts.items() if value == max(counts.values())] # most likely states

    # add more statistics here (output error if not a sharp peak)

    assert len(max_states) == 1

    output_state_untranslated = format_from_onehot(max_states[0])
    return output_state_untranslated