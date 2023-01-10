# QAOA circuits

import math

import networkx as nx
import numpy as np
from qiskit import (Aer, ClassicalRegister, QuantumCircuit, QuantumRegister,
                    execute)
from qiskit.compiler import transpile
from qiskit.providers.aer import AerSimulator

import tspqaoa.qaoa_cirq as qaoa_cirq
import tspqaoa.qaoa_qiskit as qaoa_qiskit
import tspqaoa.qaoa_qulacs as qaoa_qulacs
from tspqaoa.graph_utils import misra_gries_edge_coloring
from tspqaoa.utils import compute_tsp_cost_expectation


def append_zz_term(qc, q1, q2, gamma, qsdk="qiskit"):
    if qsdk=="qiskit":
        qaoa_qiskit.append_zz_term(qc, q1, q2, gamma)


def append_x_term(qc, q1, beta, qsdk="qiskit"):
    if qsdk=="qiskit":
        qaoa_qiskit.append_x_term(qc, q1, beta)


def append_zzzz_term(qc, q1, q2, q3, q4, angle, qsdk="qiskit"):
    if qsdk=="qiskit":
        qaoa_qiskit.append_zzzz_term(qc, q1, q2, q3, q4, angle)


def append_4_qubit_pauli_rotation_term(qc, q1, q2, q3, q4, beta, pauli="zzzz", qsdk="qiskit"):
    if qsdk=="qiskit":
        qaoa_qiskit.append_4_qubit_pauli_rotation_term(qc, q1, q2, q3, q4, beta, pauli=pauli)


def get_tsp_cost_operator_circuit(
    G, gamma, pen, encoding="onehot", structure="controlled z", qsdk="qiskit"):
    if qsdk=="qiskit":
        return(qaoa_qiskit.get_tsp_cost_operator_circuit(
                                                         G, gamma, pen, encoding=encoding,
                                                         structure=structure))


def get_ordering_swap_partial_mixing_circuit(
    G, i, j, u, v, beta, T, encoding="onehot", structure="pauli rotations", qsdk="qiskit"):
    if qsdk=="qiskit":
        return(qaoa_qiskit.get_ordering_swap_partial_mixing_circuit(G, i, j, u, v, beta, T,
                                                                    encoding=encoding,
                                                                    structure=structure))


def get_color_parity_ordering_swap_mixer_circuit(G, beta, T1, T2, encoding="onehot", qsdk="qiskit"):
    if qsdk=="qiskit":
        return(qaoa_qiskit.get_color_parity_ordering_swap_mixer_circuit(G, beta, T1, T2, encoding=encoding))


def get_simultaneous_ordering_swap_mixer(G, beta, T1, T2, encoding="onehot", qsdk="qiskit"):
    if qsdk=="qiskit":
        return(qaoa_qiskit.get_simultaneous_ordering_swap_mixer(G, beta, T1, T2, encoding=encoding))


def get_tsp_init_circuit(G, init_state=None, encoding="onehot", qsdk="qiskit"):
    if qsdk=="qiskit":
        return(qaoa_qiskit.get_tsp_init_circuit(G, init_state=init_state, encoding=encoding))


def get_tsp_qaoa_circuit(
    G, beta, gamma, T1=5, T2=5, pen=2,
    transpile_to_basis=True, save_state=True, encoding="onehot", qsdk="qiskit"
):
    if qsdk=="qiskit":
        return(qaoa_qiskit.get_tsp_qaoa_circuit(
                                                G, beta, gamma, T1=T1, T2=T2, pen=pen,
                                                transpile_to_basis=transpile_to_basis,
                                                save_state=transpile_to_basis, encoding=encoding))


def get_tsp_expectation_value_method(G, pen, i_n=[], device="CPU", qsdk="qiskit"):
    if qsdk=="qiskit":
        return(qaoa_qiskit.get_tsp_expectation_value_method(G, pen, i_n=i_n, device=device))