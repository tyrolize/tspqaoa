# methods for Quantum Local Search implementation

import networkx as nx
import numpy as np
from qiskit import (Aer, ClassicalRegister, QuantumCircuit, QuantumRegister,
                    execute)
from qiskit.providers.aer import AerSimulator

from tspqaoa.optimization import get_optimized_angles, run_qaoa, run_tsp_solver
from tspqaoa.qaoa import get_tsp_qaoa_circuit
from tspqaoa.utils import format_from_onehot, unformat_to_onehot


def state_split(global_state, local_nodes):
    """
    Splits the global state into groups based on deletion of local nodes.
    These are the path pieces which will not be optimized in this cycle.

    Parameters
    ----------
    global_state : List of integers (size n)
    local_state : List of integers (size k<n)

    Returns
    -------
    path_pieces : List of Lists of integers
    """
    path_pieces = []
    l = len(global_state)
    i = 0
    global_state_reordered = global_state
    for n in range(l):
        if global_state[n] in local_nodes:
            global_state_reordered = global_state[n:] + global_state[:n]
            break
    for n in range(l):
        if global_state_reordered[n] in local_nodes:
            if global_state_reordered[i:n]:
                path_pieces.append(global_state_reordered[i:n])
            i = n+1
    if i < l:
        path_pieces.append(global_state_reordered[i:l])
    return path_pieces


def qls_state(global_state, local_nodes):
    """
    Writes the qls state. This will act as init state for the next
    QLS cycle.

    Parameters
    ----------
    global_state : List of integers (size n)
    local_state : List of integers (size k<n)

    Returns
    -------
    qls_state : List of integers
    path_pieces : List of lists of integers
        needed to reinsert the path pieces
    """
    path_pieces = state_split(global_state, local_nodes)
    qls_state = global_state
    for pp in path_pieces:
        if len(pp)>=3:
            for n in pp[1:-1]:
                qls_state.remove(n)
    return qls_state, path_pieces


def path_piece_insert(qls_state, path_pieces):
    """
    Inserts the path pieces back to the qls state to create a global solution state.

    Parameters
    ----------
    qls_state : List of integers (size n)
    path_pieces : List of Lists integers

    Returns
    -------
    global_state : List of Lists of integers
        reconstructed global state
    """
    global_state = qls_state
    for pp in path_pieces:
        for i in range(len(global_state)-1):
            if global_state[i] == pp[0] and global_state[i+1] == pp[-1]:
                global_state[i:i+2] = pp
            elif global_state[i] == pp[-1] and global_state[i+1] == pp[0]:
                global_state[i:i+2] = list(reversed(pp))
        if global_state[-1] == pp[0] and global_state[0] == pp[-1]:
            global_state[-1:] = pp
            global_state.pop(0)
        elif global_state[-1] == pp[1] and global_state[0] == pp[0]:
            global_state[-1:] = list(reversed(pp))
            global_state.pop(0)
    return global_state


def invariant_neighbours(path_pieces):
    invariant_neighbours = []
    for pp in path_pieces:
        if pp[0] != pp[-1]:
            invariant_neighbours.append([pp[0],pp[-1]])
    return invariant_neighbours


def graph_rewrite(G, global_state, local_state, pen=10):
    """
    Rewrites the graph for the QLS routine.

    Parameters
    ----------
    G : networkx.Graph
        Graph to rewrite (and solve TSP on)
    global_state : List of integers (size n)
        current global state of the solution
    local_state : List of integers (size k<n)
        local state (neighbourhood) of the solution

    Returns
    -------
    G_qls : networkx.Graph
        rewritten graph with size bounded by 3n
        for n-size of the neighbourhood
    """
    G_qls = G.copy() # initialize the rewritten graph with original graph
    path_pieces = state_split(global_state, local_state)
    for pp in path_pieces:
        pp_int = [int(i) for i in pp]
        d = 0
        l = len(pp_int)
        if l > 1:
            for i in range(l-1): # compute the weight of path piece
                d += G_qls[pp_int[i]][pp_int[i+1]]['weight']
        for i in range(1,l-1): # remove the bulk nodes of the path piece
            G_qls.remove_node(pp_int[i])
        for n in G_qls.nodes: # update the weights
            if n not in pp_int:
                G_qls[n][pp_int[0]]['weight'] += d/2
                G_qls[n][pp_int[-1]]['weight'] += d/2
        G_qls[pp_int[0]][pp_int[-1]]['weight'] = 0 # set the weight between boundary nodes of pp to -pen
    return G_qls
       

def qls_main_subroutine(G, global_state, local_state, device="GPU"):
    """
    One O(k)-local update of the global state.
    This is the main QLS subroutine. k-opt.

    Parameters
    ----------
    G : networkx.Graph
        Graph to rewrite (and solve TSP on)
    global_state : List of integers (size n)
        current global state of the solution
    local_state : List of integers (size k<n)
        local state (neighbourhood) of the solution

    Returns
    -------
    new_global_state : List of integers (size n)
        updated global state of the solution
    """
    G_qls = graph_rewrite(G, global_state, local_state)

    init_state, path_pieces = qls_state(global_state, local_state) # list of int
    print("init state: ", init_state)
    print("path pieces:", path_pieces)
    i_n = invariant_neighbours(path_pieces)
    updated_qls_state = run_qaoa(G_qls, init_state, i_n=i_n, device=device)
    print("updated qls state: ", updated_qls_state)
    global_state = path_piece_insert(updated_qls_state, path_pieces)
    print("updated global state: ", global_state)
    return global_state


def benchmark_main_subroutine(G, global_state, local_state):
    """
    One O(k)-local update of the global state.
    This is the main QLS subroutine.

    Parameters
    ----------
    G : networkx.Graph
        Graph to rewrite (and solve TSP on)
    global_state : List of integers (size n)
        current global state of the solution
    local_state : List of integers (size k<n)
        local state (neighbourhood) of the solution

    Returns
    -------
    new_global_state : List of integers (size n)
        updated global state of the solution
    """
    G_qls = graph_rewrite(G, global_state, local_state)

    init_state, path_pieces = qls_state(global_state, local_state) # list of int
    print("init state: ", init_state)
    print("path pieces:", path_pieces)
    updated_qls_state = run_tsp_solver(G_qls, init_state)
    print("updated qls state: ", updated_qls_state)
    global_state = path_piece_insert(updated_qls_state, path_pieces)
    print("updated global state: ", global_state)
    return global_state