# various operations and conversions on string and list state representations

from functools import partial
import math
import numpy as np


def check_only_allowed_chars(test_str, allowed):
    return set(test_str) <= set(allowed)


def is_square(n):
    return math.sqrt(n).is_integer()


def onehot_city_string(n, l):
    """
    Creates a "onehot" string for one city.

    Parameters
    ----------
    n : integer to encode
    l : length of the string

    Returns
    -------
    s : onehot encoding of n
    """
    assert int(l)>=(int(n)-1)
    return '0'*int(n) + '1' + '0'*(l-int(n)-1)


def format_from_onehot(s):
    """
    Formats the measured output of QAOA circuit in the "onehot" encoding to
    a list of cities.

    Parameters
    ----------
    s : string with the measurement result
    translate : dictionary with city correspondencies (1..k) to graph notation

    Returns
    -------
    city_list : list with the cities
    """
    N = len(s)
    assert is_square(N)
    assert check_only_allowed_chars(s,'01')
    city_list = []
    for i in range(int(math.sqrt(N))):
        for j in range(int(math.sqrt(N))):
            if s[i*int(math.sqrt(N))+j] == '1':
                city_list.append(j)
                break
    return(city_list)

def unformat_to_onehot(city_list, translate=None):
    """
    Inverse of format_qaoa_output.
    Converts a list of cities to the onehot encoding.

    Parameters
    ----------
    city_list : list with the integer cities
    translate : dictionary with city correspondencies graph notation to (1..k)

    Returns
    -------
    s : string with the measurement result
    """
    s = ''
    l = len(city_list)
    if translate:
        translated_city_list = [x if x not in translate else translate[x] for x in city_list]
        for c in translated_city_list:
            s += onehot_city_string(c, l)
    else:
        for c in city_list:
            s += onehot_city_string(c, l)
    return s


def is_valid_path(s, N): # pafloxy
    if len(s) != N**2 or s.count('1') != N:
        return False
    rep_matrix = np.zeros((N,N), dtype=int)
    for index, val in enumerate(s) :
        if val== '1':
            u = index % N
            i = math.floor(index/N)
            rep_matrix[i, u] = 1
    if not np.abs(np.linalg.det(rep_matrix)) == 1:
        return False
    return True


def are_neighbours_invariant(l, i_n):
    neighbours_in_s = list(zip(l, l[1:]))
    if len(l)>2:
        neighbours_in_s.append((l[-1],l[0]))
    for nn in i_n:
        if set(nn) not in [set(i) for i in neighbours_in_s]:
            return False
    return True


def get_tsp_cost(s, G, pen, i_n=[]):
    N = G.number_of_nodes()
    assert len(s) == N**2
    cost = 0
    l = format_from_onehot(s)
    #print(l)
    for i in range(N):
        u = l[i]
        v = l[(i+1)%N]
        try: # quick and dirty fix for the case [u][v] is not an edge (else key error)
            cost += G[u][v]['weight']
        except:
            cost += pen/N
    if is_valid_path(s, N) and are_neighbours_invariant(l, i_n):
        return cost
    elif is_valid_path(s, N):
        return cost+pen/2
    else:
        return cost+pen


def compute_tsp_cost_expectation(counts, G, pen, i_n=[]):
    
    """
    Computes expectation value of cost based on measurement results
    
    Args:
        counts: dict
                key as bitstring, val as count
           
        G: networkx graph

        pen: penalty for wrong formatted paths
        
    Returns:
        avg: float
             expectation value
    """
    
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        
        obj = get_tsp_cost(bitstring, G, pen, i_n)
        avg += obj * count
        sum_count += count
        
    return avg/sum_count