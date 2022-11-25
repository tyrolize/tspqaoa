import networkx as nx
import numpy as np

from .optimization import get_optimized_angles
from .qaoa import get_tsp_qaoa_circuit


def qls(G, state, state_neighbourhood):
    # take state_neighbourhood
    # update the neighbourhood by qaoa
    # stitch it back in