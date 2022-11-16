# enrichment and operations on graphs

from itertools import count as itcount

import networkx as nx


def misra_gries_edge_coloring(G):
    nx.set_edge_attributes(G, values=None, name="misra_gries_color")
    uncolored_edges = list(G.edges())
    while uncolored_edges:
        u, v = uncolored_edges[0]
        F = [v] # F is the maximal fan of u
        for n in G.neighbors(u):
            if (n!=v and
                G[n][u]["misra_gries_color"] != None and
                G[n][u]["misra_gries_color"] not in
                [G[F[-1]][i]["misra_gries_color"] for i in G.neighbors(F[-1])]):
                F.append(n)
        c = next((x) for x in itcount() # c is free on u
            if (x) not in set([G[u][i]["misra_gries_color"] for i in G.neighbors(u)]))
        d = next((x) for x in itcount() # d is free on F[k]
            if (x) not in set([G[F[-1]][i]["misra_gries_color"] for i in G.neighbors(F[-1])]))
        n = u # current node for the path construction
        color = d # current color
        visited_edges = [] # mark visited edges to stop double flipping
        while True: # invert cd path
            edge_found = 0
            for i, j in G.edges(n):
                if (G[i][j] not in visited_edges and 
                    G[i][j]["misra_gries_color"] == color):
                    if color == c:
                        G[i][j]["misra_gries_color"]=d
                        color = d
                    elif color == d:
                        G[i][j]["misra_gries_color"]=c
                        color = c
                    n = j
                    visited_edges.append(G[i][j])
                    edge_found = 1
                    break
            if edge_found == 0:
                break
        for i in range(len(F)): # find w satisfying w in F, [F[1]..w] a fan, d free on w
            w = F[i]
            if d not in set([G[w][j]["misra_gries_color"] for j in G.neighbors(w)]):
                Fw = F[0:i+1]
                for j in range(len(F)-1): #rotate the fan Fw
                    G[F[j]][u]["misra_gries_color"] = G[F[j+1]][u]["misra_gries_color"]
                G[F[-1]][u]["misra_gries_color"] = d
                break # break after the first w is found and the fan is rotated
        uncolored_edges.pop(0) # remove the colored edge
    return G