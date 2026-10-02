"""The Fiedler vector: the Laplacian eigenvector of the second-smallest eigenvalue."""

import networkx as nx
import numpy as np
from scipy.linalg import eigh

from spectral_graph.laplacian import (
    _require_undirected,
    laplacian_matrix,
    normalized_laplacian_matrix,
)
from spectral_graph.spectrum import DENSE_SOLVER_MAX_NODES, smallest_eigsh


def fiedler_vector(G: nx.Graph, normalized: bool = False) -> np.ndarray:
    """The unit-norm Fiedler vector of a connected graph, one entry per node.

    Its sign pattern is the spectral bipartition; its eigenvalue is the
    algebraic connectivity. Raises ValueError for a graph that is directed,
    disconnected or under two nodes.
    """
    # Ahead of `nx.is_connected`, whose refusal of a directed graph says
    # nothing about why or what to do instead.
    _require_undirected(G)

    n = G.number_of_nodes()
    if n < 2:
        raise ValueError("Graph must have at least 2 nodes")
    if not nx.is_connected(G):
        raise ValueError("Graph must be connected to compute Fiedler vector")

    L = normalized_laplacian_matrix(G) if normalized else laplacian_matrix(G)
    if n < DENSE_SOLVER_MAX_NODES:
        _, eigenvectors = eigh(L.toarray())
    else:
        _, eigenvectors = smallest_eigsh(L, k=2)
    return np.asarray(eigenvectors[:, 1])
