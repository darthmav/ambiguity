"""Spectral embedding: nodes placed by the Laplacian's lowest eigenvectors."""

import networkx as nx
import numpy as np
from scipy.linalg import eigh

from spectral_graph.laplacian import laplacian_matrix, normalized_laplacian_matrix
from spectral_graph.spectrum import DENSE_SOLVER_MAX_NODES, smallest_eigsh


def spectral_embedding(
    G: nx.Graph,
    dim: int = 2,
    normalized: bool = False,
    use_fiedler: bool = True,
) -> np.ndarray:
    """An (n_nodes, dim) embedding from the eigenvectors of the smallest eigenvalues.

    With `use_fiedler` the constant first eigenvector is skipped and the
    embedding starts at the Fiedler vector; without it, the first `dim`
    eigenvectors are used as they are. Raises ValueError when `dim` is under 1
    or the graph has too few nodes for it.
    """
    k = dim + (1 if use_fiedler else 0)
    n = G.number_of_nodes()
    if n < k:
        raise ValueError(f"Graph has {n} nodes, need at least {k} for {dim}D embedding")
    if dim < 1:
        raise ValueError("Embedding dimension must be at least 1")

    L = normalized_laplacian_matrix(G) if normalized else laplacian_matrix(G)
    if n < DENSE_SOLVER_MAX_NODES:
        _, eigenvectors = eigh(L.toarray())
    else:
        _, eigenvectors = smallest_eigsh(L, k=k)

    first = 1 if use_fiedler else 0
    return np.asarray(eigenvectors[:, first:first + dim])
