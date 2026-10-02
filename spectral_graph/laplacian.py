"""Laplacian constructions: L = D - A, and L_norm = I - D^(-1/2) A D^(-1/2).

Every function takes an undirected NetworkX graph and returns a scipy sparse
CSR matrix.
"""

import networkx as nx
import numpy as np
from scipy import sparse


def _require_undirected(G: nx.Graph) -> None:
    """Refuse a directed graph, naming the conversion.

    Every construction here assumes a symmetric adjacency matrix. A directed
    graph does not fail on its own: the symmetric eigensolvers read a single
    triangle of a non-symmetric L and return a plausible number for a matrix
    nobody passed (0.865 against a true 0.267 on this project's knowledge
    graph). The conversion is the caller's modelling decision, so it is not
    made here.
    """
    if G.is_directed():
        raise ValueError(
            f"spectral_graph requires an undirected graph, got "
            f"{type(G).__name__}. Spectral theory here assumes a symmetric "
            f"adjacency matrix; a directed one silently yields eigenvalues of "
            f"a matrix you did not pass. Convert explicitly with "
            f"G.to_undirected() so the choice is yours and is visible."
        )


def _degree_vector(G: nx.Graph, A: sparse.csr_matrix | None = None) -> np.ndarray:
    """Weighted degrees, as the row sums of the adjacency matrix.

    Row sums rather than `G.degree()`, which counts a self-loop twice against
    the single w `nx.adjacency_matrix` puts on the diagonal: taking degrees
    from A is what makes `D - A` annihilate the constant vector for any weights
    and self-loops. `A` is accepted so a caller that built it already does not
    build it twice.
    """
    _require_undirected(G)
    if A is None:
        A = adjacency_matrix(G)
    return np.asarray(A.sum(axis=1), dtype=np.float64).ravel()


def adjacency_matrix(G: nx.Graph) -> sparse.csr_matrix:
    """The (weighted) adjacency matrix, as float64."""
    _require_undirected(G)
    return nx.adjacency_matrix(G).astype(np.float64)


def laplacian_matrix(G: nx.Graph) -> sparse.csr_matrix:
    """The unnormalized Laplacian L = D - A.

    Positive semi-definite, with 0 as an eigenvalue whose multiplicity is the
    number of connected components.
    """
    A = adjacency_matrix(G)
    D = sparse.diags(_degree_vector(G, A), format="csr")
    return D - A


def normalized_laplacian_matrix(G: nx.Graph) -> sparse.csr_matrix:
    """The symmetric normalized Laplacian I - D^(-1/2) A D^(-1/2).

    Its eigenvalues lie in [0, 2]. An isolated node's row is left as the
    identity's.
    """
    A = adjacency_matrix(G)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_inv_sqrt = 1.0 / np.sqrt(_degree_vector(G, A))
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sparse.diags(d_inv_sqrt, format="csr")
    identity = sparse.eye(G.number_of_nodes(), format="csr")
    return identity - D_inv_sqrt @ A @ D_inv_sqrt
