"""
Spectrum computation for spectral graph theory.

Provides functions to compute eigenvalues and eigenvectors of Laplacian
matrices using appropriate solvers (dense vs sparse) based on graph size.
"""

from typing import Any

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import ArpackError, eigsh

from spectral_graph.laplacian import laplacian_matrix, normalized_laplacian_matrix


def smallest_eigsh(
    L: sparse.spmatrix,
    k: int,
    return_eigenvectors: bool = True,
) -> Any:
    """The k smallest eigenpairs of a Laplacian, via shift-invert.

    `eigsh(L, k, which="SM")` -- what every sparse branch here used to call --
    asks ARPACK to converge the bottom of the spectrum while iterating on `L`
    itself. Krylov convergence there is governed by the *relative* separation
    of the eigenvalues being chased, and on a graph with a bottleneck that
    separation is precisely what is tiny: `lambda_2` sits a few parts per
    million above `lambda_1 = 0`. So the mode is slowest on exactly the graphs
    a spectral analysis is for. Measured by `scripts/spectral_benchmark.py` on
    this machine, against the shift-invert call below on the same matrix
    (times vary a little run to run; the order of magnitude does not):

        path_1200        lambda_2 = 6.9e-06    ~6.3s  vs  0.002s   (~3000x)
        cycle_1200       lambda_2 = 2.7e-05    ~2.8s  vs  0.002s   (~1400x)
        barbell_1000     lambda_2 = 2.3e-05    ~8.4s  vs  0.017s   (~500x)

    And it does not reliably finish. ARPACK draws a random starting residual
    unless given one, so the slow case is also an intermittent one: on
    `path_1200`, `which="SM"` exhausts its iteration budget and raises
    `ArpackNoConvergence` in roughly 1 run in 6, after ~6s of work. That is
    the part that makes this a correctness problem and not only a cost one --
    a caller sees an exception from a graph that worked the last five times.
    Shift-invert failed 0 times in the same trial, and is the more accurate of
    the two where both converge (4.4e-09 vs 2.8e-08 relative against a `tol=0`
    reference on the barbell).

    Shift-invert factorizes `L - sigma*I` once and iterates on its inverse,
    which maps the crowded bottom of the spectrum to the well-separated top,
    so the bottleneck stops being the hard case.

    The trade is real but bounded the right way: on an expander, where
    `lambda_2 = O(1)` and `SM` already converges in a few iterations, the
    factorization is pure overhead and this is ~15x *slower* (9ms vs 135ms on
    a 1200-node 6-regular graph). Losing a hundred milliseconds on the easy
    case to win six seconds on the hard one is the trade worth making, because
    the hard case is the one that scales into a hang.

    `sigma` is a hair below zero rather than zero: 0 is an eigenvalue of every
    Laplacian, so `sigma=0` asks SuperLU to factorize a singular matrix and
    raises `RuntimeError: Factor is exactly singular`. It is scaled by the
    largest diagonal entry so the shift stays small relative to a weighted
    graph's own units.

    Falls back to the old `which="SM"` call if the factorization fails. That
    is a fall back to the less reliable path by design: it is a last resort
    for a matrix shift-invert cannot factorize at all, and leaves such a
    matrix no worse off than it was before this function existed.
    """
    scale = float(np.abs(L.diagonal()).max()) or 1.0
    try:
        result = eigsh(
            L,
            k=k,
            sigma=-1e-6 * scale,
            which="LM",
            return_eigenvectors=return_eigenvectors,
        )
    except (RuntimeError, MemoryError, ArpackError):
        result = eigsh(L, k=k, which="SM", return_eigenvectors=return_eigenvectors)

    if not return_eigenvectors:
        return np.sort(result)
    eigenvalues, eigenvectors = result
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]


def _dense_window(n: int, k: int, which: str) -> slice:
    """The k eigenpairs `which` asks for, out of all n a dense solver returns.

    The dense solvers return the whole spectrum in ascending order, so a
    caller that asked for k of them has to be handed k -- the end `which`
    names -- or the dense path answers a different question from the sparse
    one. It did: `compute_spectrum(G, k=13)` on a graph under 50 nodes came
    back with every eigenvalue the graph has, and `topics()`, which reads the
    largest gap in what it is handed, searched the whole spectrum rather than
    the low end -- proposing more clusters than `MAX_AUTO_CLUSTERS` allows on
    any corpus small enough to take this path. On a Laplacian, which is
    positive semi-definite, "LM"/"LA" are the top of the spectrum and
    "SM"/"SA" the bottom.
    """
    k = min(k, n)
    return slice(n - k, n) if which in ("LM", "LA") else slice(0, k)


def compute_spectrum(
    G: nx.Graph,
    k: int | None = None,
    normalized: bool = False,
    which: str = "SM",
) -> np.ndarray:
    """
    Compute eigenvalues of the graph Laplacian.

    Automatically chooses dense or sparse solver based on graph size.

    Parameters
    ----------
    G : networkx.Graph
        Input graph
    k : int, optional
        Number of eigenvalues to compute. If None, computes all eigenvalues.
    normalized : bool, default False
        If True, use normalized Laplacian; otherwise use unnormalized.
    which : str, default 'SM'
        Which k eigenvalues to return. Honoured on both paths -- the dense
        solver computes them all and returns the end asked for:
        - 'SM': smallest magnitude
        - 'LM': largest magnitude
        - 'SA': smallest algebraic
        - 'LA': largest algebraic

    Returns
    -------
    numpy.ndarray
        Array of eigenvalues in ascending order: all n when `k` is None,
        otherwise min(k, n) of them.

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import compute_spectrum
    >>> G = nx.path_graph(5)
    >>> eigenvalues = compute_spectrum(G)
    >>> eigenvalues.shape
    (5,)
    >>> # First eigenvalue should be 0 (or very close)
    >>> abs(eigenvalues[0]) < 1e-10
    True
    >>> compute_spectrum(G, k=2).shape
    (2,)
    """
    n = G.number_of_nodes()

    # Choose Laplacian type
    if normalized:
        L = normalized_laplacian_matrix(G)
    else:
        L = laplacian_matrix(G)

    # Use dense solver for small graphs, sparse for large
    if k is None or k >= n - 1 or n < 50:
        # Dense solver - compute all eigenvalues, then keep the k asked for.
        L_dense = L.toarray() if sparse.issparse(L) else L
        eigenvalues = np.linalg.eigvalsh(L_dense)
        if k is not None:
            eigenvalues = eigenvalues[_dense_window(n, k, which)]
    elif which == "SM":
        # Shift-invert; see `smallest_eigsh` for why the default path does not
        # hand "SM" to ARPACK directly.
        eigenvalues = smallest_eigsh(L, k=k, return_eigenvectors=False)
    else:
        # A caller that asked for the other end of the spectrum gets what it
        # asked for: shift-invert is a bottom-of-the-spectrum technique.
        eigenvalues = eigsh(L, k=k, which=which, return_eigenvectors=False)
        eigenvalues = np.sort(eigenvalues)

    return eigenvalues


def compute_eigenpairs(
    G: nx.Graph,
    k: int = 2,
    normalized: bool = False,
    which: str = "SM",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute k eigenvalue-eigenvector pairs of the graph Laplacian.

    Automatically chooses dense or sparse solver based on graph size.

    Parameters
    ----------
    G : networkx.Graph
        Input graph
    k : int, default 2
        Number of eigenpairs to compute
    normalized : bool, default False
        If True, use normalized Laplacian; otherwise use unnormalized.
    which : str, default 'SM'
        Which k eigenpairs to return. Honoured on both paths -- the dense
        solver computes them all and returns the end asked for:
        - 'SM': smallest magnitude
        - 'LM': largest magnitude
        - 'SA': smallest algebraic
        - 'LA': largest algebraic

    Returns
    -------
    tuple
        (eigenvalues, eigenvectors) where:
        - eigenvalues: 1D array of shape (k,), ascending
        - eigenvectors: 2D array of shape (n, k), column i is eigenvector for eigenvalue i

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import compute_eigenpairs
    >>> G = nx.path_graph(5)
    >>> eigenvalues, eigenvectors = compute_eigenpairs(G, k=3)
    >>> eigenvalues.shape
    (3,)
    >>> eigenvectors.shape
    (5, 3)
    """
    n = G.number_of_nodes()

    # Choose Laplacian type
    if normalized:
        L = normalized_laplacian_matrix(G)
    else:
        L = laplacian_matrix(G)

    # Use dense solver for small graphs, sparse for large
    if k >= n - 1 or n < 50:
        # Dense solver
        L_dense = L.toarray() if sparse.issparse(L) else L
        eigenvalues, eigenvectors = eigh(L_dense)
        # The k `which` asks for. This took the first k whatever `which` said,
        # so a dense "LM" returned the smallest eigenpairs.
        window = _dense_window(n, k, which)
        eigenvalues = eigenvalues[window]
        eigenvectors = eigenvectors[:, window]
    elif which == "SM":
        eigenvalues, eigenvectors = smallest_eigsh(L, k=k)
    else:
        # Sparse solver
        eigenvalues, eigenvectors = eigsh(L, k=k, which=which)
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def algebraic_connectivity(G: nx.Graph, normalized: bool = False) -> float:
    """
    Compute the algebraic connectivity (Fiedler value) of a graph.

    The algebraic connectivity is the second smallest eigenvalue of the
    Laplacian. It is positive if and only if the graph is connected.
    Larger values indicate better connectivity.

    Parameters
    ----------
    G : networkx.Graph
        Input graph
    normalized : bool, default False
        If True, use normalized Laplacian

    Returns
    -------
    float
        The algebraic connectivity (λ₂)

    Raises
    ------
    ValueError
        If the graph is disconnected (λ₂ = 0)

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import algebraic_connectivity
    >>> G = nx.path_graph(5)
    >>> ac = algebraic_connectivity(G)
    >>> f"{ac:.4f}"
    '0.3820'
    """
    eigenvalues = compute_spectrum(G, k=2, normalized=normalized, which="SM")

    if len(eigenvalues) < 2:
        raise ValueError("Graph too small to compute algebraic connectivity")

    lambda2 = float(eigenvalues[1])

    if lambda2 < 1e-10:
        raise ValueError("Graph is disconnected (algebraic connectivity = 0)")

    return lambda2


if __name__ == "__main__":
    # Run simple validation
    import networkx as nx

    print("Testing spectrum.py...")

    # Test on path graph
    G = nx.path_graph(5)

    # Test compute_spectrum
    eigenvalues = compute_spectrum(G)
    print(f"All eigenvalues: {eigenvalues}")
    assert len(eigenvalues) == 5
    assert abs(eigenvalues[0]) < 1e-10  # First eigenvalue is 0
    print("✓ compute_spectrum works correctly")

    # Test compute_eigenpairs
    evals, evecs = compute_eigenpairs(G, k=3)
    print(f"First 3 eigenvalues: {evals}")
    print(f"Eigenvector matrix shape: {evecs.shape}")
    assert evals.shape == (3,)
    assert evecs.shape == (5, 3)
    print("✓ compute_eigenpairs works correctly")

    # Test algebraic_connectivity
    ac = algebraic_connectivity(G)
    print(f"Algebraic connectivity: {ac:.4f}")
    # For path graph P_n, λ₂ = 2(1 - cos(π/n))
    expected = 2 * (1 - np.cos(np.pi / 5))
    assert abs(ac - expected) < 1e-6, f"Expected {expected}, got {ac}"
    print("✓ algebraic_connectivity matches theoretical value for path graph")

    # Cross-check with NetworkX
    ac_nx = nx.algebraic_connectivity(G)
    assert abs(ac - ac_nx) < 1e-6, f"Mismatch with NetworkX: {ac} vs {ac_nx}"
    print("✓ algebraic_connectivity matches NetworkX")

    # Test on larger graph with sparse solver
    G_large = nx.barabasi_albert_graph(100, 3, seed=42)
    eigenvalues_sparse = compute_spectrum(G_large, k=5)
    print(f"5 smallest eigenvalues of BA graph: {eigenvalues_sparse}")
    assert len(eigenvalues_sparse) == 5
    print("✓ Sparse solver works for larger graphs")

    print("\nAll spectrum.py tests passed!")
