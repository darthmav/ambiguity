"""Laplacian eigenvalues, with the solver chosen by graph size."""

from typing import Any

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackError, eigsh

from spectral_graph.laplacian import laplacian_matrix, normalized_laplacian_matrix

# Below this many nodes the dense solver is used: it computes the whole
# spectrum, which is cheap at this size and exact.
DENSE_SOLVER_MAX_NODES = 50


def smallest_eigsh(
    L: sparse.spmatrix,
    k: int,
    return_eigenvectors: bool = True,
) -> Any:
    """The k smallest eigenpairs of a Laplacian, via shift-invert.

    `eigsh(L, which="SM")` converges slowest exactly where a spectral analysis
    matters -- on a bottleneck, where lambda_2 sits a few parts per million
    above 0 -- and intermittently exhausts its iteration budget there
    (measured by `scripts/spectral_benchmark.py`: up to ~3000x slower, and
    `ArpackNoConvergence` in about 1 run in 6 on a 1200-node path).
    Shift-invert factorizes `L - sigma*I` once and iterates on its inverse,
    which maps the crowded bottom of the spectrum to the well-separated top.

    `sigma` sits a hair below zero, scaled by the largest diagonal entry,
    because 0 is an eigenvalue of every Laplacian and a singular factorization
    raises. A matrix that cannot be factorized at all falls back to `which="SM"`.

    Returns the eigenvalues ascending, with their eigenvectors as columns
    when `return_eigenvectors` is true.
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
    """The k of the n ascending eigenvalues `which` names.

    "LM"/"LA" are the top of a positive semi-definite spectrum, "SM"/"SA" the
    bottom; the dense path must hand back the same k the sparse one would.
    """
    k = min(k, n)
    return slice(n - k, n) if which in ("LM", "LA") else slice(0, k)


def compute_spectrum(
    G: nx.Graph,
    k: int | None = None,
    normalized: bool = False,
    which: str = "SM",
) -> np.ndarray:
    """Eigenvalues of the graph Laplacian, ascending.

    All n of them when `k` is None, otherwise the `min(k, n)` at the end of the
    spectrum `which` names ("SM"/"SA" the smallest, "LM"/"LA" the largest), on
    the dense and sparse paths alike. `normalized` selects the normalized
    Laplacian.
    """
    n = G.number_of_nodes()
    L = normalized_laplacian_matrix(G) if normalized else laplacian_matrix(G)

    if k is None or k >= n - 1 or n < DENSE_SOLVER_MAX_NODES:
        eigenvalues = np.linalg.eigvalsh(L.toarray())
        return eigenvalues if k is None else eigenvalues[_dense_window(n, k, which)]
    if which == "SM":
        return np.asarray(smallest_eigsh(L, k=k, return_eigenvectors=False))
    # Shift-invert is a bottom-of-the-spectrum technique; the other end is
    # asked of ARPACK directly.
    return np.sort(eigsh(L, k=k, which=which, return_eigenvectors=False))
