"""Spectral clustering, conductance, the sweep cut and the Cheeger bounds.

k-means is implemented here in NumPy (k-means++ seeding, Lloyd iterations) so
the package depends on nothing beyond numpy, scipy and networkx.
"""

from collections.abc import Iterable
from typing import Any

import networkx as nx
import numpy as np

from spectral_graph.embedding import spectral_embedding
from spectral_graph.laplacian import _degree_vector, _require_undirected
from spectral_graph.spectrum import compute_spectrum


def _kmeans(
    X: np.ndarray,
    k: int,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-8,
    random_state: int = 0,
) -> np.ndarray:
    """Integer labels for the rows of X: Lloyd's algorithm with k-means++ seeding.

    Restarts `n_init` times and keeps the run of lowest inertia; deterministic
    for a fixed `random_state`.
    """
    n = X.shape[0]
    rng = np.random.default_rng(random_state)

    best_labels = np.zeros(n, dtype=int)
    best_inertia = np.inf

    for _ in range(n_init):
        centers = np.empty((k, X.shape[1]), dtype=float)
        centers[0] = X[rng.integers(n)]
        closest = np.sum((X - centers[0]) ** 2, axis=1)
        for j in range(1, k):
            total = closest.sum()
            if total <= 0:
                # Every point already coincides with a center; pick uniformly.
                centers[j] = X[rng.integers(n)]
            else:
                centers[j] = X[rng.choice(n, p=closest / total)]
            closest = np.minimum(closest, np.sum((X - centers[j]) ** 2, axis=1))

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)

            new_centers = centers.copy()
            for j in range(k):
                members = X[labels == j]
                if len(members) > 0:
                    new_centers[j] = members.mean(axis=0)
                # An empty cluster keeps its center rather than collapsing.

            shift = np.sum((new_centers - centers) ** 2)
            centers = new_centers
            if shift <= tol:
                break

        dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        inertia = float(np.min(dists, axis=1).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels

    return best_labels


def spectral_clustering(
    G: nx.Graph,
    k: int = 2,
    normalized: bool = True,
    random_state: int = 0,
    n_init: int = 10,
) -> np.ndarray:
    """Labels for k clusters, ordered as `list(G.nodes())`.

    Embeds the nodes with the k eigenvectors of the smallest eigenvalues,
    the trivial one included, and runs k-means there; `normalized` uses the
    normalized Laplacian and row-normalizes the embedding first
    (Ng-Jordan-Weiss). Raises ValueError when k is under 1 or over the node
    count.
    """
    n = G.number_of_nodes()
    if k < 1:
        raise ValueError("Number of clusters must be at least 1")
    if k > n:
        raise ValueError(f"Cannot form {k} clusters from a graph with {n} nodes")
    if k == 1:
        return np.zeros(n, dtype=int)

    X = spectral_embedding(G, dim=k, normalized=normalized, use_fiedler=False)
    if normalized:
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.where(row_norms > 1e-10, row_norms, 1.0)

    return _kmeans(X, k, n_init=n_init, random_state=random_state)


def conductance(G: nx.Graph, S: Iterable[Any]) -> float:
    """phi(S) = w(S, V\\S) / min(vol(S), vol(V\\S)), honouring edge weights.

    Infinite for an empty or whole-graph S, which is not a cut.
    """
    # This reads G directly and never builds a Laplacian, so it needs the guard
    # itself: on a DiGraph `G[u]` yields successors only and halves the cut.
    _require_undirected(G)

    S = set(S)
    if not S or len(S) == G.number_of_nodes():
        return float("inf")

    boundary = 0.0
    vol_S = 0.0
    for u in S:
        for v, data in G[u].items():
            w = data.get("weight", 1.0)
            vol_S += w
            if v not in S:
                boundary += w

    # A's row sums, the Laplacian's own degree convention: like the vol_S loop
    # above, they count a self-loop once.
    total_vol = float(_degree_vector(G).sum())
    denom = min(vol_S, total_vol - vol_S)
    if denom <= 0:
        return float("inf")
    return boundary / denom


def sweep_cut(G: nx.Graph, normalized: bool = True) -> tuple[set[Any], float]:
    """The lowest-conductance prefix of the nodes ordered by the Fiedler vector.

    The constructive half of the Cheeger inequality: the cut found satisfies
    phi <= sqrt(2 * lambda_2). With `normalized`, the normalized Laplacian's
    Fiedler vector is rescaled by D^(-1/2), the vector the bound is stated for.
    Returns `(node set, conductance)`; raises ValueError under two nodes.
    """
    n = G.number_of_nodes()
    if n < 2:
        raise ValueError("Graph must have at least 2 nodes")

    vec = spectral_embedding(G, dim=1, normalized=normalized, use_fiedler=True)[:, 0]
    nodes = list(G.nodes())
    degrees = _degree_vector(G)
    if normalized:
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = 1.0 / np.sqrt(degrees)
        scale[~np.isfinite(scale)] = 0.0
        vec = vec * scale

    total_vol = float(degrees.sum())
    in_S: set[Any] = set()
    boundary = 0.0
    vol_S = 0.0
    best_set: set[Any] = set()
    best_phi = float("inf")

    # Every prefix but the last; the whole graph is not a cut.
    for idx in np.argsort(vec)[:-1]:
        u = nodes[idx]
        for v, data in G[u].items():
            w = data.get("weight", 1.0)
            vol_S += w
            if v == u:
                # A self-loop has both ends on one side. `v in in_S` cannot see
                # that yet: u joins in_S only after this loop.
                continue
            # An edge to a node already inside stops crossing the cut.
            boundary += -w if v in in_S else w
        in_S.add(u)

        denom = min(vol_S, total_vol - vol_S)
        if denom <= 0:
            continue
        phi = boundary / denom
        if phi < best_phi:
            best_phi = phi
            best_set = set(in_S)

    return best_set, best_phi


def cheeger_bounds(G: nx.Graph) -> tuple[float, float]:
    """(lambda_2 / 2, sqrt(2 * lambda_2)): the bounds on the graph's conductance.

    lambda_2 is the normalized Laplacian's second-smallest eigenvalue.
    """
    eigenvalues = compute_spectrum(G, k=2, normalized=True, which="SM")
    if len(eigenvalues) < 2:
        raise ValueError("Graph too small to compute Cheeger bounds")

    lambda2 = max(float(eigenvalues[1]), 0.0)  # clamp solver noise around 0
    return lambda2 / 2.0, float(np.sqrt(2.0 * lambda2))
