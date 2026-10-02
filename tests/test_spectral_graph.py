"""Closed-form tests for the spectral_graph package.

These assert against exact analytic spectra rather than against whatever the
code happens to produce:

- Path graph P_n:     lambda_k = 2 - 2*cos(pi*k/n),  k = 0..n-1
- Cycle graph C_n:    lambda_k = 2 - 2*cos(2*pi*k/n)
- Complete graph K_n: spectrum {0} + {n} with multiplicity n-1
- Fiedler value of P_n: lambda_2 = 2 - 2*cos(pi/n)

plus a cross-check of every unnormalized spectrum against NetworkX's
`laplacian_spectrum`, which is independent of this package's own code path.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

# Not installed: pyproject's `pythonpath` puts the project root on the path.
from spectral_graph import (
    adjacency_matrix,
    cheeger_bounds,
    compute_spectrum,
    conductance,
    fiedler_vector,
    laplacian_matrix,
    normalized_laplacian_matrix,
    spectral_clustering,
    spectral_embedding,
    sweep_cut,
)
from spectral_graph.spectrum import smallest_eigsh

TOL = 1e-9


def path_spectrum(n: int) -> np.ndarray:
    """Analytic Laplacian spectrum of the path graph P_n."""
    return np.array([2 - 2 * np.cos(np.pi * k / n) for k in range(n)])


def cycle_spectrum(n: int) -> np.ndarray:
    """Analytic Laplacian spectrum of the cycle graph C_n."""
    return np.sort([2 - 2 * np.cos(2 * np.pi * k / n) for k in range(n)])


def lambda_2(G: nx.Graph) -> float:
    """The algebraic connectivity, read off the package's own spectrum."""
    return float(compute_spectrum(G, k=2, which="SM")[1])


# ---------------------------------------------------------------------------
# Laplacian construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 5, 9, 20])
def test_laplacian_equals_degree_minus_adjacency(n: int) -> None:
    G = nx.path_graph(n)
    L = laplacian_matrix(G).toarray()
    A = adjacency_matrix(G).toarray()
    assert np.allclose(L, np.diag(A.sum(axis=1)) - A, atol=TOL)


@pytest.mark.parametrize("n", [3, 7, 15])
def test_laplacian_annihilates_constant_vector(n: int) -> None:
    """L @ 1 = 0 exactly: the constant vector spans the null space of a
    connected graph's Laplacian."""
    G = nx.path_graph(n)
    L = laplacian_matrix(G)
    assert np.allclose(L @ np.ones(n), 0.0, atol=TOL)


def test_normalized_laplacian_eigenvalues_lie_in_zero_two() -> None:
    for G in (nx.path_graph(12), nx.cycle_graph(9), nx.complete_graph(6),
              nx.karate_club_graph()):
        eigenvalues = compute_spectrum(G, normalized=True)
        assert eigenvalues.min() > -TOL
        assert eigenvalues.max() < 2.0 + 1e-9


def test_weighted_graph_uses_weighted_degrees() -> None:
    """A weighted graph must keep L = D - A consistent; otherwise the
    normalized Laplacian stops being positive semi-definite."""
    G = nx.karate_club_graph()  # carries edge weights
    assert np.allclose(
        laplacian_matrix(G).toarray(), nx.laplacian_matrix(G).toarray(), atol=TOL
    )
    L_norm = normalized_laplacian_matrix(G).toarray()
    assert np.allclose(
        L_norm, nx.normalized_laplacian_matrix(G).toarray(), atol=TOL
    )
    assert np.linalg.eigvalsh(L_norm).min() > -TOL


# ---------------------------------------------------------------------------
# Closed-form spectra
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 5, 8, 13, 30])
def test_path_graph_closed_form_spectrum(n: int) -> None:
    """P_n has lambda_k = 2 - 2*cos(pi*k/n) for k = 0..n-1."""
    G = nx.path_graph(n)
    eigenvalues = compute_spectrum(G)
    assert eigenvalues.shape == (n,)
    assert np.allclose(eigenvalues, path_spectrum(n), atol=1e-10)


@pytest.mark.parametrize("n", [3, 6, 11, 20])
def test_cycle_graph_closed_form_spectrum(n: int) -> None:
    """C_n has lambda_k = 2 - 2*cos(2*pi*k/n)."""
    G = nx.cycle_graph(n)
    eigenvalues = compute_spectrum(G)
    assert np.allclose(eigenvalues, cycle_spectrum(n), atol=1e-10)


@pytest.mark.parametrize("n", [2, 4, 7, 12])
def test_complete_graph_closed_form_spectrum(n: int) -> None:
    """K_n has spectrum {0} together with n repeated n-1 times."""
    G = nx.complete_graph(n)
    eigenvalues = compute_spectrum(G)

    assert eigenvalues.shape == (n,)
    assert abs(eigenvalues[0]) < 1e-10
    assert np.allclose(eigenvalues[1:], float(n), atol=1e-10)
    # Multiplicity of n is exactly n-1.
    assert int(np.sum(np.abs(eigenvalues - n) < 1e-10)) == n - 1


def test_disconnected_graph_zero_multiplicity_counts_components() -> None:
    """dim ker(L) = number of connected components."""
    G = nx.disjoint_union_all([nx.path_graph(4), nx.cycle_graph(5), nx.complete_graph(3)])
    eigenvalues = compute_spectrum(G)
    zeros = int(np.sum(np.abs(eigenvalues) < 1e-8))
    assert zeros == nx.number_connected_components(G) == 3


@pytest.mark.parametrize(
    "G",
    [
        nx.path_graph(9),
        nx.cycle_graph(9),
        nx.complete_graph(9),
        nx.star_graph(8),
        nx.barbell_graph(5, 2),
        nx.karate_club_graph(),
    ],
    ids=["path", "cycle", "complete", "star", "barbell", "karate"],
)
def test_spectrum_matches_networkx_laplacian_spectrum(G: nx.Graph) -> None:
    """Independent cross-check against NetworkX's own solver."""
    ours = compute_spectrum(G)
    theirs = np.sort(nx.laplacian_spectrum(G))
    assert np.allclose(ours, theirs, atol=1e-8)


def test_star_graph_closed_form_spectrum() -> None:
    """K_{1,n-1} has spectrum {0, 1 (multiplicity n-2), n}."""
    n = 8
    G = nx.star_graph(n - 1)
    eigenvalues = compute_spectrum(G)
    assert abs(eigenvalues[0]) < 1e-10
    assert np.allclose(eigenvalues[1:-1], 1.0, atol=1e-10)
    assert abs(eigenvalues[-1] - n) < 1e-10


# ---------------------------------------------------------------------------
# Algebraic connectivity and the Fiedler vector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 5, 10, 25])
def test_path_graph_fiedler_value_closed_form(n: int) -> None:
    """The Fiedler value of P_n is 2 - 2*cos(pi/n)."""
    G = nx.path_graph(n)
    expected = 2 - 2 * np.cos(np.pi / n)
    assert abs(lambda_2(G) - expected) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("n", [3, 6, 10])
def test_complete_graph_fiedler_value_is_n(n: int) -> None:
    G = nx.complete_graph(n)
    assert lambda_2(G) == pytest.approx(float(n), abs=1e-10)


def test_algebraic_connectivity_matches_networkx() -> None:
    for G in (nx.path_graph(11), nx.cycle_graph(11), nx.barbell_graph(5, 1)):
        assert lambda_2(G) == pytest.approx(nx.algebraic_connectivity(G), abs=1e-7)


def test_fiedler_vector_is_an_eigenvector_of_lambda_two() -> None:
    """L v = lambda_2 v, and v is orthogonal to the constant vector."""
    G = nx.path_graph(9)
    v = fiedler_vector(G)
    L = laplacian_matrix(G)
    lambda2 = lambda_2(G)

    assert np.allclose(L @ v, lambda2 * v, atol=1e-9)
    assert abs(float(np.sum(v))) < 1e-9
    assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-9)


def test_fiedler_vector_of_path_is_monotone() -> None:
    """On P_n the Fiedler vector is v_i = cos(pi*(i + 1/2)/n), monotone in i,
    so the sign cut splits the path in half."""
    n = 10
    G = nx.path_graph(n)
    v = fiedler_vector(G)
    if v[0] < 0:
        v = -v
    assert np.all(np.diff(v) < 0), f"not monotone: {v}"

    expected = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    expected /= np.linalg.norm(expected)
    assert np.allclose(v, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def test_spectral_embedding_shape_and_orthogonality() -> None:
    G = nx.path_graph(12)
    X = spectral_embedding(G, dim=3)
    assert X.shape == (12, 3)
    # Eigenvectors of a symmetric matrix are mutually orthonormal.
    assert np.allclose(X.T @ X, np.eye(3), atol=1e-9)
    # Each embedding column is orthogonal to the constant vector.
    assert np.allclose(X.sum(axis=0), 0.0, atol=1e-9)


def test_spectral_embedding_rejects_oversized_dimension() -> None:
    G = nx.path_graph(3)
    with pytest.raises(ValueError):
        spectral_embedding(G, dim=5)


# ---------------------------------------------------------------------------
# Clustering, conductance, Cheeger
# ---------------------------------------------------------------------------


def test_spectral_clustering_separates_two_cliques() -> None:
    G = nx.disjoint_union(nx.complete_graph(6), nx.complete_graph(6))
    G.add_edge(0, 6)
    labels = spectral_clustering(G, k=2)

    assert len(set(labels[:6])) == 1
    assert len(set(labels[6:])) == 1
    assert labels[0] != labels[6]


def test_spectral_clustering_is_reproducible() -> None:
    G = nx.barbell_graph(6, 0)
    a = spectral_clustering(G, k=2, random_state=3)
    b = spectral_clustering(G, k=2, random_state=3)
    assert np.array_equal(a, b)


def test_conductance_closed_form_on_path() -> None:
    """Cutting P_4 in the middle: one crossing edge, volume 3 on each side."""
    G = nx.path_graph(4)
    assert conductance(G, {0, 1}) == pytest.approx(1 / 3, abs=1e-12)
    assert conductance(G, {2, 3}) == pytest.approx(1 / 3, abs=1e-12)
    assert conductance(G, set()) == float("inf")


def test_sweep_cut_finds_the_bridge_of_a_barbell() -> None:
    """Barbell(5,0): the sweep cut is a whole bell, conductance 1/21."""
    G = nx.barbell_graph(5, 0)
    S, phi = sweep_cut(G)
    assert sorted(S) in ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])
    assert phi == pytest.approx(1 / 21, abs=1e-12)


@pytest.mark.parametrize(
    "G",
    [
        nx.path_graph(12),
        nx.cycle_graph(12),
        nx.barbell_graph(6, 0),
        nx.karate_club_graph(),
    ],
    ids=["path", "cycle", "barbell", "karate"],
)
def test_cheeger_inequality_brackets_the_sweep_cut(G: nx.Graph) -> None:
    """lambda_2/2 <= phi(sweep cut) <= sqrt(2*lambda_2)."""
    lower, upper = cheeger_bounds(G)
    _, phi = sweep_cut(G)
    assert lower - 1e-9 <= phi <= upper + 1e-9


def test_cheeger_bounds_closed_form_on_complete_graph() -> None:
    """K_n has normalized lambda_2 = n/(n-1)."""
    n = 8
    lower, upper = cheeger_bounds(nx.complete_graph(n))
    lambda2 = n / (n - 1)
    assert lower == pytest.approx(lambda2 / 2, abs=1e-9)
    assert upper == pytest.approx(np.sqrt(2 * lambda2), abs=1e-9)


# --- Directed input, and self-loops -----------------------------------------
# Two defects found by auditing this package against inputs the project would
# actually hand it, rather than against the graph families above. Both were
# silent: neither raised, and both returned a plausible number.


def _public_graph_functions() -> list:
    """Every exported function whose first parameter is a graph.

    Discovered from `__all__` rather than listed by hand, so a function added
    to the package later is covered by the directed-input test without anyone
    remembering to add it here.
    """
    import inspect

    import spectral_graph as sg

    found = []
    for name in sg.__all__:
        obj = getattr(sg, name)
        if not callable(obj):
            continue
        try:
            params = list(inspect.signature(obj).parameters.values())
        except (TypeError, ValueError):  # pragma: no cover - builtins
            continue
        if params and params[0].annotation in (nx.Graph, "nx.Graph"):
            found.append((name, obj))
    return found


@pytest.mark.parametrize(
    "name,fn", _public_graph_functions(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_every_public_entry_point_refuses_a_directed_graph(name, fn) -> None:
    """A DiGraph must raise, not return a number computed from half a matrix.

    `nx.adjacency_matrix` on a directed graph returns a non-symmetric A, so
    `L = D - A` is non-symmetric too, and `eigvalsh`/`eigsh` read a single
    triangle of it. Nothing raises and nothing is logged: the caller gets a
    well-formed float that is the algebraic connectivity of a matrix they
    never passed. On this project's own 928-node knowledge graph -- which is
    a DiGraph -- that path returned 0.865 against a true value of 0.267.

    The refusal is checked at every public entry point because the low-level
    guard in `laplacian.py` cannot cover the functions that read `G` directly
    without ever building a Laplacian; `conductance` is one.
    """
    D = nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 3)])
    extra = {"conductance": ({0, 1},)}.get(name, ())
    with pytest.raises(ValueError, match="undirected"):
        fn(D, *extra)


def test_the_undirected_projection_is_accepted_and_is_the_real_answer() -> None:
    """The caller converts, explicitly. The package does not do it for them.

    `to_undirected()` is a modelling decision -- it declares that a reversed
    edge means the same thing -- and it changes the answer. Making it silently
    would be the same wrong-number-without-warning failure the guard exists to
    stop, just moved somewhere less visible.
    """
    D = nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 3)])
    lambda2 = lambda_2(D.to_undirected())
    assert lambda2 == pytest.approx(nx.algebraic_connectivity(D.to_undirected()), abs=1e-9)


@pytest.mark.parametrize("weight", [1.0, 7.5], ids=["unweighted", "weighted"])
def test_a_self_loop_leaves_the_laplacian_singular(weight: float) -> None:
    """L @ 1 == 0 is what makes L a Laplacian, and a self-loop broke it.

    `G.degree()` counts a self-loop twice while `nx.adjacency_matrix` puts a
    single w on the diagonal, so `D - A` kept +w per self-loop and `L @ 1`
    came back [0, 1, 0] on a three-node graph. That is not a rounding
    artefact: it makes L non-PSD, puts a negative eigenvalue in the spectrum,
    and moves every eigenvalue after it. Degrees now come from A's row sums,
    which is also what NetworkX does.
    """
    G = nx.Graph()
    G.add_edge(0, 1, weight=weight)
    G.add_edge(1, 2, weight=weight)
    G.add_edge(1, 1, weight=weight)

    L = laplacian_matrix(G).toarray()
    assert np.allclose(L @ np.ones(3), 0.0, atol=TOL)
    assert np.allclose(L, L.T, atol=TOL)
    assert compute_spectrum(G).min() > -TOL
    assert np.allclose(L, nx.laplacian_matrix(G).toarray().astype(float), atol=TOL)


def test_a_self_loop_keeps_the_normalized_spectrum_in_range() -> None:
    """The normalized Laplacian's eigenvalues live in [0, 2]; a self-loop kept
    them there only once the degrees agreed with the adjacency they normalize."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    G.add_edge(1, 1)

    L_norm = normalized_laplacian_matrix(G).toarray()
    eigenvalues = np.linalg.eigvalsh(L_norm)
    assert eigenvalues.min() > -TOL
    assert eigenvalues.max() < 2.0 + 1e-9
    assert np.allclose(
        L_norm, nx.normalized_laplacian_matrix(G).toarray().astype(float), atol=1e-9
    )


def test_conductance_uses_one_volume_convention_on_both_sides() -> None:
    """vol(S) and vol(V) counted a self-loop differently, so phi could exceed 1.

    The numerator walks `G[u]`, which reaches u as its own neighbour once; the
    denominator used `G.degree()`, which counts it twice. Both now come from
    A's row sums.
    """
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    G.add_edge(0, 0)
    phi = conductance(G, {0, 1})
    assert 0.0 <= phi <= 1.0
    total_vol = float(adjacency_matrix(G).toarray().sum())
    vol_S = float(adjacency_matrix(G).toarray()[[0, 1], :].sum())
    assert phi == pytest.approx(1.0 / min(vol_S, total_vol - vol_S), abs=1e-12)


def test_sweep_cut_and_conductance_agree_on_the_set_sweep_cut_returns() -> None:
    """The two were computing different phi for the same cut on a self-loop.

    `sweep_cut` reports the conductance of the prefix it chose, so it is the
    same quantity `conductance` computes for that node set -- and on a graph
    with a self-loop they disagreed by 91%. Two compounding causes, both the
    self-loop bug the Laplacian already had: the loop was charged to the
    boundary (`v in in_S` is False for v == u, because u joins in_S only after
    the loop over its neighbours), and the total volume came from `G.degree()`,
    which counts a self-loop twice against the single w in A's row sum that
    the running vol_S accumulates.

    It matters beyond the arithmetic: sweep_cut *minimises* the number it
    reports, so a wrong objective can select a different prefix, and the phi
    it returns is what gets checked against the Cheeger bound.
    """
    G = nx.barbell_graph(5, 0)
    G.add_edge(0, 0)

    S, phi = sweep_cut(G)
    assert phi == pytest.approx(conductance(G, S), abs=1e-12)

    # The self-loop is invisible to the cut: it changes no edge's endpoints.
    clean = nx.barbell_graph(5, 0)
    _, phi_clean = sweep_cut(clean)
    assert phi == pytest.approx(phi_clean, abs=1e-12)


def test_sweep_cut_agrees_with_conductance_on_a_weighted_self_loop() -> None:
    """The unweighted case can pass by accident when every w is 1."""
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 2.0), (1, 2, 3.0), (2, 3, 0.5), (3, 0, 1.0)])
    G.add_edge(0, 0, weight=4.0)
    S, phi = sweep_cut(G)
    assert phi == pytest.approx(conductance(G, S), abs=1e-12)


@pytest.mark.parametrize(
    "G",
    [
        nx.path_graph(60),
        nx.barbell_graph(30, 10),
        nx.convert_node_labels_to_integers(nx.grid_2d_graph(8, 8)),
        nx.random_regular_graph(4, 60, seed=1),
    ],
    ids=["path_60", "barbell_70", "grid_64", "regular_60"],
)
def test_sparse_path_matches_the_dense_answer(G: nx.Graph) -> None:
    """Every graph here is over the n < 50 threshold, so it takes eigsh.

    The suite was entirely under that threshold, which left the sparse branch
    -- a different solver, reached by graph size alone -- untested. It is also
    the branch that changed when `which="SM"` was replaced by shift-invert, so
    it is the one that needs pinning to the dense answer rather than to
    whatever it produced last.
    """
    dense = np.sort(np.linalg.eigvalsh(nx.laplacian_matrix(G).toarray().astype(float)))
    assert lambda_2(G) == pytest.approx(dense[1], rel=1e-8)

    L = laplacian_matrix(G)
    evals, evecs = smallest_eigsh(L, k=3)
    assert np.allclose(evals, dense[:3], atol=1e-8)
    assert evecs.shape == (G.number_of_nodes(), 3)
    assert np.allclose(L @ evecs, evecs * evals, atol=1e-6)

    # The Fiedler vector is an eigenvector of L for lambda_2, whichever solver
    # produced it. Sign and scale are free, so check the residual, not entries.
    v = fiedler_vector(G)
    assert np.linalg.norm(L @ v - dense[1] * v) < 1e-6


def test_shift_invert_survives_the_singular_shift() -> None:
    """0 is an eigenvalue of every Laplacian, so sigma must miss it.

    `sigma=0` asks SuperLU to factorize a singular matrix and raises
    `RuntimeError: Factor is exactly singular`. The helper offsets sigma by a
    fraction of the largest diagonal entry; this pins that it holds on a
    weighted graph, where "a small shift" is not an absolute quantity.
    """
    G = nx.barbell_graph(30, 10)
    for u, v in G.edges():
        G[u][v]["weight"] = 1000.0

    evals = smallest_eigsh(laplacian_matrix(G), k=2, return_eigenvectors=False)
    dense = np.sort(np.linalg.eigvalsh(laplacian_matrix(G).toarray()))
    assert np.allclose(evals, dense[:2], rtol=1e-8, atol=1e-9)


def test_the_dense_path_returns_the_k_eigenvalues_asked_for():
    """Under 50 nodes the dense solver answered with the whole spectrum.

    `topics()` reads the largest gap in what it is handed, so on a small
    corpus it searched every eigenvalue rather than the low end -- and could
    propose more clusters than `MAX_AUTO_CLUSTERS` allows.
    """
    G = nx.karate_club_graph()  # 34 nodes: the dense path
    everything = np.sort(np.linalg.eigvalsh(normalized_laplacian_matrix(G).toarray()))

    low = compute_spectrum(G, k=5, normalized=True, which="SM")
    high = compute_spectrum(G, k=3, normalized=True, which="LA")

    assert low.shape == (5,) and np.allclose(low, everything[:5])
    assert high.shape == (3,) and np.allclose(high, everything[-3:])
    assert compute_spectrum(G).shape == (34,)
