#!/usr/bin/env python3
"""Measure what spectral methods buy you, per graph architecture.

Usage:
    python scripts/spectral_benchmark.py                 # every suite
    python scripts/spectral_benchmark.py --suite recovery
    python scripts/spectral_benchmark.py --json out.json

The package's own reports establish that the algorithms are *correct* on a
handful of graphs. The open question is a different one -- on which shapes of
graph is a spectral method worth reaching for, and on which does something
cheaper do as well or better. That question is only answerable against a
baseline, so every claim here is a delta against a non-spectral competitor
from networkx: Kernighan-Lin for cuts, greedy modularity and label propagation
for communities, and a random partition as the floor.

The architectures are chosen to span the axis that theory says should matter --
whether the graph *has* a bottleneck at all -- and to include null controls
(expander, Erdos-Renyi) where the honest answer is "there is nothing to find".
A method that scores well on the planted partitions and also claims structure
in an expander has not earned the win.

Everything here uses numpy/scipy/networkx only, the same stack the package
restricts itself to. Deterministic: every generator and every k-means restart
is seeded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from networkx.algorithms import community as nxc
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectral_graph import (
    cheeger_bounds,
    compute_spectrum,
    conductance,
    fiedler_vector,
    spectral_clustering,
    sweep_cut,
)
from spectral_graph.laplacian import laplacian_matrix

SEED = 20260902


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand Index between two labelings.

    Implemented here rather than imported from sklearn so the benchmark keeps
    the package's dependency floor (numpy/scipy/networkx). ARI is the right
    score for this comparison because the baselines do not all produce the
    same number of clusters -- label propagation picks its own -- and ARI is
    defined across differing cluster counts and corrected for chance, so the
    random baseline sits at 0.0 rather than at some size-dependent offset.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n = a.size
    if n < 2:
        return 0.0

    a_ids = {v: i for i, v in enumerate(np.unique(a))}
    b_ids = {v: i for i, v in enumerate(np.unique(b))}
    table = np.zeros((len(a_ids), len(b_ids)), dtype=np.int64)
    for x, y in zip(a, b, strict=True):
        table[a_ids[x], b_ids[y]] += 1

    def comb2(x: np.ndarray | np.int64) -> np.ndarray | np.int64:
        return x * (x - 1) // 2

    sum_cells = comb2(table).sum()
    sum_rows = comb2(table.sum(axis=1)).sum()
    sum_cols = comb2(table.sum(axis=0)).sum()
    total = comb2(np.int64(n))

    expected = sum_rows * sum_cols / total
    maximum = (sum_rows + sum_cols) / 2.0
    if maximum == expected:
        # Both labelings are trivial (one cluster, or all singletons).
        return 1.0 if sum_cells == sum_rows == sum_cols else 0.0
    return float((sum_cells - expected) / (maximum - expected))


def degree_heterogeneity(G: nx.Graph) -> float:
    """Coefficient of variation of the degree sequence.

    This is the number that predicts whether the normalized Laplacian will
    outperform the unnormalized one: normalization exists to stop high-degree
    nodes from dominating the embedding, so it can only help where the degrees
    actually differ. Near 0 on a regular graph, near 1+ on a scale-free one.
    """
    d = np.array([deg for _, deg in G.degree()], dtype=float)
    return float(d.std() / d.mean()) if d.mean() > 0 else 0.0


def eigengap_k(G: nx.Graph, max_k: int = 10) -> tuple[int, float]:
    """The eigengap heuristic's guess at the number of clusters.

    Returns (k_hat, gap). The heuristic reads k off the largest jump in the
    low end of the normalized spectrum: k weakly-coupled clusters produce k
    eigenvalues near 0 followed by a gap. Reported alongside the true k so
    the tables show where the heuristic is trustworthy -- on a graph with no
    community structure it still returns *some* k, and the size of the gap is
    the only thing that says whether to believe it.
    """
    k = min(max_k + 1, G.number_of_nodes() - 1)
    ev = compute_spectrum(G, k=k, normalized=True, which="SM")
    ev = np.sort(np.maximum(ev, 0.0))
    if ev.size < 3:
        return 1, 0.0
    gaps = np.diff(ev)
    # Ignore the gap out of the trivial eigenvalue; k=1 is not a finding.
    idx = int(np.argmax(gaps[1:])) + 1
    return idx + 1, float(gaps[idx])


# --------------------------------------------------------------------------
# Architectures
# --------------------------------------------------------------------------


@dataclass
class Architecture:
    """One graph shape, with the ground truth it was built to carry."""

    name: str
    kind: str  # "planted" (has ground truth) or "control" (has none)
    note: str
    build: Callable[[], nx.Graph]
    true_k: int | None = None


def _largest_component(G: nx.Graph) -> nx.Graph:
    """Restrict to the largest connected component, keeping node labels.

    Every technique here needs a connected graph: lambda_2 is 0 on a
    disconnected one and the Fiedler cut degenerates to "one component versus
    the rest", which is a true answer to a question nobody asked. Node labels
    are preserved so the `block` attribute still names each node's true
    community.
    """
    if nx.is_connected(G):
        return G
    nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(nodes).copy()


def _blocked(G: nx.Graph, sizes: list[int]) -> nx.Graph:
    """Tag consecutive runs of nodes with the block they belong to."""
    block = 0
    start = 0
    for size in sizes:
        for u in range(start, start + size):
            if u in G:
                G.nodes[u]["block"] = block
        start += size
        block += 1
    return G


def _sbm(sizes: list[int], p_in: float, p_out: float, seed: int) -> nx.Graph:
    probs = [[p_in if i == j else p_out for j in range(len(sizes))] for i in range(len(sizes))]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    G = nx.Graph(G)  # drop the graph-level block metadata, keep it simple
    return _largest_component(_blocked(G, sizes))


def _ring_of_cliques(n_cliques: int, clique_size: int) -> nx.Graph:
    G = nx.ring_of_cliques(n_cliques, clique_size)
    return _largest_component(_blocked(G, [clique_size] * n_cliques))


def _barbell(size: int, bridge: int) -> nx.Graph:
    G = nx.barbell_graph(size, bridge)
    # The path in the middle belongs to neither bell; split it down the middle
    # so the ground truth is the bipartition the graph was built to have.
    return _largest_component(_blocked(G, [size + bridge // 2, size + (bridge + 1) // 2]))


def _hierarchical(seed: int) -> nx.Graph:
    """Four communities nested in two super-communities.

    The interesting case for the eigengap heuristic, which has two defensible
    answers here (k=2 and k=4) and no way to prefer one.
    """
    sizes = [40, 40, 40, 40]
    probs = [
        [0.35, 0.09, 0.005, 0.005],
        [0.09, 0.35, 0.005, 0.005],
        [0.005, 0.005, 0.35, 0.09],
        [0.005, 0.005, 0.09, 0.35],
    ]
    G = nx.Graph(nx.stochastic_block_model(sizes, probs, seed=seed))
    return _largest_component(_blocked(G, sizes))


def _bipartite_corpus(seed: int) -> nx.Graph:
    """A document/entity bipartite graph with planted topics.

    This is the shape of this project's own knowledge graph -- documents on
    one side, the entities they mention on the other, every edge crossing --
    so it is the architecture whose result actually transfers to
    `graphrag_server.py`. Topics overlap slightly, via entities that two
    topics both mention, which is what a real corpus looks like.
    """
    rng = np.random.default_rng(seed)
    n_topics, docs, ents = 3, 25, 30
    G: nx.Graph = nx.Graph()
    for t in range(n_topics):
        for d in range(docs):
            G.add_node(("d", t, d), block=t, bipartite=0)
        for e in range(ents):
            G.add_node(("e", t, e), block=t, bipartite=1)
    for t in range(n_topics):
        for d in range(docs):
            for e in rng.choice(ents, size=6, replace=False):
                G.add_edge(("d", t, d), ("e", t, int(e)))
            # A few mentions leak into a neighbouring topic.
            if rng.random() < 0.25:
                other = (t + 1) % n_topics
                G.add_edge(("d", t, d), ("e", other, int(rng.integers(ents))))
    return _largest_component(G)


def _core_periphery(seed: int) -> nx.Graph:
    """A dense core with a sparse fringe attached to it.

    Not a community structure: there is no cut with low conductance, because
    every periphery node hangs off the core. Included because it *looks* like
    two groups to a human and to modularity, and the spectrum should say
    otherwise.
    """
    rng = np.random.default_rng(seed)
    core, periph = 40, 120
    G: nx.Graph = nx.Graph()
    for u in range(core):
        G.add_node(u, block=0)
    for u in range(core, core + periph):
        G.add_node(u, block=1)
    for u in range(core):
        for v in range(u + 1, core):
            if rng.random() < 0.5:
                G.add_edge(u, v)
    for u in range(core, core + periph):
        for v in rng.choice(core, size=2, replace=False):
            G.add_edge(u, int(v))
    return _largest_component(G)


ARCHITECTURES: list[Architecture] = [
    Architecture(
        "sbm_2_strong", "planted", "two well-separated communities",
        lambda: _sbm([90, 90], 0.30, 0.010, SEED), true_k=2,
    ),
    Architecture(
        "sbm_4_medium", "planted", "four communities, moderate coupling",
        lambda: _sbm([50, 50, 50, 50], 0.28, 0.012, SEED), true_k=4,
    ),
    Architecture(
        "sbm_2_weak", "planted", "two communities, near the detectability limit",
        lambda: _sbm([100, 100], 0.14, 0.070, SEED), true_k=2,
    ),
    Architecture(
        "ring_of_cliques", "planted", "6 cliques joined in a ring",
        lambda: _ring_of_cliques(6, 12), true_k=6,
    ),
    Architecture(
        "hierarchical_sbm", "planted", "4 communities nested in 2",
        lambda: _hierarchical(SEED), true_k=4,
    ),
    Architecture(
        "barbell", "planted", "two cliques joined by a path: one hard bottleneck",
        lambda: _barbell(60, 20), true_k=2,
    ),
    Architecture(
        "bipartite_corpus", "planted", "document/entity graph, 3 planted topics",
        lambda: _bipartite_corpus(SEED), true_k=3,
    ),
    Architecture(
        "core_periphery", "planted", "dense core, sparse fringe (no real cut)",
        lambda: _core_periphery(SEED), true_k=2,
    ),
    Architecture(
        "erdos_renyi", "control", "no structure at all",
        lambda: _largest_component(nx.erdos_renyi_graph(180, 0.06, seed=SEED)),
    ),
    Architecture(
        "random_regular", "control", "expander: provably no low-conductance cut",
        lambda: _largest_component(nx.random_regular_graph(6, 180, seed=SEED)),
    ),
    Architecture(
        "barabasi_albert", "control", "scale-free, hub-dominated",
        lambda: _largest_component(nx.barabasi_albert_graph(180, 3, seed=SEED)),
    ),
    Architecture(
        "watts_strogatz", "control", "small-world ring",
        lambda: _largest_component(nx.watts_strogatz_graph(180, 6, 0.05, seed=SEED)),
    ),
    Architecture(
        "grid_2d", "control", "geometric mesh: a good cut with no communities",
        lambda: nx.convert_node_labels_to_integers(nx.grid_2d_graph(14, 14)),
    ),
    Architecture(
        "balanced_tree", "control", "hierarchy with no cycles",
        lambda: nx.balanced_tree(3, 4),
    ),
]


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def _labels_from_communities(G: nx.Graph, communities) -> np.ndarray:
    index = {u: i for i, u in enumerate(G.nodes())}
    labels = np.zeros(G.number_of_nodes(), dtype=int)
    for c, nodes in enumerate(communities):
        for u in nodes:
            labels[index[u]] = c
    return labels


def baseline_greedy_modularity(G: nx.Graph, k: int) -> np.ndarray:
    """Greedy modularity (Clauset-Newman-Moore), forced to exactly k groups."""
    communities = nxc.greedy_modularity_communities(G, cutoff=k, best_n=k)
    return _labels_from_communities(G, communities)


def baseline_label_propagation(G: nx.Graph, seed: int = SEED) -> np.ndarray:
    """Asynchronous label propagation. Picks its own number of communities."""
    return _labels_from_communities(G, list(nxc.asyn_lpa_communities(G, seed=seed)))


def baseline_random(G: nx.Graph, k: int, seed: int = SEED) -> np.ndarray:
    """Uniformly random assignment: the floor every method must clear."""
    return np.random.default_rng(seed).integers(0, k, size=G.number_of_nodes())


def baseline_kernighan_lin(G: nx.Graph, seed: int = SEED) -> set:
    """Kernighan-Lin bisection: the classic non-spectral local-search cut."""
    left, _ = nxc.kernighan_lin_bisection(G, seed=seed)
    return set(left)


def baseline_random_bisection(G: nx.Graph, seed: int = SEED) -> set:
    nodes = list(G.nodes())
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(nodes), size=len(nodes) // 2, replace=False)
    return {nodes[int(i)] for i in picked}


# --------------------------------------------------------------------------
# Suites
# --------------------------------------------------------------------------


def _timed(fn: Callable, *args, **kwargs) -> tuple[object, float]:
    """Call `fn(*args)` and return (result, wall seconds).

    Arguments are passed through rather than closed over in a lambda: a lambda
    written in a loop binds the loop variable by reference, which is harmless
    while the call is immediate and a live trap the moment anyone defers one.
    """
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def fiedler_support(G: nx.Graph, normalized: bool) -> float:
    """How many nodes the Fiedler vector actually spreads its mass over.

    The inverse participation ratio, `1 / sum(v_i^4)` for a unit-norm `v`: the
    number of nodes carrying the vector, as opposed to the number it has
    entries for. This is the column that explains the `sbm_2_weak` result,
    where the normalized Laplacian recovers the planted cut and the
    unnormalized one scores exactly chance. The intuitive explanation --
    degree heterogeneity -- is wrong here and the benchmark says so: degCV is
    0.20 for both. What actually happens is that the unnormalized Fiedler
    vector *localizes*, concentrating on ~5 of 200 nodes, so its sign carries
    no global partition; the normalized one spreads over ~103 of 200 and its
    sign is the cut.
    """
    v = fiedler_vector(G, normalized=normalized)
    v = v / np.linalg.norm(v)
    return float(1.0 / (v**4).sum())


def run_recovery(archs: list[Architecture]) -> list[dict]:
    """Can the method recover a community structure that is actually there?"""
    rows = []
    for arch in archs:
        if arch.kind != "planted":
            continue
        G = arch.build()
        truth = np.array([G.nodes[u]["block"] for u in G.nodes()])
        k = arch.true_k or len(set(truth))

        spec_n, t_spec = _timed(spectral_clustering, G, k=k, normalized=True)
        spec_u, _ = _timed(spectral_clustering, G, k=k, normalized=False)
        greedy, t_greedy = _timed(baseline_greedy_modularity, G, k)
        lpa, t_lpa = _timed(baseline_label_propagation, G)
        rand = baseline_random(G, k)

        k_hat, gap = eigengap_k(G)
        scores = {
            "spectral (norm)": adjusted_rand_index(truth, spec_n),
            "spectral (unnorm)": adjusted_rand_index(truth, spec_u),
            "greedy modularity": adjusted_rand_index(truth, greedy),
            "label propagation": adjusted_rand_index(truth, lpa),
            "random": adjusted_rand_index(truth, rand),
        }
        best = max(scores, key=lambda name: scores[name])
        # A "winner" among methods that are all at chance is not a finding.
        # Below this, every method has failed to recover anything and saying
        # which failed by the smallest margin would read as a result.
        no_structure_found = scores[best] < 0.05
        rows.append(
            {
                "architecture": arch.name,
                "note": arch.note,
                "n": G.number_of_nodes(),
                "m": G.number_of_edges(),
                "true_k": k,
                "eigengap_k": k_hat,
                "eigengap": gap,
                "deg_cv": degree_heterogeneity(G),
                "support_norm": fiedler_support(G, normalized=True),
                "support_unnorm": fiedler_support(G, normalized=False),
                "ari": scores,
                "t_spectral": t_spec,
                "t_greedy": t_greedy,
                "t_lpa": t_lpa,
                "winner": "none (all at chance)" if no_structure_found else best,
                "no_structure_found": no_structure_found,
                "spectral_margin": scores["spectral (norm)"]
                - max(scores["greedy modularity"], scores["label propagation"]),
            }
        )
    return rows


def run_cut(archs: list[Architecture]) -> list[dict]:
    """How good is the cut, and is the Cheeger certificate worth anything?"""
    rows = []
    for arch in archs:
        G = arch.build()
        (S_sweep, phi_sweep), t_sweep = _timed(sweep_cut, G, normalized=True)
        S_kl, t_kl = _timed(baseline_kernighan_lin, G)
        phi_kl = conductance(G, S_kl)
        phi_rand = conductance(G, baseline_random_bisection(G))
        lo, hi = cheeger_bounds(G)
        lam2 = float(np.sort(compute_spectrum(G, k=2, which="SM"))[1])
        # Cheeger is stated for the *normalized* Laplacian, so mu_2 is the
        # eigenvalue the bracket is built from. Reported next to lambda_2
        # because the two differ by an order of magnitude on a graph with
        # heterogeneous degrees and confusing them silently rescales the
        # bracket.
        mu2 = float(np.sort(compute_spectrum(G, k=2, normalized=True, which="SM"))[1])

        rows.append(
            {
                "architecture": arch.name,
                "kind": arch.kind,
                "n": G.number_of_nodes(),
                "lambda2": lam2,
                "mu2": mu2,
                "phi_sweep": phi_sweep,
                "phi_kl": phi_kl,
                "phi_random": phi_rand,
                "cheeger_lo": lo,
                "cheeger_hi": hi,
                "bracket_ratio": (hi / lo) if lo > 1e-12 else float("inf"),
                "in_bracket": bool(lo - 1e-9 <= phi_sweep <= hi + 1e-9),
                "balance": min(len(S_sweep), G.number_of_nodes() - len(S_sweep))
                / max(len(S_sweep), G.number_of_nodes() - len(S_sweep), 1),
                "t_sweep": t_sweep,
                "t_kl": t_kl,
            }
        )
    return rows


def run_solver() -> list[dict]:
    """What the package's sparse path costs, against the standard alternative.

    Every sparse branch in the package calls `eigsh(..., which="SM")`. ARPACK
    in that mode iterates on L directly, and its convergence rate is set by
    the *relative* separation of the eigenvalues it is chasing -- so it is
    slowest exactly where lambda_2 is smallest, which is exactly the graph
    you reached for a spectral method to analyse. Shift-invert
    (`sigma`, `which="LM"`) inverts that relationship. Both are measured here
    on the same matrices; the answers agree, so this is a cost finding, not a
    correctness one.
    """
    cases = [
        ("path", nx.path_graph(1200)),
        ("cycle", nx.cycle_graph(1200)),
        ("barbell", nx.barbell_graph(400, 200)),
        ("grid_40x40", nx.convert_node_labels_to_integers(nx.grid_2d_graph(40, 40))),
        ("random_regular", nx.random_regular_graph(6, 1200, seed=SEED)),
        ("barabasi_albert", nx.barabasi_albert_graph(1200, 3, seed=SEED)),
    ]
    rows = []
    for name, G in cases:
        L = laplacian_matrix(G)

        def _sm(L=L):
            return np.sort(eigsh(L, k=2, which="SM", return_eigenvectors=False))[1]

        # sigma is a hair below 0: sigma=0 exactly is a singular factorization,
        # since 0 is an eigenvalue of every Laplacian.
        def _si(L=L):
            return np.sort(eigsh(L, k=2, sigma=-1e-6, which="LM", return_eigenvectors=False))[1]

        # ARPACK starts from a random residual vector unless one is supplied,
        # so "SM" on a bottlenecked graph does not merely take longer -- it
        # sometimes exhausts its iteration budget and raises. Measured here at
        # 1 failure in 6 runs on path_1200, each burning ~6s before giving up.
        # A benchmark that dies on that has measured the interesting case and
        # then thrown the number away, so the failure is a result, not a crash.
        try:
            sm, t_sm = _timed(_sm)
            sm_failed = False
        except ArpackNoConvergence:
            sm, t_sm, sm_failed = float("nan"), float("nan"), True
        si, t_si = _timed(_si)
        rows.append(
            {
                "graph": name,
                "n": G.number_of_nodes(),
                "lambda2": float(si),
                "t_sm": t_sm,
                "t_shift_invert": t_si,
                "sm_failed": sm_failed,
                "speedup": t_sm / t_si if t_si > 0 else float("inf"),
                # 1e-6 relative, not 1e-8: on the barbell lambda_2 is 2.3e-05
                # and ARPACK's own residual tolerance puts the two answers
                # 2.8e-08 apart relative -- with shift-invert the closer of the
                # two to a tol=0 reference. A tighter check here does not
                # detect a wrong answer, it detects the solver's precision.
                "agree": (
                    False
                    if sm_failed
                    else bool(abs(float(sm) - float(si)) <= 1e-6 * max(abs(float(si)), 1e-12))
                ),
            }
        )
    return rows


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)
    ]
    out = ["  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)).rstrip()]
    out.append("  ".join("-" * w for w in widths))
    for r in rows:
        out.append("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip())
    return "\n".join(out)


def print_recovery(rows: list[dict]) -> None:
    print("\n=== 1. Community recovery (ARI vs planted truth; 1.0 = exact, 0.0 = chance) ===\n")
    print(
        _table(
            ["architecture", "n", "k", "k̂", "spec-N", "spec-U", "greedy", "lpa", "rand",
             "ms(spec)", "ms(greedy)", "winner"],
            [
                [
                    r["architecture"],
                    str(r["n"]),
                    str(r["true_k"]),
                    str(r["eigengap_k"]),
                    f"{r['ari']['spectral (norm)']:.3f}",
                    f"{r['ari']['spectral (unnorm)']:.3f}",
                    f"{r['ari']['greedy modularity']:.3f}",
                    f"{r['ari']['label propagation']:.3f}",
                    f"{r['ari']['random']:.3f}",
                    f"{r['t_spectral'] * 1000:.1f}",
                    f"{r['t_greedy'] * 1000:.1f}",
                    r["winner"],
                ]
                for r in rows
            ],
        )
    )
    print("\n  spec-N/U = spectral clustering on the normalized / unnormalized Laplacian.")
    print("  k̂ = the eigengap heuristic's guess at k, against the planted k.")

    print("\n  Why normalization matters -- Fiedler vector support (nodes carrying the mass):\n")
    print(
        _table(
            ["architecture", "n", "degCV", "support (norm)", "support (unnorm)", "spec-N", "spec-U"],
            [
                [
                    r["architecture"],
                    str(r["n"]),
                    f"{r['deg_cv']:.2f}",
                    f"{r['support_norm']:.1f}",
                    f"{r['support_unnorm']:.1f}",
                    f"{r['ari']['spectral (norm)']:.3f}",
                    f"{r['ari']['spectral (unnorm)']:.3f}",
                ]
                for r in rows
            ],
        )
    )
    print("\n  degCV does NOT predict the gap between the two -- see sbm_2_weak, where degCV is")
    print("  identical for both and the ARI gap is 0.6. Localization does: an unnormalized")
    print("  Fiedler vector supported on a handful of nodes has no global cut in its sign.")


def print_cut(rows: list[dict]) -> None:
    print("\n=== 2. Cut quality and the Cheeger certificate (lower φ is better) ===\n")
    print(
        _table(
            ["architecture", "kind", "λ₂", "μ₂", "φ sweep", "φ KL", "φ rand",
             "Cheeger bracket", "hi/lo", "in?"],
            [
                [
                    r["architecture"],
                    r["kind"],
                    f"{r['lambda2']:.4f}",
                    f"{r['mu2']:.4f}",
                    f"{r['phi_sweep']:.4f}",
                    f"{r['phi_kl']:.4f}",
                    f"{r['phi_random']:.4f}",
                    f"[{r['cheeger_lo']:.3f}, {r['cheeger_hi']:.3f}]",
                    f"{r['bracket_ratio']:.1f}x",
                    "yes" if r["in_bracket"] else "NO",
                ]
                for r in rows
            ],
        )
    )
    print("\n  λ₂ is the unnormalized Fiedler value; μ₂ the normalized one, which is what the")
    print("  Cheeger bracket [μ₂/2, √(2μ₂)] is stated for.")
    print("  φ sweep = Fiedler sweep cut; φ KL = Kernighan-Lin; φ rand = random bisection.")
    print("  hi/lo = width of the bracket. A wide bracket is a weak certificate.")


def print_solver(rows: list[dict]) -> None:
    print("\n=== 3. Sparse eigensolver cost: the package's which='SM' vs shift-invert ===\n")
    print(
        _table(
            ["graph", "n", "λ₂", "SM (s)", "shift-inv (s)", "SM slower by", "same answer"],
            [
                [
                    r["graph"],
                    str(r["n"]),
                    f"{r['lambda2']:.3e}",
                    "no convergence" if r["sm_failed"] else f"{r['t_sm']:.3f}",
                    f"{r['t_shift_invert']:.3f}",
                    "n/a" if r["sm_failed"]
                    else (f"{r['speedup']:.0f}x" if r["speedup"] >= 10 else f"{r['speedup']:.2f}x"),
                    "SM FAILED" if r["sm_failed"] else ("yes" if r["agree"] else "NO"),
                ]
                for r in rows
            ],
        )
    )
    print("\n  When 'SM' converges it returns the same lambda_2 to 1e-6 relative. The gap")
    print("  tracks 1/lambda_2: it is slowest on precisely the bottlenecked graphs a")
    print("  spectral analysis is for, and pays a fixed factorization cost it cannot")
    print("  recover on an expander (the rows below 1x).")
    print()
    print("  It does not always converge. ARPACK draws a random starting residual unless")
    print("  given one, so on path_1200 this raises ArpackNoConvergence roughly 1 run in 6,")
    print("  after ~6s of work -- an intermittent exception, not a slow answer. Shift-invert")
    print("  did not fail once in the same trial. That is why")
    print("  `spectral_graph.spectrum.smallest_eigsh` takes the shift-invert path: it bounds")
    print("  the worst case instead of the best one.")


def print_verdict(recovery: list[dict], cut: list[dict]) -> None:
    print("\n=== 4. Verdict: where the spectrum earns its keep ===\n")
    wins, ties, losses = [], [], []
    for r in recovery:
        margin = r["spectral_margin"]
        if margin > 0.05:
            wins.append((r["architecture"], margin))
        elif margin < -0.05:
            losses.append((r["architecture"], margin))
        else:
            ties.append((r["architecture"], margin))

    def fmt(items):
        return ", ".join(f"{name} ({m:+.2f})" for name, m in items) or "none"

    print(f"  Spectral beats the best baseline on: {fmt(wins)}")
    print(f"  Statistical tie on:                  {fmt(ties)}")
    print(f"  Spectral loses on:                   {fmt(losses)}")

    print("\n  Null controls -- can the method certify that there is nothing to find?")
    print("  The Cheeger *lower* bound μ₂/2 is a proof of absence: no cut anywhere in the")
    print("  graph beats it. That, not the conductance actually found, is what distinguishes")
    print("  'there is no bottleneck' from 'I did not look hard enough'.\n")
    for r in cut:
        if r["kind"] != "control":
            continue
        floor = r["cheeger_lo"]
        if floor > 0.10:
            verdict = f"certified: no cut below φ={floor:.3f} exists"
        else:
            verdict = f"a good cut is possible, and one was found at φ={r['phi_sweep']:.4f}"
        print(f"    {r['architecture']:16s} μ₂={r['mu2']:.4f}  {verdict}")

    sweep_wins = sum(1 for r in cut if r["phi_sweep"] < r["phi_kl"] - 1e-9)
    kl_wins = sum(1 for r in cut if r["phi_kl"] < r["phi_sweep"] - 1e-9)
    print(f"\n  Cut quality across all {len(cut)} architectures: "
          f"sweep cut better on {sweep_wins}, Kernighan-Lin better on {kl_wins}, "
          f"tied on {len(cut) - sweep_wins - kl_wins}.")
    # KL optimizes a balanced bisection; the sweep cut is free to return an
    # unbalanced one. Splitting the tally by balance is what makes the
    # comparison mean something rather than averaging two different objectives.
    unbalanced = [r for r in cut if r["balance"] < 0.5]
    if unbalanced:
        sweep_better = sum(1 for r in unbalanced if r["phi_sweep"] < r["phi_kl"] - 1e-9)
        print(f"  Restricted to the {len(unbalanced)} architectures whose best cut is unbalanced "
              f"(balance < 0.5), the sweep cut wins {sweep_better}: Kernighan-Lin is constrained "
              f"to halves and cannot express those cuts at all.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--suite",
        choices=["all", "recovery", "cut", "solver"],
        default="all",
        help="which measurement to run (default: all)",
    )
    parser.add_argument("--json", metavar="PATH", help="also write the raw numbers to this file")
    args = parser.parse_args()

    results: dict = {}
    recovery: list[dict] = []
    cut: list[dict] = []

    if args.suite in ("all", "recovery"):
        recovery = run_recovery(ARCHITECTURES)
        results["recovery"] = recovery
        print_recovery(recovery)

    if args.suite in ("all", "cut"):
        cut = run_cut(ARCHITECTURES)
        results["cut"] = cut
        print_cut(cut)

    if args.suite in ("all", "solver"):
        solver = run_solver()
        results["solver"] = solver
        print_solver(solver)

    if args.suite == "all":
        print_verdict(recovery, cut)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=float))
        print(f"\nRaw numbers written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
