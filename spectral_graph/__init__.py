"""Spectral graph theory on NumPy/SciPy, behind the corpus diagnostics.

`corpus_spectral` reads the knowledge graph's connectivity, topics and
bottleneck through these functions. Every entry point takes an undirected
NetworkX graph and refuses a directed one.

The package lives at the project root rather than inside the installed
distribution, so it is importable only with the root on `sys.path`.
"""

from spectral_graph.clustering import (
    cheeger_bounds,
    conductance,
    spectral_clustering,
    sweep_cut,
)
from spectral_graph.embedding import spectral_embedding
from spectral_graph.fiedler import fiedler_vector
from spectral_graph.laplacian import (
    adjacency_matrix,
    laplacian_matrix,
    normalized_laplacian_matrix,
)
from spectral_graph.spectrum import compute_spectrum

__all__ = [
    "adjacency_matrix",
    "cheeger_bounds",
    "compute_spectrum",
    "conductance",
    "fiedler_vector",
    "laplacian_matrix",
    "normalized_laplacian_matrix",
    "spectral_clustering",
    "spectral_embedding",
    "sweep_cut",
]
