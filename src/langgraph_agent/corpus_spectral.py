"""Whole-graph spectral analysis of the corpus, mixed into `GraphRAGKnowledgeBase`.

Four questions about the knowledge graph's shape, apart from the store that
holds it: `connectivity`, `topics`, `bottleneck` and `duplicate_entities`.
Each refuses to report a finding it cannot support -- `topics` answers
`no_clear_structure` rather than invent a `k`, `bottleneck` separates a
certified absence from an inconclusive one, `connectivity` reports `lambda_2`
as None rather than 0.0 when there is nothing to measure, and
`duplicate_entities` only proposes.

`spectral_graph` lives at the project root, outside the installed package, so
it is imported inside each method: a missing diagnostic must not stop the
module loading.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from itertools import combinations
from typing import Any

import networkx as nx

# Below this conductance a cut counts as a narrow waist: leaving the set,
# roughly one edge in ten crosses. A convention, not a theorem; the raw
# conductance and both Cheeger bounds are always returned beside it.
BOTTLENECK_CONDUCTANCE = 0.1
# How many times larger the winning eigengap must be than the runner-up before
# the cluster count it implies is believed. Across 18 corpora with a planted
# topic count the eigengap was right every time at 5.1-23.4; on graphs with no
# communities it still returned some k, at 1.0-1.8
# (`scripts/spectral_benchmark.py`). 3.0 sits in the empty space between.
EIGENGAP_DECISIVENESS = 3.0
# The largest k the eigengap may propose: a map with more parts is not read,
# and the heuristic's wrong answers sat at the top of its range.
MAX_AUTO_CLUSTERS = 12
# How alike two entity names must read before the pair is proposed as a merge
# (`Builder` / `Builders` scores 0.93). A pair that is not a case variant must
# also clear `DUPLICATE_CONTAINMENT`, which is what excludes `Ent11` /
# `Ent11x`.
DUPLICATE_NAME_SIMILARITY = 0.85
# How much of the rarer entity's neighbourhood the commoner one must cover.
# Containment, not Jaccard: a real duplicate is asymmetric -- `Builder` in 29
# documents, `Builders` in 4 -- which scores containment 1.00 and Jaccard 0.14.
DUPLICATE_CONTAINMENT = 0.5
# Entities are compared only against others sharing this many leading
# characters, case-folded: 620x fewer comparisons, and no loss, since a pair
# that shares no prefix cannot clear `DUPLICATE_NAME_SIMILARITY`.
DUPLICATE_BLOCK_PREFIX = 4


class CorpusSpectralMixin:
    """The four whole-graph diagnostics, mixed into `GraphRAGKnowledgeBase`."""

    # Supplied by the class this is mixed into.
    graph: nx.DiGraph
    _connectivity_cache: tuple[tuple[int, int], dict[str, Any]] | None

    def connectivity(self) -> dict[str, Any]:
        """Structural health of the knowledge graph: components, and lambda_2.

        An entity-extraction regression drops edges without changing the document
        count or raising; it shows here first, as more components or a collapsing
        `lambda_2`.

        Components come from networkx, not from counting zero eigenvalues: that is
        40x the cost, and on the normalized Laplacian an isolated node contributes
        eigenvalue 1, not 0. `lambda_2` is the normalized one of the largest
        component -- on a disconnected whole graph it is always 0, and unnormalized
        it grows with degree, making rebuilds incomparable. None, not 0.0, when the
        largest component has under two nodes.
        """
        # Every edge runs document -> entity, so the undirected view states the
        # same relation; `spectral_graph` refuses a DiGraph rather than
        # guessing that.
        #
        # Fetched pages and markup, script and config files are entity-free by
        # design (`_is_web_document`, `ENTITY_FREE_SUFFIXES`), so they would be
        # permanent false isolates burying the regression this exists to catch.
        # They are left out, and counted apart as `web_documents` and
        # `entity_free_sources`.
        from langgraph_agent.graphrag_server import _is_web_document, _mints_entities

        documents = [n for n, a in self.graph.nodes(data=True) if a.get("type") == "document"]
        web = {n for n in documents if _is_web_document(n)}
        sources = {n for n in documents if n not in web and not _mints_entities(n)}
        excluded = web | sources
        linked = self.graph.subgraph([n for n in self.graph if n not in excluded])
        undirected = linked.to_undirected(as_view=True)
        n = undirected.number_of_nodes()

        if n == 0:
            return {"components": 0, "largest_component": 0, "isolated_nodes": 0,
                    "lambda_2": None, "web_documents": len(web),
                    "entity_free_sources": len(sources)}

        components = nx.number_connected_components(undirected)
        largest = max(nx.connected_components(undirected), key=len)
        isolated = sum(1 for _, degree in undirected.degree() if degree == 0)

        lambda_2: float | None = None
        unavailable: str | None = None
        if len(largest) < 2:
            unavailable = "largest component has fewer than 2 nodes"
        else:
            try:
                from spectral_graph import compute_spectrum

                spectrum = compute_spectrum(
                    undirected.subgraph(largest), k=2, normalized=True, which="SM"
                )
                # lambda_2 >= 0; a small negative is solver noise.
                lambda_2 = max(float(spectrum[1]), 0.0)
            except ImportError:
                unavailable = "spectral_graph is not on sys.path"
            except Exception as exc:  # pragma: no cover - solver-dependent
                # A diagnostic must not cost the header its counters; named, so
                # "could not measure" never reads as "measured 0".
                unavailable = f"{type(exc).__name__}: {exc}"

        result: dict[str, Any] = {
            "components": components,
            "largest_component": len(largest),
            "isolated_nodes": isolated,
            "lambda_2": lambda_2,
            "web_documents": len(web),
            "entity_free_sources": len(sources),
        }
        if unavailable is not None:
            result["lambda_2_unavailable"] = unavailable
        return result

    def topics(
        self, k: int | None = None, max_entities: int = 6, max_documents: int = 4
    ) -> dict[str, Any]:
        """Group the corpus into topic communities, or say there are none.

        Ng-Jordan-Weiss spectral clustering on the largest component: on a bipartite
        document/entity graph each cluster is documents together with the entities
        that define them, so it reads as a topic.

        The eigengap always proposes *some* k, so a proposal is believed only when its
        gap beats the runner-up by `EIGENGAP_DECISIVENESS`; below that the verdict is
        `no_clear_structure` and no clusters are returned. An explicit `k` skips the
        gate, and the decisiveness is reported either way. Each cluster carries its
        own conductance, the independent check: a k that split a real community shows
        up as clusters with high conductance.
        """
        undirected = self.graph.to_undirected(as_view=True)
        if undirected.number_of_nodes() == 0:
            return {"verdict": "no_graph", "note": "The graph is empty.", "clusters": []}

        largest = max(nx.connected_components(undirected), key=len)
        if len(largest) < 4:
            return {
                "verdict": "no_graph",
                "note": "The largest component is too small to divide into topics.",
                "clusters": [],
            }
        component = undirected.subgraph(largest)
        n = component.number_of_nodes()

        # A caller's bad k is an error, not a verdict: `unavailable` is for a
        # measurement that could not be taken.
        if k is not None and not 2 <= k <= n:
            raise ValueError(
                f"k must be between 2 and {n} (the largest component), got {k}"
            )

        try:
            import numpy as np

            from spectral_graph import compute_spectrum, conductance, spectral_clustering

            # One eigenvalue past the largest k worth proposing, so its gap is
            # in the window.
            probe = min(MAX_AUTO_CLUSTERS + 1, n - 1)
            spectrum = np.sort(
                np.maximum(compute_spectrum(component, k=probe, normalized=True, which="SM"), 0.0)
            )
            # Skip the gap out of the trivial eigenvalue: k = 1 is not a finding.
            gaps = np.diff(spectrum)[1:]
            order = np.argsort(gaps)[::-1]
            best = float(gaps[order[0]])
            runner_up = float(gaps[order[1]]) if len(order) > 1 else 0.0
            decisiveness = best / runner_up if runner_up > 1e-12 else float("inf")
            suggested = int(order[0]) + 2

            if k is None:
                if decisiveness < EIGENGAP_DECISIVENESS:
                    return {
                        "verdict": "no_clear_structure",
                        "note": (
                            f"The eigengap suggests {suggested} clusters but only "
                            f"{decisiveness:.1f}x more strongly than the next candidate, "
                            f"under the {EIGENGAP_DECISIVENESS}x this needs to be worth "
                            f"reporting. Corpora with no topic structure still produce a "
                            f"suggestion; this one looks like that. Pass an explicit k to "
                            f"cluster anyway."
                        ),
                        "suggested_k": suggested,
                        "decisiveness": decisiveness,
                        "threshold": EIGENGAP_DECISIVENESS,
                        "clusters": [],
                    }
                k, k_source = suggested, "eigengap"
            else:
                k_source = "requested"

            labels = spectral_clustering(component, k=k, normalized=True)
        except ImportError:
            return {"verdict": "unavailable", "note": "spectral_graph is not on sys.path.",
                    "clusters": []}
        except Exception as exc:  # pragma: no cover - solver-dependent
            return {"verdict": "unavailable", "note": f"{type(exc).__name__}: {exc}",
                    "clusters": []}

        nodes = list(component.nodes())
        clusters: list[dict[str, Any]] = []
        for label in range(k):
            members = [nodes[i] for i in range(len(nodes)) if labels[i] == label]
            if not members:
                continue
            member_set = set(members)
            documents = [
                node for node in members
                if self.graph.nodes[node].get("type") == "document"
            ]
            entities = [
                node for node in members
                if self.graph.nodes[node].get("type") == "entity"
            ]
            # In effect the cluster's name: what its documents have in common.
            top_entities = sorted(
                entities, key=lambda node: (-component.degree(node), str(node))
            )[:max_entities]
            clusters.append(
                {
                    "id": label,
                    "size": len(members),
                    "documents": len(documents),
                    "entities": len(entities),
                    "conductance": (
                        float(conductance(component, member_set))
                        if 0 < len(member_set) < n
                        else None
                    ),
                    "top_entities": top_entities,
                    "sample_documents": sorted(documents, key=str)[:max_documents],
                }
            )

        clusters.sort(key=lambda cluster: -cluster["size"])
        return {
            "verdict": "clustered",
            "k": k,
            "k_source": k_source,
            "suggested_k": suggested,
            "decisiveness": decisiveness,
            "threshold": EIGENGAP_DECISIVENESS,
            "component_size": n,
            "clusters": clusters,
        }

    def bottleneck(self, limit: int = 12) -> dict[str, Any]:
        """The narrowest cut in the corpus, and the nodes that bridge it.

        Sweeps the normalized Fiedler vector for the prefix of lowest conductance and
        names the nodes whose edges cross it -- the terms a search should expand on
        when a query straddles two topic areas, which degree alone does not find.

        A minimisation always returns *some* cut, so the verdict has three states:

        - `certified_none` -- Cheeger's lower bound `mu_2 / 2` is above
          `BOTTLENECK_CONDUCTANCE`: no narrow waist exists anywhere. A theorem about
          the graph, not about the cut found.
        - `found` -- the sweep cut is at or below the line; the bridges are real.
        - `inconclusive` -- the bound permits a bottleneck and the sweep found none.
          Cheeger's bracket is wide, so the sweep can miss one; saying so is the
          honest answer.

        Runs on the largest component: on a disconnected graph the Fiedler vector
        just separates a component from the rest, which `connectivity()` reports.
        """
        # Document -> entity edges: the undirected view states the same
        # relation.
        undirected = self.graph.to_undirected(as_view=True)

        if undirected.number_of_nodes() == 0:
            return {"verdict": "no_graph", "note": "The graph is empty.",
                    "conductance": None, "bridge_nodes": []}

        largest = max(nx.connected_components(undirected), key=len)
        if len(largest) < 2:
            return {
                "verdict": "no_graph",
                "note": "The largest component has a single node; there is nothing to cut.",
                "conductance": None,
                "bridge_nodes": [],
            }

        component = undirected.subgraph(largest)

        try:
            from spectral_graph import cheeger_bounds, compute_spectrum, sweep_cut

            side, phi = sweep_cut(component, normalized=True)
            lower, upper = cheeger_bounds(component)
            # mu_3 as well, to detect a tie -- see `tied_cuts`.
            spectrum = compute_spectrum(component, k=3, normalized=True, which="SM")
        except ImportError:
            return {
                "verdict": "unavailable",
                "note": "spectral_graph is not on sys.path.",
                "conductance": None,
                "bridge_nodes": [],
            }
        except Exception as exc:  # pragma: no cover - solver-dependent
            return {
                "verdict": "unavailable",
                "note": f"{type(exc).__name__}: {exc}",
                "conductance": None,
                "bridge_nodes": [],
            }

        if lower > BOTTLENECK_CONDUCTANCE:
            verdict = "certified_none"
        elif phi <= BOTTLENECK_CONDUCTANCE:
            verdict = "found"
        else:
            verdict = "inconclusive"

        # mu_2 ~= mu_3: more than two topic areas, and the Fiedler vector picks
        # one of several equally narrow cuts arbitrarily. The sides can then
        # differ between runs while the conductance and the bridges stay put,
        # which reads as a broken diagnostic unless it is said.
        mu_2, mu_3 = float(spectrum[1]), float(spectrum[2])
        tied_cuts = bool(mu_3 - mu_2 <= 0.1 * mu_3) if mu_3 > 1e-12 else False

        # The nodes carrying the cut, by how much of it they carry; total
        # degree beside it, since a bridge is a node whose few edges happen to
        # be load-bearing.
        crossing: dict[str, int] = {}
        crossing_edges = 0
        for source, target in component.edges():
            if (source in side) != (target in side):
                crossing_edges += 1
                crossing[source] = crossing.get(source, 0) + 1
                crossing[target] = crossing.get(target, 0) + 1

        bridge_nodes = [
            {
                "id": node,
                "type": self.graph.nodes[node].get("type", "unknown"),
                "crossing_edges": count,
                "degree": component.degree(node),
                "side": "a" if node in side else "b",
            }
            for node, count in sorted(
                crossing.items(), key=lambda item: (-item[1], str(item[0]))
            )[:limit]
        ]

        return {
            "verdict": verdict,
            "conductance": float(phi),
            "cheeger_lower": float(lower),
            "cheeger_upper": float(upper),
            "threshold": BOTTLENECK_CONDUCTANCE,
            "component_size": component.number_of_nodes(),
            "side_a": len(side),
            "side_b": component.number_of_nodes() - len(side),
            "crossing_edges": crossing_edges,
            "bridge_nodes": bridge_nodes,
            "total_bridge_nodes": len(crossing),
            "tied_cuts": tied_cuts,
            "mu_2": mu_2,
            "mu_3": mu_3,
        }

    def duplicate_entities(
        self,
        limit: int = 20,
        name_similarity: float | None = None,
        containment: float | None = None,
    ) -> dict[str, Any]:
        """Entities that are two spellings of one name, as merge candidates.

        `add_document` mints an entity per capitalised token, so one thing arrives as
        `Builder`, `BUILDER` and `Builders`. This proposes such pairs with the
        structural evidence for each; it never merges, since what means the same thing
        is a decision the graph cannot make.

        Names generate the candidates and structure is only the evidence. Ranked by
        structure, a real corpus returns co-occurrence, not duplication: topical
        entities share documents by definition, and most entities are pendants with
        identical one-document neighbourhoods. So two names for one thing that share
        no characters are not found, by design.

        `case` pairs -- one token bar capitalisation -- are certain on the name alone
        and need no structural floor (the shouted form is typically one heading).
        `lexical` pairs must also clear `DUPLICATE_CONTAINMENT`.
        """
        undirected = self.graph.to_undirected(as_view=True)
        if undirected.number_of_nodes() == 0:
            return {"verdict": "no_graph", "note": "The graph is empty.", "pairs": []}

        entities = [
            node for node, attrs in self.graph.nodes(data=True)
            if attrs.get("type") == "entity"
        ]
        if len(entities) < 2:
            return {
                "verdict": "no_graph",
                "note": "Too few entities to compare.",
                "pairs": [],
            }

        min_name = (
            DUPLICATE_NAME_SIMILARITY if name_similarity is None else float(name_similarity)
        )
        min_containment = (
            DUPLICATE_CONTAINMENT if containment is None else float(containment)
        )

        # Blocked by a case-folded prefix, so this stays near-linear as the
        # corpus grows.
        blocks: dict[str, list[str]] = {}
        for entity in entities:
            blocks.setdefault(str(entity).lower()[:DUPLICATE_BLOCK_PREFIX], []).append(
                str(entity)
            )

        neighbours = {
            entity: set(undirected.neighbors(entity)) for entity in entities
        }

        pairs: list[dict[str, Any]] = []
        comparisons = 0
        for block in blocks.values():
            for left, right in combinations(sorted(block), 2):
                comparisons += 1
                same_token = left.lower() == right.lower()
                similarity = (
                    1.0 if same_token
                    else SequenceMatcher(None, left.lower(), right.lower()).ratio()
                )
                if similarity < min_name:
                    continue

                here, there = neighbours[left], neighbours[right]
                if not here or not there:
                    continue
                shared = here & there
                overlap = len(shared) / min(len(here), len(there))

                # A case variant needs no corroboration; a likeness does.
                if not same_token and overlap < min_containment:
                    continue

                pairs.append(
                    {
                        "entities": sorted([left, right]),
                        "kind": "case" if same_token else "lexical",
                        "name_similarity": similarity,
                        "shared_documents": len(shared),
                        "containment": overlap,
                        # Jaccard beside containment: the two disagreeing is
                        # what shows the asymmetry.
                        "neighbourhood_overlap": (
                            len(shared) / len(here | there) if (here | there) else 0.0
                        ),
                        "degrees": [len(here), len(there)],
                    }
                )

        # Certain before likely, then by how alike the names read, then by the
        # evidence behind the pair.
        pairs.sort(
            key=lambda pair: (
                pair["kind"] != "case",
                -pair["name_similarity"],
                -pair["shared_documents"],
                pair["entities"],
            )
        )
        return {
            "verdict": "scanned",
            "pairs": pairs[:limit],
            "total_pairs": len(pairs),
            "entities_compared": len(entities),
            "comparisons": comparisons,
            "name_similarity": min_name,
            "containment": min_containment,
        }

