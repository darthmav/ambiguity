"""Whole-graph spectral analysis of the corpus.

Four questions about the shape of the knowledge graph, kept apart from the
store that holds it. `GraphRAGKnowledgeBase` mixes this in, so every caller
still writes `kb.topics()` and `stats()` still reaches `self.connectivity()`;
nothing about the API moved.

They live here because of a limit they were quietly breaking.
`MAX_INDEXABLE_BYTES` was 100,000 characters then and `graphrag_server` had
grown to 100,132 -- so the module defining the corpus was being **skipped by
every reindex**, silently, and could not be retrieved by the Researcher that reads
this project's own code. It crossed the line twice in one afternoon, once from
adding the staleness check and once from two one-line guards, which is what
makes it structural rather than incidental: any edit to that file was a coin
toss. These four methods are 29.5 KB of it and are a genuinely separate
concern -- the store answers "what is in the corpus", these answer "what shape
is it" -- so moving them buys ~30 KB of headroom and a better seam at the same
time.

Each of the four is documented at its own docstring. What they share is a
refusal to report a finding they cannot support: `topics` returns
`no_clear_structure` rather than invent a `k`, `bottleneck` has a
`certified_none` verdict backed by Cheeger's lower bound and an `inconclusive`
one for the gap between that bound and the sweep, `connectivity` reports
`lambda_2` as `None` rather than 0.0 when there is nothing to measure, and
`duplicate_entities` only ever proposes. The measurements behind each of those
choices are in CLAUDE.md.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from itertools import combinations
from typing import Any

import networkx as nx

# Below this conductance a cut counts as a genuine narrow waist: leaving the
# set, roughly one edge in ten crosses. A convention, not a theorem, and it is
# named here because two things read it -- the verdict below and the console.
# The raw conductance and both Cheeger bounds are always returned alongside it,
# so a caller who wants a different line can draw one.
BOTTLENECK_CONDUCTANCE = 0.1
# How many times larger the winning eigengap must be than the runner-up before
# the number of clusters it implies is worth believing. Not a guess: measured
# in `scripts/spectral_benchmark.py` and again on this graph shape. Across 18
# corpora with a planted topic count the eigengap picked k correctly every
# time, at a decisiveness of 5.1 to 23.4; on graphs with no community structure
# at all -- a grid, a small-world ring, an expander, one dense topic -- it still
# returned some k, at 1.0 to 1.8. Nothing observed lands between 1.8 and 4.5,
# so 3.0 sits in open space rather than on a boundary.
EIGENGAP_DECISIVENESS = 3.0
# The largest k the eigengap is allowed to propose. A whole-corpus map with
# more parts than this is not a map anyone reads, and the heuristic's failures
# in the benchmark were all at the top of its range (k = 10 for a barbell whose
# answer is 2), so the ceiling is also where the bad answers live.
MAX_AUTO_CLUSTERS = 12
# How alike two entity names must read before the pair is worth proposing as a
# merge. Only pairs that are *not* already identical bar their case are held to
# it: `Builder` / `Builders` scores 0.93, `Entity` / `Entities` 0.92, and the
# false positive the old synthetic fixture was built around, `Ent11` / `Ent11x`,
# scores 0.91 -- which is why a lexical pair must clear
# `DUPLICATE_CONTAINMENT` as well, and that is what excludes it.
DUPLICATE_NAME_SIMILARITY = 0.85
# How much of the rarer entity's neighbourhood the commoner one must cover
# before a lexically similar pair is offered. **Containment, not Jaccard**, and
# the difference is the whole reason the old design found nothing: a real
# duplicate is *asymmetric*. `Builder` is mentioned by 29 documents and
# `Builders` by 4, a subset relation that scores containment 1.00 and Jaccard
# 0.14 -- so Jaccard, and the embedding distance built on the same symmetry
# assumption, both rank the true duplicate below thousands of unrelated pairs.
DUPLICATE_CONTAINMENT = 0.5
# Entities are compared only against others sharing this many leading
# characters, case-folded. On this corpus that is 1,528 comparisons instead of
# 948,753 -- 620x fewer -- and it costs nothing this method could otherwise
# find, because a pair that agrees on no prefix cannot clear
# `DUPLICATE_NAME_SIMILARITY` on names of the length the extractor mints.
DUPLICATE_BLOCK_PREFIX = 4


class CorpusSpectralMixin:
    """The four whole-graph diagnostics, mixed into `GraphRAGKnowledgeBase`.

    A mixin rather than free functions taking a knowledge base, so that no call
    site had to change and `stats()` could go on calling `self.connectivity()`.
    """

    # Supplied by the class this is mixed into.
    graph: nx.DiGraph
    _connectivity_cache: tuple[tuple[int, int], dict[str, Any]] | None

    def connectivity(self) -> dict[str, Any]:
        """Structural health of the knowledge graph: components, and lambda_2.

        A reindex that silently drops edges -- an entity-extraction regression
        in `add_document`, say -- does not change the document count and does
        not raise. It shows up here first, as a rising component count or a
        collapsing `lambda_2`, long before it shows up as worse search.

        **Components come from networkx, not from the spectrum.** The textbook
        identity is that the multiplicity of eigenvalue 0 equals the number of
        connected components, and the spectral applicability study proposes
        counting near-zero eigenvalues for exactly that reason. Two measured
        objections, both on this project's own graph shape (920 nodes):

        1. It is 42x the cost of the linear-time answer -- 32ms of
           eigendecomposition against 0.76ms of `number_connected_components`
           -- for a number networkx already computes exactly.
        2. On the *normalized* Laplacian it is simply wrong. The identity holds
           for `L = D - A`; for `I - D^-1/2 A D^-1/2` an isolated node has
           `D^-1/2 = 0`, so the `I` term leaves a bare 1 on its diagonal and it
           contributes eigenvalue **1, not 0**. This graph has 29 isolated
           nodes out of 30 components, so the spectral count returns 1 where
           the truth is 30.

        **lambda_2 is measured on the largest component, and normalized.** Two
        deliberate choices:

        - On the whole graph lambda_2 is identically 0 whenever the corpus is
          disconnected, and it is -- 30 components in the shape measured here.
          A health signal that reads 0.0 every time is not a signal. The
          largest component's lambda_2 is the number that actually moves when
          the body of the corpus knits together or comes apart.
        - Normalized, so it lands in [0, 2] and does not scale with degree.
          The unnormalized lambda_2 grows as documents mention more entities,
          which makes this reindex's value incomparable with last week's --
          and comparing across reindexes is the entire purpose.

        Returns `lambda_2: None` rather than a number when the largest
        component has fewer than two nodes: lambda_2 is undefined there, and 0.0
        would read as "totally disconnected" rather than "nothing to measure".
        """
        # Every edge runs document -> entity, so reversing one reads "entity is
        # mentioned by document" -- the same relation, not a different claim.
        # That is what makes to_undirected() safe to apply on the caller's
        # behalf here, and it is applied explicitly because `spectral_graph`
        # refuses a DiGraph rather than guessing (see `_require_undirected`).
        # Pages the research phase fetched are deliberately entity-free -- see
        # `_is_web_document` -- so every one of them is an isolated node by
        # design. They are excluded here rather than counted, because this
        # method exists to catch an entity-extraction *regression*, which shows
        # up as exactly that symptom and nowhere else. Fifteen permanent
        # false isolates would bury the reading that matters, the same way a
        # staleness verdict that flickers teaches the operator to ignore the one
        # that does not. They are reported separately as `web_documents`, so the
        # exclusion is visible rather than silent.
        # Markup, script and config files are entity-free for the same reason
        # (`ENTITY_FREE_SUFFIXES`) and excluded the same way, but counted apart
        # as `entity_free_sources`, so neither number claims the other's files.
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
                # Imported here, not at module scope. `spectral_graph` lives at
                # the project root and is not part of the installed
                # `langgraph_agent` distribution, so it is importable only when
                # the root is on sys.path -- true for the console and the test
                # suite, false for an MCP server launched from anywhere else. A
                # top-level import would turn a missing diagnostic into a module
                # that will not load at all.
                from spectral_graph import compute_spectrum

                spectrum = compute_spectrum(
                    undirected.subgraph(largest), k=2, normalized=True, which="SM"
                )
                # Clamp solver noise: lambda_1 is 0 and lambda_2 >= 0, so a
                # small negative here is arithmetic, not a finding.
                lambda_2 = max(float(spectrum[1]), 0.0)
            except ImportError:
                unavailable = "spectral_graph is not on sys.path"
            except Exception as exc:  # pragma: no cover - solver-dependent
                # Same posture as the chunk count in `stats()`: this is a
                # diagnostic, and losing it must not cost the console the
                # counters it renders the header from. Named rather than
                # dropped, so "could not measure" never reads as "measured 0".
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

        The A2 application of the spectral applicability study:
        Ng-Jordan-Weiss spectral clustering over the normalized Laplacian,
        which on a bipartite document/entity graph puts documents together with
        the entities that define them -- so each cluster reads as a topic
        rather than as a list of ids. `neighborhood()` shows one node's
        surroundings; this is the whole-corpus map that degree-filtered sweeps
        cannot produce.

        **The number of clusters is where this application was weakest, and it
        is not wired straight to the eigengap.** The report proposes choosing
        `k` from the eigengap; the architecture benchmark
        measured that heuristic getting `k` wrong on 3 of 8 architectures,
        including k = 10 for a barbell whose answer is 2. The heuristic always
        returns *some* k, so on a corpus with no topic structure it invents
        one, and clusters presented without that caveat are a fabricated map.

        What makes it usable is that the failures are not merely wrong, they
        are *undecided*: the winning gap barely beats the runner-up. Measured
        across 18 corpora with a planted topic count the eigengap was correct
        every time at a decisiveness of 5.1-23.4, while a grid, a small-world
        ring, an expander and a single dense topic all landed at 1.0-1.8. Below
        `EIGENGAP_DECISIVENESS` the verdict is `no_clear_structure` and no
        clusters are returned, because a map of a corpus that has no topics is
        worse than no map.

        An explicit `k` skips that gate -- a caller asking for six clusters has
        made the decision -- but the decisiveness is still reported, so the
        answer never hides how much the corpus agreed with it.

        Every cluster carries its own conductance, which is the second and
        independent check: a cluster that is genuinely a community has a low
        one, and a `k` that split a real community in half shows up as several
        clusters with high conductance even when the eigengap looked decisive.
        The two signals catch different failures and neither replaces the other.

        Runs on the largest connected component, for the reason
        `connectivity()` and `bottleneck()` do: components are already clusters,
        so on a disconnected graph the eigenvectors would spend themselves
        rediscovering the orphans `connectivity()` already counted.
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

        # A caller's bad k is an error, not a verdict. `unavailable` means the
        # measurement could not be taken; answering a malformed request with it
        # would file the caller's mistake under the solver's failures.
        if k is not None and not 2 <= k <= n:
            raise ValueError(
                f"k must be between 2 and {n} (the largest component), got {k}"
            )

        try:
            # numpy alongside spectral_graph rather than at module scope: it is
            # used only here, and this module is imported by the MCP server and
            # by every test that touches the corpus.
            import numpy as np

            from spectral_graph import compute_spectrum, conductance, spectral_clustering

            # One eigenvalue past the largest k worth proposing, so the gap that
            # would select that k is itself inside the window.
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
        # Annotated: the value types are heterogeneous, so without this mypy
        # infers a union from the literal and the sort key below stops typing.
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
            # The cluster's name, in effect: its best-connected entities are
            # what the documents in it have in common, which is the thing a
            # reader wants and a list of node ids is not.
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

        The A3 application of the spectral applicability study. Sweeps the
        normalized Fiedler vector for the prefix of lowest conductance, then
        names the nodes whose edges actually cross it -- the few entities or
        documents through which two otherwise separate topic areas connect.
        Those are the terms a search should expand on when a query straddles
        both, and the nodes whose removal would fragment the corpus. Degree
        alone does not find them: a bridge entity mentioned by two documents
        has degree 2, which is unremarkable everywhere else in the graph.

        **The verdict has three states, not two, and the middle one is the
        reason this is worth building.** A minimisation always returns
        *something*: ask for the narrowest cut in a perfectly well-knit corpus
        and you get one anyway, and reporting it as a bridge would be a
        fabricated finding of exactly the kind `search` was fixed for. What
        separates them is Cheeger's lower bound, `mu_2 / 2`, which is a proof
        that no cut anywhere in the graph beats it:

        - `certified_none` -- the lower bound is itself above
          `BOTTLENECK_CONDUCTANCE`, so no narrow waist exists *anywhere*. This
          is a theorem about the whole graph, not a statement about the cut
          that was found, and no amount of searching would turn one up.
        - `found` -- the sweep cut came in at or below the line. The bridge
          nodes below are real.
        - `inconclusive` -- the bound permits a bottleneck and the sweep cut did
          not find one. Cheeger brackets the true conductance between
          `mu_2 / 2` and `sqrt(2 * mu_2)`, and that bracket is wide (from 4x
          to 546x, measured across the benchmark's eight architectures), so
          the sweep cut genuinely can miss. Saying so is the honest answer; collapsing it
          into "no bottleneck" would report a gap in the evidence as a finding.

        Runs on the largest connected component, for the same reason
        `connectivity()` measures `lambda_2` there: on a disconnected graph the
        Fiedler vector is a component indicator, so the sweep cut returns one
        component against the rest at conductance 0. That is a true answer to a
        question nobody asked -- "your corpus has an orphan" is what
        `connectivity()` is for, and it would crowd out the real bridge every
        time.
        """
        # Same modelling note as `connectivity()`: every edge runs
        # document -> entity, so reversing one reads "entity is mentioned by
        # document" -- the same relation, not a different claim.
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
            # mu_3 as well as mu_2, to detect a tie -- see `tied_cuts` below.
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

        # `mu_2 ~= mu_3` means the graph has more than two topic areas, and the
        # Fiedler vector picks one of several equally-narrow cuts arbitrarily.
        # Worth reporting rather than hiding: running this twice on such a
        # corpus returns different *sides* -- measured 99/198 and 97/200 on
        # alternating runs of the same three-topic graph -- while the
        # conductance (0.008264, all 12 runs) and the bridge entities
        # (BRIDGE0/BRIDGE1, all 12 runs) stay put. An operator who sees the
        # split move and has not been told why will read a working diagnostic
        # as a broken one. It is also a real finding in its own right: a tie
        # says there are three or more areas here, not two.
        mu_2, mu_3 = float(spectrum[1]), float(spectrum[2])
        tied_cuts = bool(mu_3 - mu_2 <= 0.1 * mu_3) if mu_3 > 1e-12 else False

        # The nodes carrying the cut, ranked by how much of it they carry. A
        # node's crossing count is what makes it a bridge; its total degree is
        # reported beside it because the two coming apart is the whole point --
        # a bridge is a node whose few edges happen to be the load-bearing ones.
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

        `add_document` mints an entity per capitalised token, so the same thing
        arrives under several names: `Builder` and `BUILDER` from a heading,
        `Entity` and `Entities` from a plural, `Builder` and `Builders`. This
        proposes those pairs and reports the structural evidence for each. It
        never merges -- which entities mean the same thing is a decision about
        meaning that the graph cannot make.

        **This used to rank candidates by distance in a spectral embedding, and
        the measurement retired that outright.** On this project's own corpus
        it produced 33,060 candidates of which **67% sat at distance exactly
        0.0000 with a neighbourhood overlap of 1.00** -- the docstring's own
        "strongest merge evidence there is" -- and the top of the list read
        `['LEGAL', 'Virginia']`, `['Canada', 'Professional']`,
        `['Consequences', 'PIPEDA']`. Those are pendant collisions: two
        entities each mentioned by exactly one document, the same one, are
        structurally identical by construction, and 60% of this corpus's
        entities have degree 1. Meanwhile the true duplicates -- `Builder` /
        `Builders`, `Entity` / `Entities`, and all thirty case variants --
        were **not candidates at any rank**. Graded against ground truth the
        spectral ranking scored 0% precision and 0% recall; so did Jaccard, and
        so did containment used as a ranker. Name similarity scored 100%
        precision on its top 20.

        Two things went wrong and only one of them is the pendants.

        *A real duplicate is asymmetric.* `Builder` is mentioned by 29
        documents and `Builders` by 4. That is a subset, not a match, and both
        the embedding distance and the Jaccard overlap are built on symmetry --
        they score the pair 0.14 and rank it below thousands of unrelated ones.
        Containment (`shared / min(degree)`) reads 1.00 on the same pair, and
        is what the evidence here is measured with.

        *And structure cannot generate candidates on a real corpus at all.*
        Tightening it does not help: at Jaccard 1.00 with at least three shared
        documents, the survivors on this corpus are `Oppenheim` / `Schafer`
        (two authors cited in the same three papers), `Nyquist` / `Frequency`,
        and `BUILDER_DEADLINE_SECONDS` / `NODE_DEADLINE_SECONDS`. Every one is
        co-occurrence, not duplication. The synthetic corpus that once
        justified the structural signal assigned entities to documents **at
        random**, which makes an identical neighbourhood astronomically
        improbable and therefore strong evidence. Real corpora are the opposite:
        entities belonging to one topic are mentioned in the same documents --
        that is what a topic *is* -- so identical neighbourhoods are ordinary
        and mean "discussed together". The property the structural test depended
        on is precisely the property a real corpus does not have.

        So names generate the candidates and structure is the evidence, which
        inverts the old docstring's "structure beats names here and names
        actively mislead". That claim was true of the fixture and false of the
        corpus.

        **What this gives up, explicitly: two names for one thing that share no
        characters.** `LanguageModel` for an entity already called something
        else is not found and cannot be, and nothing here should be read as
        looking for it. That case is not merely unimplemented -- it was
        measured, and on real data every method that reaches for it returns
        collocations instead. Anyone reinstating a structural generator should
        re-run that measurement first.

        Pairs come in two kinds, and the difference is how much they need to
        prove. `case` -- the two names are the same token bar capitalisation --
        is certain on the name alone and carries no structural requirement,
        which matters because these are the most asymmetric pairs in the corpus
        (`BUILDER` appears in one document, `Builder` in 29) and any evidence
        floor would drop every one of them. `lexical` is a likeness rather than
        a certainty, so it must also clear `DUPLICATE_CONTAINMENT`; that is
        what separates `Builder` / `Builders` from two merely similar names for
        different things.
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

        # Blocked by a case-folded prefix so this stays linear in practice. An
        # all-pairs scan is 948,753 comparisons on this corpus against 1,528
        # here, and allocates nothing quadratic as the corpus grows.
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

                # A case variant is the same token and needs no corroboration;
                # a mere likeness does. Holding both to the same floor would
                # drop every case variant in this corpus, since the shouted
                # form is typically a single heading in a single document.
                if not same_token and overlap < min_containment:
                    continue

                pairs.append(
                    {
                        "entities": sorted([left, right]),
                        "kind": "case" if same_token else "lexical",
                        "name_similarity": similarity,
                        "shared_documents": len(shared),
                        "containment": overlap,
                        # Jaccard, kept beside containment rather than instead
                        # of it: it is the number that reads low on a real
                        # duplicate, and seeing the two disagree is what shows
                        # the asymmetry rather than hiding it.
                        "neighbourhood_overlap": (
                            len(shared) / len(here | there) if (here | there) else 0.0
                        ),
                        "degrees": [len(here), len(there)],
                    }
                )

        # Certain before likely, then by how alike the names read, then by how
        # much evidence stands behind the pair.
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

