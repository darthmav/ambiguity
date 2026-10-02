"""Okapi BM25 over the corpus's chunks -- the lexical half of `search`.

A dense embedding is a poor instrument for "contains this exact rare token":
asked about 541 identifiers each defined in one file, dense retrieval ranked
the defining file first 53.4% of the time and dense-plus-lexical 65.1%.

- *BM25 re-ranks the dense window rather than retrieving in parallel.* Full
  fusion measured within the noise of re-ranking, and would leave a lexical-only
  hit with no cosine -- the score the relevance floor is read off.
- *Identifiers are indexed whole and in pieces*, so a question spelling a
  constant exactly matches it and one describing it in words still matches.
- *Ranks are fused, not scores*: a BM25 score is unbounded and corpus-relative,
  so any weighted sum with a cosine needs a constant tuned on one corpus.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

# Okapi BM25's two standard parameters, at the literature's values: tuning them
# on a corpus this small would not survive the next document.
BM25_K1 = 1.5
BM25_B = 0.75

# Reciprocal rank fusion's damping constant, the standard value: at 60 the two
# rankings have to agree to promote something.
RRF_K = 60

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Underscores, the camelCase seam, and the seam at the end of an acronym --
# without the last, `GraphRAGKnowledgeBase` yields `ragknowledge`, a term no
# query contains.
_SUBWORD = re.compile(r"_|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str) -> list[str]:
    """Split text into lexical terms, keeping identifiers whole and in pieces."""
    terms: list[str] = []
    for word in _WORD.findall(text):
        terms.append(word.lower())
        terms.extend(
            piece.lower() for piece in _SUBWORD.split(word) if len(piece) > 1
        )
    return terms


class BM25Index:
    """A BM25 index over a fixed set of chunks.

    Immutable: a corpus change drops the index rather than editing it, since a
    rebuild costs milliseconds and an index that edits itself is a second thing
    that can disagree with the store.
    """

    def __init__(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        if len(ids) != len(texts):
            raise ValueError(
                f"ids and texts must line up: {len(ids)} ids, {len(texts)} texts"
            )
        self.ids: list[str] = list(ids)
        self._tf: list[Counter[str]] = []
        self._length: list[int] = []
        document_frequency: Counter[str] = Counter()

        for text in texts:
            terms = tokenize(text)
            counts = Counter(terms)
            self._tf.append(counts)
            self._length.append(len(terms))
            document_frequency.update(counts.keys())

        self._position = {chunk_id: i for i, chunk_id in enumerate(self.ids)}
        total = len(self._tf)
        # An empty store must still answer a question.
        self._average_length = (sum(self._length) / total) if total else 0.0
        self._idf: dict[str, float] = {
            term: math.log(1 + (total - n + 0.5) / (n + 0.5))
            for term, n in document_frequency.items()
        }

    def __len__(self) -> int:
        return len(self.ids)

    def score(self, query: str, chunk_ids: Iterable[str]) -> dict[str, float]:
        """Score just the chunks named -- the dense window a re-rank orders.

        A chunk this index has never seen scores 0.0 rather than raising: an
        index built a moment earlier may be one document behind the store.
        """
        terms = [term for term in tokenize(query) if term in self._idf]
        scores: dict[str, float] = {}

        for chunk_id in chunk_ids:
            position = self._position.get(chunk_id)
            if position is None:
                scores[chunk_id] = 0.0
                continue

            counts = self._tf[position]
            length = self._length[position]
            total = 0.0
            for term in terms:
                frequency = counts.get(term)
                if not frequency:
                    continue
                denominator = frequency + BM25_K1 * (
                    1 - BM25_B + BM25_B * length / (self._average_length or 1.0)
                )
                total += self._idf[term] * frequency * (BM25_K1 + 1) / denominator
            scores[chunk_id] = total

        return scores


def reciprocal_rank_fusion(
    *rankings: Sequence[str], k: int = RRF_K
) -> list[str]:
    """Merge rankings by summing 1/(k + rank), best first.

    Ties keep the order of the first ranking that contains the item, so a
    lexical half with nothing to say -- every score zero, every rank arbitrary
    -- cannot reshuffle a dense ordering it does not disagree with.
    """
    points: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            points[item] = points.get(item, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(item, len(first_seen))
    return sorted(points, key=lambda item: (-points[item], first_seen[item]))


def lexical_order(
    query: str, chunk_ids: Sequence[str], index: BM25Index | None
) -> list[str]:
    """Order `chunk_ids` by BM25, best first.

    The input order comes back unchanged when there is no index or the query
    shares no term with any candidate: the lexical half has no opinion, and an
    arbitrary permutation would feed the fusion noise.
    """
    if index is None or not chunk_ids:
        return list(chunk_ids)

    scores: Mapping[str, float] = index.score(query, chunk_ids)
    if not any(scores.values()):
        return list(chunk_ids)

    original = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}
    return sorted(chunk_ids, key=lambda c: (-scores[c], original[c]))
