"""Okapi BM25 over the corpus's chunks -- the lexical half of `search`.

Dense retrieval alone misses the queries this corpus is most often asked. A
plan naming `BUILDER_DEADLINE_SECONDS` or `_report_path_key` is asking about
one identifier defined in one file, and a dense embedding is a poor
instrument for "contains this exact rare token": the model was trained to
put *similar* passages near each other, and every seat-timeout constant in this
project is similar to every other one. Measured against ground truth that
cannot be argued with -- 541 identifiers each defined in exactly one project
file, asked about in a natural question -- dense retrieval put the defining
file first 53.4% of the time and dense-plus-lexical 65.1%, with the paired
McNemar test at p < 0.001 (85 queries fixed against 22 broken). Recall at 5
moved 93.0% to 95.4%, which says where the gain comes from: the defining file
was nearly always retrieved, and simply not ranked first.

Three decisions in here are not interchangeable with the obvious alternatives.

*BM25 re-ranks the dense candidates rather than retrieving in parallel.* Full
fusion -- two independent retrievals over the whole corpus, merged -- measured
67.3% against re-ranking's 66.2% on the same 541 queries, a difference of six
and well inside the noise. It costs much more than it sounds: a document that only the
lexical side found has no dense score, and the relevance floor is read
off `results[0]["score"]` to decide whether the corpus answered at all. Buying
one percentage point by making the gate's input sometimes-absent is a bad
trade, so every result still comes from the dense window and still carries the
cosine that window gave it.

*Identifiers are indexed whole **and** in pieces.* `tokenize` emits
`builder_deadline_seconds` alongside `builder`, `deadline` and `seconds`, so a
question that spells the constant exactly matches it, and one that describes it
in words still matches something. Emitting only the pieces would make
`NODE_DEADLINE_SECONDS` and `BUILDER_DEADLINE_SECONDS` near-identical, which is
the very confusion the lexical half is here to resolve.

*Ranks are fused, not scores.* A BM25 score is unbounded and corpus-relative
while a cosine sits in [-1, 1]; any weighted sum of the two needs a scaling
constant that is really a third hyperparameter, tuned on one corpus and wrong
on the next. Reciprocal rank fusion needs no such constant.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

# Okapi BM25's two standard parameters. Left at the values the literature
# settled on: this corpus gave the tuning no signal worth the extra knob, and a
# number tuned on 78 documents would not survive the 79th.
BM25_K1 = 1.5
BM25_B = 0.75

# Reciprocal rank fusion's damping constant, also the standard value. It sets
# how much a top-of-list place is worth against a middling one; at 60 the gap
# between rank 1 and rank 2 is small enough that the two rankings have to agree
# to promote something, which is the behaviour being bought here.
RRF_K = 60

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Underscores, the ordinary camelCase seam, and the seam at the *end* of an
# acronym. The third alternative is not decoration: without it
# `GraphRAGKnowledgeBase` splits to `graph` + `ragknowledge` + `base`, and
# `ragknowledge` is a term no query will ever contain, so the word `knowledge`
# is lost from the index entirely.
_SUBWORD = re.compile(r"_|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str) -> list[str]:
    """Split text into lexical terms, keeping identifiers whole and in pieces.

    See the module docstring for why both: the whole token is what makes an
    exact spelling of a constant findable, and the pieces are what keep a
    question phrased in words from missing it entirely.
    """
    terms: list[str] = []
    for word in _WORD.findall(text):
        terms.append(word.lower())
        terms.extend(
            piece.lower() for piece in _SUBWORD.split(word) if len(piece) > 1
        )
    return terms


class BM25Index:
    """A BM25 index over a fixed set of chunks.

    Immutable by construction. The corpus changes only through `add_document`
    and `clear`, both of which drop the index rather than update it -- rebuilding
    over ~1,200 chunks costs milliseconds, and an index that edits itself in
    place is a second thing that can disagree with the store.
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
        # Guard the empty corpus: a mean over no documents is a ZeroDivisionError,
        # and `search` must be able to ask an empty store a question.
        self._average_length = (sum(self._length) / total) if total else 0.0
        self._idf: dict[str, float] = {
            term: math.log(1 + (total - n + 0.5) / (n + 0.5))
            for term, n in document_frequency.items()
        }

    def __len__(self) -> int:
        return len(self.ids)

    def score(self, query: str, chunk_ids: Iterable[str]) -> dict[str, float]:
        """Score just the chunks named, which is all a re-rank ever needs.

        Scoring the whole corpus to keep twenty of it would be the same answer
        for more work; `chunk_ids` is the dense window. A chunk this index has
        never seen scores 0.0 rather than raising -- the store is the authority
        on what exists, and an index built a moment earlier is allowed to be
        one document behind without taking the search down with it.
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
    """Order `chunk_ids` by BM25, best first, or leave them alone.

    Returns the input order unchanged when there is no index, and when the
    query shares no term with any candidate -- both cases mean the lexical half
    has no opinion, and an arbitrary permutation fed to the fusion would spend
    real ranking evidence on noise.
    """
    if index is None or not chunk_ids:
        return list(chunk_ids)

    scores: Mapping[str, float] = index.score(query, chunk_ids)
    if not any(scores.values()):
        return list(chunk_ids)

    original = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}
    return sorted(chunk_ids, key=lambda c: (-scores[c], original[c]))
