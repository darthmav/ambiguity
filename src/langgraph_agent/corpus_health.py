"""Is the corpus still the project? The comparison nothing else makes.

This module exists because of a failure that had no symptom. The corpus is a
function of what is on disk, `index_project_files` is the only thing that
rebuilds it, and nothing calls that on its own -- so a project that grew past
its last reindex retrieves against the corpus it had rather than the one it
has. Measured on this repository on 2026-09-09: the store held **8 documents,
all of them uploads, against a walk offering 103**. Every project query
therefore scored under the relevance floor, `_gather_research`
discarded retrieval and fell through to the Researcher's seat on every single
run, and the step-burning loop that causes is already written up in CLAUDE.md.

Nothing reported any of it. `rag_stats` said `indexed`. The counters were
non-zero and consistent with each other. The full test suite passed. The only
way to see it was to compare two numbers that nobody had ever put side by
side, which is what this does.

It lives apart from `graphrag_server` rather than inside it for a reason worth
recording: the code was written there first and pushed that module from 98,920
characters to 104,582, past the 100,000-character `MAX_INDEXABLE_BYTES` of the
time -- so the file that defines the corpus would have been dropped from the
corpus by the next reindex, in silence, as the direct result of adding the
check meant to catch exactly that. That the limit, rather than the subject
matter, decided where this module begins is why it was eventually raised; see
`MAX_INDEXABLE_BYTES`. The split stands on its own merits and `oversized`
below still reports a file the walk offers and the indexer must skip.

Three states, and they are not the same request:

* `missing` -- in the walk, not in the corpus. Work written since the last
  reindex. Press Reindex.
* `extra` -- in the corpus, not in the walk. A file deleted, renamed or newly
  excluded, whose text is nowhere in the project and still answers searches.
  Press Reindex.
* `oversized` -- in the walk, too large to index, correctly absent from the
  corpus. Reindexing will not help; the file has to be split, or the limit
  raised. Reported separately *because* it is not stale: folding it into
  `missing` would leave the header permanently asking for a rebuild that
  cannot change anything, and a permanent warning is one nobody reads.
* `unreadable` -- in the walk, but not UTF-8 text (or not readable at all),
  which the indexer skips. The same kind of permanent absence as `oversized`,
  for the same reason reported apart from `missing`: one Latin-1 notes file
  used to hold the header at "stale: 1 not indexed" through every rebuild.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langgraph_agent.graphrag_server import MAX_INDEXABLE_BYTES, iter_project_files

# How long a walk of the tree is reused before being taken again. The console
# polls the header every five seconds and the walk is ~19ms over this project:
# affordable, and pointless to repeat that often, since files do not appear at
# that rate and the answer is advisory.
WALK_CACHE_SECONDS = 30.0

# At most this many names travel with a report. The counts are the signal; the
# names are there so the operator can see what *kind* of thing is involved
# without opening a shell.
STALENESS_SAMPLE = 12

_walk_cache: dict[str, tuple[float, frozenset[str], tuple[str, ...]]] = {}


def _walk(root: str, use_cache: bool) -> tuple[frozenset[str], tuple[str, ...]]:
    """Split the walk into what a reindex would take and what it would skip.

    Size is resolved exactly, in two steps, because every field this feeds is
    an accusation and one standing false accusation would teach the operator to
    ignore the whole signal. `index_project_files` measures *characters*;
    `stat` counts *bytes*; a UTF-8 file is never fewer bytes than characters.
    So `st_size <= MAX_INDEXABLE_BYTES` already proves a file indexable and
    settles almost every file without a read. Only a file above that line is
    genuinely ambiguous, and there are a handful of those at most, so they are
    read and measured properly rather than guessed at. Guessing either way
    invents something: excluding an indexable file calls a document that
    belongs in the corpus `extra`, and including a skipped one calls a document
    the indexer is right to omit `missing`.
    """
    if use_cache:
        cached = _walk_cache.get(root)
        if cached is not None and time.monotonic() - cached[0] < WALK_CACHE_SECONDS:
            return cached[1], cached[2]

    indexable: set[str] = set()
    oversized: list[str] = []
    for path in iter_project_files(root):
        try:
            size = path.stat().st_size
        except OSError:
            # Gone between the glob and the stat; not something to expect.
            continue
        if size <= MAX_INDEXABLE_BYTES:
            indexable.add(str(path))
            continue
        try:
            if len(path.read_text(encoding="utf-8")) <= MAX_INDEXABLE_BYTES:
                indexable.add(str(path))
            else:
                oversized.append(str(path))
        except (OSError, UnicodeDecodeError):
            # Unreadable is what the indexer would hit too, and it skips.
            continue

    answer = (frozenset(indexable), tuple(sorted(oversized)))
    _walk_cache[root] = (time.monotonic(), answer[0], answer[1])
    return answer


def expected_documents(root: str = ".", *, use_cache: bool = True) -> frozenset[str]:
    """The documents a reindex would put in the corpus, as it would name them."""
    return _walk(root, use_cache)[0]


def oversized_documents(root: str = ".", *, use_cache: bool = True) -> tuple[str, ...]:
    """Files the walk offers that are too large to index, so are absent by design."""
    return _walk(root, use_cache)[1]


def forget_expected_documents() -> None:
    """Drop the cached walk, so the next answer is taken fresh.

    Called by the one writer that changes what the walk would find. The report
    has to change the moment someone acts on it: a header still saying `stale`
    half a minute after the fact is the same credibility problem the exact size
    test above is guarding against.
    """
    _walk_cache.clear()


def _indexer_can_read(path: str) -> bool:
    """Whether the indexer's own `read_text(encoding="utf-8")` succeeds on `path`.

    A file gone since the walk counts as readable: it is the walk's to drop,
    and the next one will.
    """
    try:
        Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return True
    except (OSError, UnicodeDecodeError):
        return False
    return True


def corpus_staleness(
    indexed: Iterable[str], root: str = ".", *, use_cache: bool = True
) -> dict[str, Any]:
    """Compare what the corpus holds against what a reindex would put there.

    See the module docstring for the failure this exists to catch, and for why
    `oversized` and `unreadable` are reported beside `stale` rather than
    inside it. Only files already missing from the corpus are read to tell the
    two apart -- usually none, so the header's poll stays a walk and a stat.
    """
    have = {str(document) for document in indexed}
    want, oversized = _walk(root, use_cache)

    absent = sorted(want - have)
    unreadable = [path for path in absent if not _indexer_can_read(path)]
    skipped = set(unreadable)
    missing = [path for path in absent if path not in skipped]
    extra = sorted(have - want)
    return {
        "stale": bool(missing or extra),
        "indexed": len(have),
        "expected": len(want),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "missing": missing[:STALENESS_SAMPLE],
        "extra": extra[:STALENESS_SAMPLE],
        # Not staleness: correctly absent, and a rebuild will not change it.
        "oversized_count": len(oversized),
        "oversized": list(oversized[:STALENESS_SAMPLE]),
        "unreadable_count": len(unreadable),
        "unreadable": unreadable[:STALENESS_SAMPLE],
    }
