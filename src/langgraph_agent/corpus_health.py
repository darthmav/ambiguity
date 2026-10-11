"""Is the corpus still the archive? The comparison nothing else makes.

A corpus that has drifted from the files it is built from has no symptom of
its own: every counter stays non-zero and consistent while retrieval answers
from what the archive used to hold. This compares the two:

* `missing` -- in the walk, not in the corpus: written since the last rebuild.
* `extra` -- in the corpus, not in the walk: deleted, renamed or no longer
  walked, and still answering searches.
* `oversized` -- in the walk, too large to index, so correctly absent. Kept
  apart from `missing` because a rebuild cannot change it.
* `unreadable` -- in the walk, but not UTF-8 text, which the indexer skips;
  kept apart for the same reason.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langgraph_agent.graphrag_server import MAX_INDEXABLE_BYTES, iter_corpus_files

# How long a walk is reused. The header polls every five seconds, and files do
# not appear at that rate.
WALK_CACHE_SECONDS = 30.0

# At most this many names travel with a report: the counts are the signal, the
# names show what kind of thing is involved.
STALENESS_SAMPLE = 12

_walk_cache: dict[str, tuple[float, frozenset[str], tuple[str, ...]]] = {}
# `_indexer_can_read`'s verdicts, keyed by path and good while the file's
# (mtime, size) holds: a poll over an unchanged archive stats and never reads.
# Pruned to the walk whenever it is taken afresh.
_read_verdicts: dict[str, tuple[tuple[int, int], bool]] = {}
# `archive_size`'s answer, good for as long as the walk it was counted from.
_archive_counts: dict[str, tuple[float, int]] = {}


def _walk(root: str, use_cache: bool) -> tuple[frozenset[str], tuple[str, ...]]:
    """Split the walk into what a rebuild would index and what it would skip.

    The indexer limits *characters* and `stat` counts *bytes*, and a UTF-8 file
    is never fewer bytes than characters -- so a size under the limit proves a
    file indexable without reading it, and only the few files above it are read
    and measured exactly. A guess either way would report a wrong `extra` or
    `missing`.
    """
    if use_cache:
        cached = _walk_cache.get(root)
        if cached is not None and time.monotonic() - cached[0] < WALK_CACHE_SECONDS:
            return cached[1], cached[2]

    indexable: set[str] = set()
    oversized: list[str] = []
    for path in iter_corpus_files(root):
        try:
            size = path.stat().st_size
        except OSError:
            # Gone between the walk and the stat.
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

    for gone in _read_verdicts.keys() - indexable:
        del _read_verdicts[gone]
    answer = (frozenset(indexable), tuple(sorted(oversized)))
    _walk_cache[root] = (time.monotonic(), answer[0], answer[1])
    return answer


def archive_size(root: str = ".") -> int:
    """How many files a rebuild would index: the walk less the files that are
    not UTF-8 text, which the indexer skips -- `corpus_staleness`'s `archive`
    for a corpus that holds nothing.

    What tells "nothing indexed yet" from "nothing to index": on a fresh machine
    a restart or a run rebuilds from an archive that holds nothing, and saying
    either would help sent the operator round that loop. Counted with the
    unreadable files, an archive of Latin-1 notes promised a rebuild that then
    indexed nothing.
    """
    _walk(root, True)
    stamp = _walk_cache[root][0]
    cached = _archive_counts.get(root)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    count = int(corpus_staleness((), root)["archive"])
    _archive_counts[root] = (stamp, count)
    return count


def forget_cached_walk() -> None:
    """Drop the cached walk, so the next answer is taken fresh.

    Called by every writer that changes what the walk finds, so the header
    changes the moment someone acts on it.
    """
    _walk_cache.clear()


def _indexer_can_read(path: str) -> bool:
    """Whether the indexer's own `read_text(encoding="utf-8")` succeeds on `path`.

    A file gone since the walk counts as readable: it is the walk's to drop,
    and the next one will.
    """
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    key = (stat.st_mtime_ns, stat.st_size)
    cached = _read_verdicts.get(path)
    if cached is not None and cached[0] == key:
        return cached[1]
    try:
        Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return True
    except (OSError, UnicodeDecodeError):
        readable = False
    else:
        readable = True
    _read_verdicts[path] = (key, readable)
    return readable


def corpus_staleness(
    indexed: Iterable[str], root: str = ".", *, use_cache: bool = True
) -> dict[str, Any]:
    """Compare what the corpus holds against what a rebuild would put there.

    Only files already missing from the corpus are read, to tell `unreadable`
    from `missing` -- usually none, so the header's poll stays a walk and a stat.
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
        # What a rebuild would index: the walk less what the indexer skips.
        "archive": len(want) - len(unreadable),
        "unreadable": unreadable[:STALENESS_SAMPLE],
    }
