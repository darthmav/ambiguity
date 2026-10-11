"""Generated projects: where a run's files land, and whether the corpus reads them.

A run pointed at a project writes under `projects/<name>/` instead of into the
checkout, and that directory stays out of the corpus until the operator opts it
in: whether a finished project is knowledge the Researcher should have is the
operator's decision.

The opt-in is a file on disk (`EMBEDDED_PROJECTS_FILE`), not a write into the
store, because the corpus is a function of what the walk finds and a rebuild
prunes everything else. It is JSON, which the walk never indexes, and it sits
beside the project directories, so a Builder confined to one cannot opt itself
in through `filesystem_write`.

This module imports nothing from the rest of the package, because
`graphrag_server` imports it for the walk.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

# Relative to the project root, like `UPLOADS_DIR`.
PROJECTS_DIR = "projects"

# Which projects the operator has opted into the corpus: a JSON list of names.
EMBEDDED_PROJECTS_FILE = f"{PROJECTS_DIR}/embedded.json"

# Names a project cannot take: the record above and the temporary file it is
# written through sit beside the project directories, and a project named after
# either would collide with it. Compared case-blind, as some filesystems are.
_RESERVED_NAMES = frozenset({
    PurePosixPath(EMBEDDED_PROJECTS_FILE).name,
    f"{PurePosixPath(EMBEDDED_PROJECTS_FILE).name}.tmp",
})

# Directories the walk never enters, wherever they sit under a corpus root:
# version control, virtualenvs, tool caches and build output -- what a generated
# project accumulates without anyone writing it. Matched against whole directory
# names (plus the `.egg-info` suffix), never as substrings, so `rebuild/` is not
# `build/` and a project's own `src/` and `tests/` are indexed like any others.
# Here rather than beside the walk in `graphrag_server`, which imports this
# module, so a project's file count skips what the walk skips.
CORPUS_SKIP_DIRS = frozenset({
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    ".pytest_cache", ".mypy_cache", "build", "dist",
})


def skipped_dir(name: str) -> bool:
    """Whether the walk prunes a directory of this name."""
    return name in CORPUS_SKIP_DIRS or name.endswith(".egg-info")


# A project name is one path component a person would type: no separators, no
# leading dot (a hidden directory the console would never list), no `..`. One
# pattern, in the syntax Python and JavaScript share, since the console checks
# a name with it before sending one (`project_name_rule`).
_PROJECT_NAME = re.compile(r"^(?!.*\.\.)[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

_PROJECT_NAME_RULE = {"pattern": _PROJECT_NAME.pattern, "reserved": sorted(_RESERVED_NAMES)}


def project_name_error(name: Any) -> str | None:
    """Why `name` cannot name a project, or None when it can."""
    if isinstance(name, str) and name.lower() in _RESERVED_NAMES:
        return f"{name!r} is reserved: the record of embedded projects sits under that name."
    if not isinstance(name, str) or not _PROJECT_NAME.match(name):
        return (
            f"{name!r} is not a project name: use letters, digits, '.', '_' or "
            "'-', starting with a letter or digit, at most 64 characters."
        )
    return None


def project_name_rule() -> dict[str, Any]:
    """`project_name_error`'s rule as data, for a client that checks a name first.

    A copy written into the console went without the reserved names, and a
    name it passed was refused only after the goal had gone into the transcript.
    """
    return dict(_PROJECT_NAME_RULE)


def project_dir(name: str) -> str:
    """The project's directory, relative to the root -- what `output_dir` holds."""
    return f"{PROJECTS_DIR}/{name}"


def embedded_projects(root: str | Path = ".") -> set[str]:
    """Names the operator has opted into the corpus.

    A missing or unreadable record means none, never an error: the walk runs on
    every status poll, and a corrupt file errs on the safe side.
    """
    try:
        data = json.loads((Path(root) / EMBEDDED_PROJECTS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    if not isinstance(data, list):
        return set()
    return {n for n in data if isinstance(n, str) and project_name_error(n) is None}


def set_project_embedded(name: str, embedded: bool, root: str | Path = ".") -> set[str]:
    """Opt a project into the corpus, or out of it. Returns the new set.

    Written atomically, since the walk reads this file from other threads.
    """
    error = project_name_error(name)
    if error:
        raise ValueError(error)
    names = embedded_projects(root)
    if embedded:
        names.add(name)
    else:
        names.discard(name)
    path = Path(root) / EMBEDDED_PROJECTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sorted(names), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return names


def held_out_of_corpus(relative: str | Path, embedded: set[str]) -> bool:
    """True for a walked file under `projects/` that the operator has not opted in.

    Matched on the first path component, so `docs/subprojects/` is not held out.
    """
    parts = PurePosixPath(str(relative).replace("\\", "/")).parts
    if not parts or parts[0] != PROJECTS_DIR:
        return False
    # A loose file directly under projects/ belongs to no project, so nothing
    # can opt it in.
    return len(parts) < 3 or parts[1] not in embedded


def list_projects(root: str | Path = ".") -> list[dict[str, Any]]:
    """Every project directory, with how many files it holds and whether it is embedded."""
    base = Path(root) / PROJECTS_DIR
    if not base.is_dir():
        return []
    embedded = embedded_projects(root)
    projects = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or project_name_error(entry.name) is not None:
            continue
        # Pruned as the walk prunes: a project's own .venv or node_modules is
        # tens of thousands of files the corpus will never read.
        files = 0
        for _, subdirs, names in os.walk(entry):
            subdirs[:] = [name for name in subdirs if not skipped_dir(name)]
            files += len(names)
        projects.append({
            "name": entry.name,
            "path": project_dir(entry.name),
            "files": files,
            "embedded": entry.name in embedded,
        })
    return projects
