"""Documentation claims, made executable.

Prose goes stale silently. On 2026-09-09 this project found four claims that
had been true when written and had quietly stopped being true: a comment
saying `graphrag_server.py` sat ~250 characters under the indexing limit when
it had 28,000 to spare, a seat diagnostic quoting a relevance floor of 0.40
when the constant read 0.37, a note asserting the twenty best-connected
entities were all real terms when three were sentence-openers, and twelve
citations pointing at reports deleted the same day.

Every one was found by a person reading carefully. That does not scale and it
does not repeat, which is why each of them survived being read many times
before someone measured it. The rule this file enforces instead:

    a claim worth writing down is a claim worth failing a build over.

Two kinds of guard live here. Most **recompute** the claim from the repository,
so they need no maintenance and catch drift nobody anticipated. A few compare
prose against a constant, and those carry a registry that has to be extended
when a new figure is written into the docs -- the honest architectural answer
for a figure is not to restate it at all but to read it, the way
`scripts/diagnose_seats.py` reads `relevance_floor()` from the corpus's stored
record rather than quoting it. Markdown cannot do that, so markdown gets a
registry.

These tests read the real repository rather than a fixture. That is the point:
a fixture would test the checker.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path, PurePosixPath

import pytest

from langgraph_agent.config import DEFAULT_SEATS
from langgraph_agent.graphrag_server import (
    EMBEDDING_BATCH_SIZE,
    ENTITY_STOPWORDS,
    MAX_INDEXABLE_BYTES,
    UPLOADS_DIR,
    _mints_entities,
    iter_project_files,
)
from langgraph_agent.projects import PROJECTS_DIR

ROOT = Path(__file__).resolve().parent.parent

# Documents that make claims about the project. Anything cited here has to be
# real, and anything asserted here has to be true.
PROSE_FILES = ("CLAUDE.md", "README.md")

# Paths that legitimately do not exist on a clean checkout: every one is a
# runtime artifact created on demand, and a bone-stock machine has none of
# them. `knowledge/` in particular must NOT exist until someone indexes --
# that rule has its own tests in test_corpus_absent.py.
RUNTIME_PATHS = (
    "knowledge/", "runs/", "uploads/", "research/web/", "reports/diagnostics/",
    "projects/",
)

# Artifacts named in prose that are written at runtime and never committed.
RUNTIME_NAMES = frozenset({
    "knowledge_graph.json", "last_run.json", "report.md", "results.json",
    "floor_calibration.json",
})

# Filenames the seat diagnostic *asks an agent to create*. They are goals, not
# citations: a checkout that contains them has a stray build artifact.
EXERCISE_ARTIFACTS = frozenset({"slugify_tool.py", "retry_helper.py"})

# Deleted files that the docs name *as deleted*. Recording what was removed and
# why is the opposite of a dangling citation. Listed by name rather than
# detected by pattern for the reason `ENTITY_STOPWORDS` is a list and not a
# rule: a pattern loose enough to excuse these would also excuse a real one,
# and its failures would be invisible. A reader can audit this.
HISTORICAL_PATHS = frozenset({
    # the spectral write-ups, removed 2026-09-09
    "reports/spectral_applicability.md",
    "reports/spectral_architecture_benchmark.md",
    # the dead run artifacts and the withdrawn line of work, same cleanup
    "AUDIT_REPORT.md", "CLARIFICATION_NEEDED.md", "run_exercise.py",
    "ai_efficiency_report.md", "docs/legal-boundaries.md",
    "docs/research/bot-detection-techniques.md",
    "reports/security-circumvention-blocker.md",
})

PATH_LIKE = re.compile(
    r"^[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|md|txt|sh|html|json|toml|ini|yml|cfg)$"
)
BACKTICKED = re.compile(r"`([^`\n]+)`")


def _cited_paths(text: str) -> set[str]:
    """Backticked spans that are paths. Backticks because that is how this
    project writes a path, and restricting to them keeps prose like "3.10+"
    and glob patterns out of the result."""
    return {m for m in BACKTICKED.findall(text) if PATH_LIKE.match(m)}


def _cited_path_exists(cited: str) -> bool:
    """Whether a cited path resolves.

    A path with a directory in it is exact. A bare basename is not sloppiness
    but this project's convention -- `nodes.py` means the nodes module, and
    spelling out `src/langgraph_agent/nodes.py` every time would bury the
    prose -- so it resolves against any file of that name in the tree. A
    deleted file's basename is nowhere, which is the case worth catching.
    """
    if "/" in cited:
        return (ROOT / cited).exists()
    return any(p.name == cited for p in ROOT.rglob(cited)
               if ".git" not in p.parts and "__pycache__" not in p.parts)


def _excused(cited: str) -> bool:
    return (cited in HISTORICAL_PATHS
            or cited in EXERCISE_ARTIFACTS
            or Path(cited).name in RUNTIME_NAMES
            or cited.startswith(RUNTIME_PATHS))


def _source_files() -> list[Path]:
    """Every Python file a reader would call part of this project.

    The root is a glob, not `serve.py` by name. Naming one file left the class
    it belongs to uncovered: `example_usage.py`, `test_cloud.py` and
    `ollama_client.py` are root-level modules too, and the last of those arrived
    carrying a strict-mode error, absent from CLAUDE.md's tree and named in
    neither CI check list -- invisible to all three guards below at once, on the
    one path where a hand-written file escapes `_lint_written_files` as well.

    `spectral_graph/` is in for the same reason and not because it is installed:
    it is importable only with the project root on `sys.path`, which is exactly
    why a reader needs the tree to know it is there.
    """
    return sorted(
        list((ROOT / "src").rglob("*.py"))
        + list((ROOT / "tests").glob("*.py"))
        + list((ROOT / "scripts").glob("*.py"))
        + list((ROOT / "spectral_graph").rglob("*.py"))
        + list(ROOT.glob("*.py"))
    )


# ---------------------------------------------------------------------------
# citations
# ---------------------------------------------------------------------------


def test_every_path_the_docs_cite_exists():
    """Twelve citations pointed at deleted reports and nothing noticed.

    A path in prose is a promise that a reader can go and look. This recomputes
    the promise, so a file deleted or renamed tomorrow fails here rather than
    being discovered by someone who went looking and found nothing.
    """
    dangling: list[str] = []
    for name in PROSE_FILES:
        for cited in _cited_paths((ROOT / name).read_text(encoding="utf-8")):
            if _excused(cited) or _cited_path_exists(cited):
                continue
            dangling.append(f"{name} cites {cited}")

    assert not dangling, "\n".join(dangling)


def test_every_path_the_source_cites_exists():
    """The same rule one level down. Five of the twelve dangling citations were
    in docstrings, where they are read by whoever is changing that code.

    Narrower than the prose rule in two ways, both deliberate. Only citations
    carrying a directory are checked: a bare `foo.py` in a docstring is an
    *example*, not a promise about a location, and a rule that cannot tell
    those apart fails on placeholder names until someone deletes the rule.
    And `tests/` is not scanned at all -- naming files that do not exist is
    what a test fixture does (`spectral_graph/imaginary.py` proves an import
    error is raised), so every hit there would be a false one.
    """
    dangling: list[str] = []
    for path in _source_files():
        if path.is_relative_to(ROOT / "tests"):
            continue
        for cited in _cited_paths(path.read_text(encoding="utf-8")):
            if "/" not in cited or _excused(cited) or _cited_path_exists(cited):
                continue
            dangling.append(f"{path.relative_to(ROOT)} cites {cited}")

    assert not dangling, "\n".join(dangling)


# ---------------------------------------------------------------------------
# the structure tree
# ---------------------------------------------------------------------------


def _structure_tree() -> set[str]:
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    block = re.search(r"## Project Structure\n+```\n(.*?)```", text, re.S)
    assert block, "CLAUDE.md has no Project Structure block"

    listed: set[str] = set()
    stack: dict[int, str] = {}
    for line in block.group(1).splitlines():
        m = re.match(r"^(.*?)(?:├──|└──)\s*(\S+)", line)
        if not m:
            continue
        depth = len(m.group(1)) // 4
        stack[depth] = m.group(2).rstrip("/")
        listed.add("/".join(stack[d] for d in range(depth + 1) if d in stack))
    return listed


def test_the_structure_tree_lists_nothing_that_is_gone():
    missing = sorted(p for p in _structure_tree() if "." in Path(p).name
                     and not (ROOT / p).exists())
    assert not missing, f"CLAUDE.md's tree lists files that do not exist: {missing}"


def test_the_structure_tree_lists_every_module_test_and_script():
    """The reverse direction, which is the one that actually rots.

    A tree is only useful if it is complete: a reader takes it as the map of
    the project, so a module missing from it is a module they will not know to
    look at. `__init__.py` is exempt -- the tree shows packages by their
    directory, and listing four one-line files would obscure rather than help.
    """
    on_disk = {
        str(p.relative_to(ROOT))
        for p in _source_files()
        if p.name != "__init__.py" and "__pycache__" not in str(p)
    }
    on_disk |= {str(p.relative_to(ROOT)) for p in (ROOT / "scripts").glob("*.sh")}
    # Root-level shell scripts are the project's entry points -- the installer
    # and the console launcher -- so a new one belongs in the tree as much as a
    # module does. `launch_console.sh` was missing from it, which is what this
    # line found.
    on_disk |= {str(p.relative_to(ROOT)) for p in ROOT.glob("*.sh")}
    unlisted = sorted(on_disk - _structure_tree())
    assert not unlisted, f"real files missing from CLAUDE.md's tree: {unlisted}"


# ---------------------------------------------------------------------------
# figures quoted in prose
# ---------------------------------------------------------------------------


def _requires_python_floor() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["requires-python"].lstrip(">=~^ ")


def test_the_documented_python_version_matches_pyproject():
    """CLAUDE.md said "Python 3.10+" while pyproject required >= 3.12.

    Not cosmetic: it is the first line a new contributor reads to decide
    whether their interpreter will do, and being told 3.10 costs them an
    install that cannot resolve.
    """
    claimed = re.search(r"Python (\d+\.\d+)\+", (ROOT / "CLAUDE.md").read_text(encoding="utf-8"))
    assert claimed, "CLAUDE.md no longer states a Python version"
    assert claimed.group(1) == _requires_python_floor(), (
        f"CLAUDE.md claims Python {claimed.group(1)}+, "
        f"pyproject requires {_requires_python_floor()}"
    )


# Figures written into prose, each with the constant that decides them. Extend
# this when a new figure is written down -- or better, do not write the figure
# down: name the constant and let the reader look it up, the way the seat
# diagnostic now does.
DOCUMENTED_FIGURES = (
    (r"`EMBEDDING_BATCH_SIZE` is (\d+)", lambda: EMBEDDING_BATCH_SIZE, "EMBEDDING_BATCH_SIZE"),
)


@pytest.mark.parametrize("pattern,truth,name", DOCUMENTED_FIGURES)
def test_a_figure_quoted_in_prose_matches_its_constant(pattern, truth, name):
    """`scripts/diagnose_seats.py` quoted a floor of 0.40 against a constant of
    0.37, in the one place an operator goes to decide which seat to trust."""
    wrong: list[str] = []
    # This file quotes the *wrong* historical values deliberately -- they are
    # what the tests exist to describe -- so it cannot be scanned for them.
    scanned = [p for p in [ROOT / n for n in PROSE_FILES] + _source_files()
               if p.name != "test_claims.py"]
    for path in scanned:
        for quoted in re.findall(pattern, path.read_text(encoding="utf-8")):
            if float(quoted) != float(truth()):
                wrong.append(f"{path.relative_to(ROOT)} quotes {quoted} for {name} "
                             f"(really {truth()})")

    assert not wrong, "\n".join(wrong)


def test_every_provider_the_seats_can_use_is_a_declared_dependency():
    """A provider a seat can be pointed at must survive `pip install`.

    `langchain-ollama` was undeclared while every seat in `DEFAULT_SEATS` was
    an ollama seat, and nothing failed loudly: a provider that will not import
    makes the seat a `StubLLM`, so a fresh install ran the whole four-agent
    loop on canned text and called it a success. It passed unnoticed on every
    developer machine that happened to have the package, which is why CI found
    it and no local run ever did.

    Read out of `config.py` rather than listed here, so a provider added
    tomorrow is covered without anyone remembering this test exists.
    """
    import tomllib as _tomllib  # noqa: PLC0415 - local to keep the import block small

    config = (ROOT / "src/langgraph_agent/config.py").read_text(encoding="utf-8")
    imported = set(re.findall(r"from (langchain_\w+) import", config))
    assert imported, "config.py imports no provider packages; has the seat wiring moved?"

    declared = _tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = {re.split(r"[<>=!\[]", d)[0].strip().replace("-", "_").lower()
             for d in declared["project"]["dependencies"]}

    missing = sorted(m for m in imported if m.lower() not in names)
    assert not missing, (
        f"config.py imports {missing}, which pyproject.toml does not declare. "
        "A seat pointed at an undeclared provider becomes a StubLLM and the "
        "run reports success on canned text."
    )


def test_the_seat_table_matches_the_shipped_defaults():
    """CLAUDE.md prints a seat/provider/model table. It is the first place
    anyone looks to answer "what runs where", and `DEFAULT_SEATS` is the only
    thing that decides it."""
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    rows = re.findall(r"\|\s*(Architect|Planner|Researcher|Builder)\s*\|"
                      r"\s*(\w+)\s*\|\s*`([^`]+)`\s*\|", text)
    assert len(rows) == len(DEFAULT_SEATS), (
        f"CLAUDE.md's seat table has {len(rows)} rows for {len(DEFAULT_SEATS)} seats"
    )
    for seat, provider, model in rows:
        real = DEFAULT_SEATS[seat.lower()]
        assert (provider, model) == (real["provider"], real["model"]), (
            f"CLAUDE.md says {seat} runs {provider}/{model}; "
            f"DEFAULT_SEATS says {real['provider']}/{real['model']}"
        )


def test_ci_runs_the_checks_the_quick_reference_documents():
    """"Run all checks" in CLAUDE.md and "CI is green" have to mean one thing.

    The workflow's own header promises the commands are copied verbatim from
    the Quick Reference rather than reworded. That promise is the reason a
    contributor can trust a local pass, and nothing enforced it: either side
    could gain a path the other never heard about, and the first sign would be
    a merge that passed CI and broke on someone's machine.
    """
    docs = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    block = re.search(r"# Run all checks\n(.*?)```", docs, re.S)
    assert block, "CLAUDE.md's Quick Reference no longer has a 'Run all checks' block"

    documented = [line.strip() for line in block.group(1).splitlines()
                  if line.strip().startswith(("ruff", "mypy", "python -m pytest"))]
    assert documented, "no check commands found in the Quick Reference"

    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    missing = [c for c in documented if c not in workflow]
    assert not missing, (
        f"CI does not run what CLAUDE.md documents: {missing}"
    )


# ---------------------------------------------------------------------------
# claims about the corpus, recomputed from the corpus
# ---------------------------------------------------------------------------


def test_every_file_the_walk_offers_is_indexable():
    """`oversized` reports a file the walk offers and the indexer must skip.

    Reported apart from `stale` because a reindex cannot fix it, and asserted
    here because the fix is to split the file and nobody notices the need.
    """
    oversized = [
        f"{p} at {len(t):,} characters"
        for p in iter_project_files(str(ROOT))
        if (t := (ROOT / p).read_text(encoding="utf-8", errors="replace"))
        and len(t) > MAX_INDEXABLE_BYTES
    ]
    assert not oversized, (
        f"over MAX_INDEXABLE_BYTES ({MAX_INDEXABLE_BYTES:,}): {oversized}"
    )


# The share of MAX_INDEXABLE_BYTES a file may reach before this suite starts
# asking for it to be dealt with. It exists because the guard above is a cliff:
# it fires the moment a file is already too large to index, which is the moment
# the only remaining fix is emergency surgery. CLAUDE.md reached 13 characters
# of headroom and nothing said a word until a paragraph would not fit, and
# `graphrag_server.py` had already been split once for the same reason. 0.8
# leaves a fifth of the budget to notice in, which on the current limit is
# 50,000 characters -- room to plan a split rather than perform one.
INDEXABLE_WARN_RATIO = 0.8


def test_no_file_is_creeping_up_on_the_index_limit():
    """The ramp the cliff above never gave anyone.

    A file that crosses this is not broken and nothing is dropped yet -- it is
    a file to split, move detail out of, or deliberately exclude, decided with
    time in hand. Raise the limit only on the reasoning at
    `MAX_INDEXABLE_BYTES` itself, never to silence this.
    """
    threshold = int(MAX_INDEXABLE_BYTES * INDEXABLE_WARN_RATIO)
    creeping = sorted(
        (
            (len(t), p)
            for p in iter_project_files(str(ROOT))
            if (t := (ROOT / p).read_text(encoding="utf-8", errors="replace"))
            and len(t) > threshold
        ),
        reverse=True,
    )
    assert not creeping, (
        f"within {1 - INDEXABLE_WARN_RATIO:.0%} of MAX_INDEXABLE_BYTES "
        f"({MAX_INDEXABLE_BYTES:,}): "
        + ", ".join(f"{p} at {n:,} characters" for n, p in creeping)
    )


# A capital is forced when position explains it: the first word of a line, the
# word after a full stop, the first cell of a table row. Anything else is a
# capital the writer chose.
_MARKER = re.compile(r"^(#{1,6}|[-*>+|]|\d+[.)])$")
_STRIP = "\"'`()[]{}<>.,!?;:*=+-/\\|"
_SENTENCE_END = (".", "!", "?", ":", ";")


def _is_upload(path: str) -> bool:
    """True for a document the operator handed *this machine*.

    Uploads are the other per-machine corner of the walk, and the reason this
    exists is in `_capital_census` below.
    """
    parts = PurePosixPath(path.replace("\\", "/")).parts
    return UPLOADS_DIR in parts[:-1]


def _is_generated_project(path: str, root: Path) -> bool:
    """True for a file of a run's generated project under `projects/`.

    The walk takes one only once the operator embeds it, which makes it the
    same per-machine corner as `uploads/`, for the same reason.
    """
    parts = Path(path).relative_to(root).parts if Path(path).is_absolute() else Path(path).parts
    return bool(parts) and parts[0] == PROJECTS_DIR


def _capital_census(root: Path = ROOT) -> dict[str, dict[str, int | set[str]]]:
    """Per minted token: documents, and capitals position does not explain.

    Skips what `add_document` skips. A page the research phase fetched is
    walked like any markdown file but mints no entities (`_mints_entities`, which
    also leaves out markup, script and config),
    so counting it audits a graph that does not exist -- and one that exists
    only on machines that have run web research. On 2026-09-10 eight pages
    under `research/web/` pushed two sentence-openers over the positional
    floor and a third word into the top twenty, failing both guards below over
    entities the real graph never held. They are not named here on purpose:
    this file is in the walk, and naming a word mid-sentence gives it the free
    capital that takes it out of the first guard's reach.

    **And skips `uploads/`, which is the same failure one directory over.** An
    upload does mint entities, unlike a fetched page, so this one is not
    mirroring `add_document` -- it is the only way a *pinned* list can be a
    claim anybody else can check. The walk includes whatever the operator
    handed this machine, so the census differed per machine and the two guards
    below could not be green in both places at once: measured on 2026-09-18,
    `Research` sits at 14 documents here and outside the top 24 in a clean
    checkout, which put it in the audited twenty on every developer machine and
    out of it in CI. That had been red in CI for three commits and was invisible
    the whole time, because a missing `scipy` aborted collection before any test
    ran -- the guard that cannot run pins nothing, one level up from the guard
    that cannot fail. An audit of what the *project* is about should not move
    because somebody uploaded a PDF's worth of prose to their own console.
    """
    census: dict[str, dict] = {}
    for path in iter_project_files(str(root)):
        if (not _mints_entities(str(path)) or _is_upload(str(path))
                or _is_generated_project(str(path), root)):
            continue
        try:
            text = (root / path).read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - unreadable file is its own problem
            continue
        if len(text) > MAX_INDEXABLE_BYTES:
            continue
        for line in text.splitlines():
            words = line.split()
            i = 0
            while i < len(words) and _MARKER.match(words[i].strip(_STRIP) or words[i]):
                i += 1
            first, prev = i, None
            for j in range(i, len(words)):
                token = words[j].strip(_STRIP)
                if (len(token) > 4 and token[:1].isupper()
                        and token.replace("_", "").isalnum()
                        and token.lower() not in ENTITY_STOPWORDS):
                    entry = census.setdefault(token, {"docs": set(), "free": 0})
                    entry["docs"].add(str(path))
                    if not (j == first or (prev and prev.endswith(_SENTENCE_END))):
                        entry["free"] += 1
                prev = words[j]
    return census


# Below this, a token is too rare to be worth a build failure: the graph reads
# a handful of pendant edges, not a hub. Four is where the 2026-09-09 audit
# drew the line, and every token it removed sat at or above it.
_POSITIONAL_DOC_FLOOR = 4


# Tokens that score zero position-free capitals and are entities anyway. Each
# needs a reason, because each is a hole in the guard below.
POSITIONAL_EXCEPTIONS = frozenset({
    # An assignment starts its line, so every capital is "forced" by position.
    # It is a real identifier in the spectral code.
    "L_dense",
})


def test_no_capital_forced_by_position_becomes_a_hub_entity():
    """A token minted by many documents with no capital that position fails to
    explain is recording where it sits rather than what it means.

    **This guard is deliberately weaker than the audit it comes from, and the
    reason is worth knowing before anyone tries to strengthen it.** It fires
    only at *zero* free capitals. The obvious improvement -- a ratio, "almost
    all its capitals are positional" -- cannot be made to work: measured on
    this corpus, `Measured` sits at 4 free against 34 forced and `Spectral` at
    2 against 24. The first is a sentence-opener and the second is a term the
    project is about, and no threshold separates them. That is the same
    finding CLAUDE.md records when it rejects the positional heuristic as a
    *filter*, met again one level up.

    So this catches a positional word **before anyone writes about it**, which
    is the window where it is unambiguous. Once a word has been discussed in
    prose it acquires free capitals and leaves this guard's reach -- which is
    exactly what happened to `Measured` and `Tests` while they were being
    fixed. The test below is what covers them afterwards.
    """
    offenders = sorted(
        (token, len(e["docs"]))
        for token, e in _capital_census().items()
        if e["free"] == 0 and len(e["docs"]) >= _POSITIONAL_DOC_FLOOR
        and token not in POSITIONAL_EXCEPTIONS
    )
    assert not offenders, (
        "these tokens are entities only because of where they sit; add each to "
        "ENTITY_STOPWORDS, or -- if it is a real identifier whose line simply "
        f"starts with it -- record why it stays: {offenders}"
    )


# The best-connected entities in the graph, as of the 2026-09-09 audit. Not a
# statistic -- a record of what a human looked at and accepted. Redone on
# 2026-09-12, when `NetworkX` displaced `Search`: the graph library this
# project is built on, named throughout the docstrings of an agent-written
# spectral-analysis module in the quisce prototype. A real term, so it stays.
# Redone again on
# 2026-09-15, the other way: removing the machine-specific dolphin-model
# modules before rollout took `NetworkX` from 13 documents to 12, and `Search`
# came back at 13 with 7 position-free capitals -- the entity the 2026-09-09
# audit had already cleared by that count. Redone on 2026-09-16: `NetworkX`
# returned at 14 documents (26 free capitals) while `Search` dropped to 13 --
# the graph library is a core term, so it stays and `Search` rotates out. And
# again the same day, the other way: the qwen3-embedding change deleted
# `RETRIEVAL_RELEVANCE_FLOOR` from the corpus's vocabulary entirely, which
# rotated `Search` back in -- already cleared by the 2026-09-09 audit. Redone
# again later on 2026-09-16, when the quisce prototype and the dolphin-model
# experiments were moved out of the checkout entirely: `NetworkX` fell from 14
# documents to 11 and out of the top twenty, and `Embedding` (11 documents, 5
# position-free capitals -- a term this project is about) took the last slot.
# Redone on 2026-09-18, in both directions at once. `Research` arrived at 13
# documents with 22 position-free capitals -- the Researcher's seat, the
# research phase, `research/web/`, `research_status` -- so it is a term the
# project is about and it stays. `Embedding` left at 12 documents, which is
# `Cheeger`'s count exactly: the two tie, the ranking breaks the tie by name,
# so the twentieth slot is settled alphabetically and one document either way
# rotates it back. Both movements predate the change that surfaced them -- this
# guard was already failing this way before two run artifacts were excluded
# from the walk, and an audit left outstanding is the claim going stale quietly.
# Redone on 2026-09-29, after `PROJECT_INDEX_EXCLUDES` grew `src/`, `tests/`,
# `prompts/`, `frontend/` and `spectral_graph/` -- the corpus stopped
# indexing the program's own source, and with it the census lost every
# entity that lived only in docstrings and comments under those trees:
# `BUILDER_DEADLINE_SECONDS`, `Cheeger`, `Embedding`, `GraphRAG`,
# `GraphRAGKnowledgeBase`, `LangGraph`, `MAX_INDEXABLE_BYTES`, `System`,
# `ValueError` and `Verdict` all left on subtraction, not on losing an
# argument about what the project is about. `System` and `ValueError` climbed
# back in from lower down the list once the shrunken census reshuffled ranks;
# named identically to two names that just left, they are the same
# entities, not new arrivals.
#
# What actually arrived is prose that used to be outnumbered by source: doc
# names (`Anthropic`, `OpenAI`, `Ollama`, `Cloud`, `Corpus`, `ANTHROPIC_API_KEY`)
# and two more read off `AgentState`'s own vocabulary (`State`, `StubLLM`).
# All eight are terms the project is about. Four more arrived and are not:
# `Hello` and `Inference` never carry a free capital at all -- `Inference is
# cloud only` opens both CLAUDE.md and README.md verbatim, and every `Hello`
# is a canned example string (`"Hello, how are you?"`, `'Hello World'`), never
# the entity that gave the word its capital. `Three` and `Verification` keep
# one free capital each, but only because a header or a docstring's first
# line puts the sentence-final word after it ("The Three Technologies",
# "Verification script for ..." three times over) -- the word describes a
# class of thing (a count, a category of script), not a thing the project has
# one of. All four joined `ENTITY_STOPWORDS` instead.
#
# Redone on 2026-09-30, after the example scripts caught up with the seats
# moving to local models (0b2152f): `example_usage.py` stopped warning that a
# missing cloud key meant canned stub output, and both it and `test_cloud.py`
# now read the seats and the recursion backstop off the package. `AGENTS`
# (config's tuple of the four seats) and `RECURSION_LIMIT` (the graph's
# superstep backstop) arrived at 3 documents each -- `serve.py`,
# `example_usage.py`, `test_cloud.py` -- and both are identifiers this project
# defines, so both stay. `ANTHROPIC_API_KEY` and `StubLLM` left on
# subtraction, falling to 2 documents each when that warning went, not by
# losing an argument about what the project is about. Later the same day the
# console stopped listening on every interface: `CONSOLE_HOST`, the variable
# that says where it listens, arrived at 3 documents -- `serve.py`, CLAUDE.md,
# README.md -- and is an identifier this project defines, so it stays.
# `ValueError` left without losing a document: still at 3, it is the tie at
# the twentieth slot, which the ranking settles by name.
AUDITED_TOP_ENTITIES = frozenset({
    "Architect", "Builder", "Researcher", "Laplacian", "Planner", "System",
    "CONSOLE_HOST", "Fiedler", "AgentState", "Exception", "AGENTS",
    "Anthropic", "Python", "Cloud", "Corpus", "OpenAI", "Ollama", "Search",
    "State", "RECURSION_LIMIT",
})


def test_the_best_connected_entities_are_the_ones_that_were_audited():
    """CLAUDE.md claims the twenty best-connected entities are all real terms.

    That claim cannot be checked by a rule -- see the test above, where the
    measurement refuses every threshold that would separate a sentence-opener
    from a domain term. What it can be is *pinned*: this is the list a person
    read and accepted, and the graph's answer has to still be that list.

    So this test fails whenever a new term reaches the top twenty, which is
    not a bug report. It is the audit asking to be redone, at the only moment
    the answer changed. Look at the newcomer, decide whether it is a term the
    project is about or a word doing the work of punctuation, then either add
    it to `ENTITY_STOPWORDS` or add it here.

    Computed from the walk rather than from an indexed corpus, so it needs no
    embedder and no store: an entity's document count *is* its degree in the
    bipartite graph, because the only edges are document -> entity.
    """
    census = _capital_census()
    ranked = sorted(census.items(), key=lambda kv: (-len(kv[1]["docs"]), kv[0]))
    top = {token for token, _ in ranked[:len(AUDITED_TOP_ENTITIES)]}

    arrived = sorted(top - AUDITED_TOP_ENTITIES)
    left = sorted(AUDITED_TOP_ENTITIES - top)
    assert not arrived and not left, (
        "the best-connected entities have moved, so the audit behind "
        '"the twenty best-connected entities are all real terms" needs '
        f"redoing.\n  newly in the top twenty: {arrived or 'none'}"
        f"\n  no longer in it: {left or 'none'}\n"
        "Rule on each newcomer: a term the project is about stays and joins "
        "AUDITED_TOP_ENTITIES; a word capitalised by position joins "
        "ENTITY_STOPWORDS."
    )


def test_the_census_skips_fetched_web_pages(tmp_path):
    """The census counts what the graph counts, and the graph skips web pages.

    Built on a scratch tree, because asserted against the checkout this passes
    vacuously wherever no research has run -- CI included -- which is exactly
    where the bug could not show.
    """
    from langgraph_agent.graphrag_server import WEB_RESEARCH_DIR

    web = tmp_path / WEB_RESEARCH_DIR
    web.mkdir(parents=True)
    (web / "example-com-page-00000000.md").write_text("see Zebracorn here\n")
    (tmp_path / "notes.md").write_text("see Quokkaline here\n")

    census = _capital_census(tmp_path)

    assert "Quokkaline" in census, "the scratch tree was not walked at all"
    assert "Zebracorn" not in census
