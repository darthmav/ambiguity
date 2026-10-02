"""Documentation claims, made executable.

Prose goes stale silently, and a person reading carefully does not scale. The
rule this file enforces instead:

    a claim worth writing down is a claim worth failing a build over.

Most guards **recompute** the claim from the repository, so they need no
upkeep and catch drift nobody anticipated. A few compare prose against a
constant through a registry, extended whenever a new figure is written into
the docs -- better still, the prose names the constant instead.

These tests read the real repository rather than a fixture: a fixture would
test the checker.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from langgraph_agent.config import DEFAULT_SEATS
from langgraph_agent.graphrag_server import (
    EMBEDDING_BATCH_SIZE,
    ENTITY_STOPWORDS,
    MAX_INDEXABLE_BYTES,
    UPLOADS_DIR,
    _mints_entities,
    iter_corpus_files,
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
    return Path(cited).name in RUNTIME_NAMES or cited.startswith(RUNTIME_PATHS)


def _source_files() -> list[Path]:
    """Every Python file a reader would call part of this project.

    The root is a glob rather than a list of names, so a new root-level module
    is covered the day it lands; `spectral_graph/` is in although it is not
    installed, because it is importable only with the project root on the path.
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
    """A path in prose is a promise that a reader can go and look; recomputed,
    so a file deleted or renamed tomorrow fails here rather than on a reader."""
    dangling: list[str] = []
    for name in PROSE_FILES:
        for cited in _cited_paths((ROOT / name).read_text(encoding="utf-8")):
            if _excused(cited) or _cited_path_exists(cited):
                continue
            dangling.append(f"{name} cites {cited}")

    assert not dangling, "\n".join(dangling)


def test_every_path_the_source_cites_exists():
    """The same rule in docstrings, narrower in two deliberate ways.

    Only citations carrying a directory are checked: a bare `foo.py` in a
    docstring is an example, not a promise about a location. And `tests/` is
    not scanned: naming files that do not exist is what a fixture does.
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
    # Root-level shell scripts are entry points -- the installer, the launcher
    # -- so a new one belongs in the tree as much as a module does.
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
    config = (ROOT / "src/langgraph_agent/config.py").read_text(encoding="utf-8")
    imported = set(re.findall(r"from (langchain_\w+) import", config))
    assert imported, "config.py imports no provider packages; has the seat wiring moved?"

    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
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

    # .env.example's override templates start from the same defaults.
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    for role, real in DEFAULT_SEATS.items():
        for key in ("provider", "model"):
            line = f"# {role.upper()}_{key.upper()}={real[key]}"
            assert re.search(rf"^{re.escape(line)}$", example, re.MULTILINE), (
                f".env.example has no `{line}` matching DEFAULT_SEATS"
            )


def test_a_container_stop_outlasts_the_node_in_flight():
    """`docker compose stop` and `podman stop` have to wait out the node in flight.

    SIGTERM stops the run and defers the exit to the `finally` that writes its
    snapshot, but a stop never cuts a seat call short: the node in flight
    returns when its call does, or at its deadline. A grace period shorter than
    the longest deadline kills the process first, and with it the snapshot the
    README promises -- which Podman's ten-second default and compose's old 30
    seconds both did.
    """
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    grace = re.search(r"^\s*stop_grace_period:\s*(\d+)s\s*$", compose, re.MULTILINE)
    assert grace, "docker-compose.yml no longer sets stop_grace_period in seconds"
    podman = re.search(r"podman run\b[^\n]*--stop-timeout[ =](\d+)",
                       (ROOT / "README.md").read_text(encoding="utf-8"))
    assert podman, "README.md's podman run no longer sets --stop-timeout"
    assert podman.group(1) == grace.group(1), (
        f"podman stop waits {podman.group(1)}s for the console, "
        f"docker compose stop {grace.group(1)}s"
    )

    # Commented lines count: they document the code's own default.
    shipped = re.findall(r"^#?\s*(NODE_DEADLINE_SECONDS|BUILDER_DEADLINE_SECONDS)=(\d+)",
                         (ROOT / ".env.example").read_text(encoding="utf-8"), re.MULTILINE)
    assert {name for name, _ in shipped} == {"NODE_DEADLINE_SECONDS", "BUILDER_DEADLINE_SECONDS"}
    name, longest = max(shipped, key=lambda pair: float(pair[1]))
    assert int(grace.group(1)) > float(longest), (
        f"a container stop waits {grace.group(1)}s, but .env.example gives {name} "
        f"{longest}s: a run stopped mid-call is killed before its snapshot is written"
    )


def test_the_documented_rpc_methods_are_the_served_ones():
    """frontend/README.md listed a `reindex` that no longer existed and missed
    fifteen methods that did."""
    import serve

    readme = (ROOT / "frontend/README.md").read_text(encoding="utf-8")
    table = readme.split("| Method | Params | Returns |", 1)[1].split("\n\n", 1)[0]
    documented = set(re.findall(r"^\| `(\w+)` \|", table, re.MULTILINE))
    served = set(serve.RPC_METHODS)
    assert documented == served, (
        f"documented but not served: {sorted(documented - served)}; "
        f"served but not documented: {sorted(served - documented)}"
    )


def test_a_bare_pytest_collects_the_suite_and_nothing_else():
    """A bare `pytest` from the root collects `tests/` alone, and can import it.

    A root-level `test_*.py` would be collected (one once ran a live agent at
    import), and without the root on the path every module importing `serve`
    fails to collect. `-P` leaves the working directory off the path, exactly
    as the `pytest` script does and `python -m pytest` does not.
    """
    collected = subprocess.run(
        [sys.executable, "-P", "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert collected.returncode == 0, collected.stdout[-2000:] + collected.stderr[-2000:]
    ids = [line for line in collected.stdout.splitlines() if "::" in line]
    assert ids and all(line.startswith("tests/") for line in ids)


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

    # And lints with the ruff the tools extra pins, so a pass means one thing.
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pinned = next(d for d in extras["project"]["optional-dependencies"]["tools"]
                  if d.startswith("ruff"))
    assert f'pip install "{pinned}"' in workflow, (
        f"pyproject pins {pinned}, but CI's lint job installs something else"
    )


# ---------------------------------------------------------------------------
# the entity census
# ---------------------------------------------------------------------------


# A capital is forced when position explains it: the first word of a line, the
# word after a full stop, the first cell of a table row. Anything else is a
# capital the writer chose.
_MARKER = re.compile(r"^(#{1,6}|[-*>+|]|\d+[.)])$")
_STRIP = "\"'`()[]{}<>.,!?;:*=+-/\\|"
_SENTENCE_END = (".", "!", "?", ":", ";")


# Top-level directories the census leaves out: the tests, whose fixtures are
# invented words and whose audit lists would count themselves; agent tooling and
# run notes; and everything per-machine. What is left is the prose this checkout
# ships -- docs, prompts, the package, the scripts -- the same on every machine.
CENSUS_SKIPPED_DIRS = frozenset({
    "tests", ".claude", ".qwen", "experimental", "knowledge", "reports",
    UPLOADS_DIR, PROJECTS_DIR,
})


def _capital_census(root: Path = ROOT) -> dict[str, dict[str, int | set[str]]]:
    """Per minted token: documents, and capitals position does not explain.

    Walks the checkout, not the corpus: the corpus holds research and
    deliberate embeds (`CORPUS_ROOTS`), which a clean checkout has none of, so
    what is audited is the extractor's behaviour on this project's own prose.
    Skips what `add_document` skips (`_mints_entities`), and per-machine
    directories too, so the pinned list reads the same in CI as anywhere else.
    """
    census: dict[str, dict] = {}
    for path in iter_corpus_files(str(root), roots=("",)):
        if (path.relative_to(root).parts[0] in CENSUS_SKIPPED_DIRS
                or not _mints_entities(str(path))):
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
# a handful of pendant edges, not a hub.
_POSITIONAL_DOC_FLOOR = 4


# Tokens that score zero position-free capitals and are entities anyway, each
# with its reason beside it: every entry is a hole in the guard below.
POSITIONAL_EXCEPTIONS: frozenset[str] = frozenset()


def test_no_capital_forced_by_position_becomes_a_hub_entity():
    """A token many documents mint, with no capital position fails to explain,
    records where it sits rather than what it means.

    Deliberately weaker than the audit it comes from: it fires only at zero
    free capitals, because no ratio separates a sentence-opener from a domain
    term (`Measured` once sat at 4 free capitals against 34 forced, `Spectral`
    at 2 against 24). It catches a positional word before anyone writes about
    it; the pinned audit below covers it after.
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


# The twenty best-connected entities, as a person last read and accepted them:
# 2026-10-02, when the embedder gained a circuit of its own and `Circuit` (the
# self-healing class) reached a sixth document. The twentieth slot is a tie at
# six documents, settled by name, so it took the place of `CircuitOpenError`;
# both are real terms. The 2026-10-01 audit, when the census widened back to
# all the prose the checkout ships, ruled `ValueError`, `GraphRAG`, `Callable`,
# `Verdict` and `Architecture` (the Architect's section and state field) real
# terms, and four sentence-openers the positional guard caught joined
# `ENTITY_STOPWORDS`. Earlier rulings: `git log -p tests/test_claims.py`.
AUDITED_TOP_ENTITIES = frozenset({
    "Builder", "Architect", "Planner", "Researcher", "Exception", "ValueError",
    "GraphRAG", "Ollama", "Python", "AgentState", "Callable", "Search",
    "Verdict", "Anthropic", "Fiedler", "Laplacian", "RECURSION_LIMIT", "AGENTS",
    "Architecture", "Circuit",
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
