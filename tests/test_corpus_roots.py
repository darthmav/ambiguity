"""The corpus is research and what the operator embedded -- never the checkout.

`iter_corpus_files` walks `CORPUS_ROOTS` alone: pages the online research
phase fetched, uploads, and generated projects the operator opted in. The
checkout's own files -- README, CLAUDE.md, install.sh, config -- are the
program, and a console coming up used to embed them unasked.
"""

from __future__ import annotations

from langgraph_agent.graphrag_server import (
    CORPUS_ROOTS,
    UPLOADS_DIR,
    WEB_RESEARCH_DIR,
    iter_corpus_files,
)
from langgraph_agent.projects import PROJECTS_DIR, set_project_embedded


def _write(root, *relatives: str) -> None:
    for relative in relatives:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")


def _walked(root) -> set[str]:
    return {str(path.relative_to(root)) for path in iter_corpus_files(str(root))}


def test_the_roots_are_research_uploads_and_projects() -> None:
    assert set(CORPUS_ROOTS) == {WEB_RESEARCH_DIR, UPLOADS_DIR, PROJECTS_DIR}


def test_the_checkout_itself_is_never_walked(tmp_path) -> None:
    _write(tmp_path, "README.md", "CLAUDE.md", "install.sh", "pyproject.toml",
           "docker-compose.yml", ".github/workflows/ci.yml", "reports/notes.md")
    assert _walked(tmp_path) == set()


def test_research_and_uploads_are_walked(tmp_path) -> None:
    _write(tmp_path, "README.md", f"{WEB_RESEARCH_DIR}/page.md", f"{UPLOADS_DIR}/notes.md")
    assert _walked(tmp_path) == {f"{WEB_RESEARCH_DIR}/page.md", f"{UPLOADS_DIR}/notes.md"}


def test_a_project_is_walked_only_once_embedded(tmp_path) -> None:
    _write(tmp_path, f"{PROJECTS_DIR}/snake/game.py")
    assert _walked(tmp_path) == set()
    set_project_embedded("snake", True, tmp_path)
    assert _walked(tmp_path) == {f"{PROJECTS_DIR}/snake/game.py"}


def test_an_embedded_project_is_walked_whole_minus_tool_directories(tmp_path) -> None:
    """A project's own `src/`, `tests/`, `scripts/` and `frontend/` are its content.

    The walk once excluded those names as substrings left over from indexing
    the checkout, so an opted-in project lost every file below its top level.
    Only tool, VCS and build directories are skipped, and by whole name.
    """
    kept = ("src/app/core.py", "tests/test_core.py", "scripts/run.sh",
            "frontend/index.html", "prompts/system.txt", "rebuild/notes.md",
            "venv_notes.md")
    skipped = (".venv/lib/site.py", "venv/lib/site.py", "build/lib/out.py",
               "dist/app.py", "app.egg-info/SOURCES.txt", "__pycache__/core.py",
               "node_modules/pkg/index.js", ".git/hooks/pre-commit.sh")
    _write(tmp_path, *(f"{PROJECTS_DIR}/app/{p}" for p in kept + skipped))
    set_project_embedded("app", True, tmp_path)

    assert _walked(tmp_path) == {f"{PROJECTS_DIR}/app/{p}" for p in kept}
