"""Generated projects: where a run writes, and what the corpus reads of it."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

import serve
from langgraph_agent.graphrag_server import iter_corpus_files
from langgraph_agent.nodes import _Deadline, _outside_output_dir, _run_builder_tools
from langgraph_agent.projects import (
    embedded_projects,
    list_projects,
    project_name_error,
    project_name_rule,
    set_project_embedded,
)


def _tree(root):
    (root / "main.py").write_text("x = 1\n")
    for name in ("snake", "chess"):
        (root / "projects" / name).mkdir(parents=True)
        (root / "projects" / name / "game.py").write_text("y = 2\n")
    (root / "projects" / "loose.md").write_text("belongs to no project\n")


def _walked(root):
    return sorted(str(p.relative_to(root)) for p in iter_corpus_files(str(root)))


def test_a_generated_project_stays_out_of_the_walk_until_embedded(tmp_path):
    # `main.py` is the checkout's own file: never walked, embedded or not.
    _tree(tmp_path)
    assert _walked(tmp_path) == []

    set_project_embedded("snake", True, tmp_path)
    assert _walked(tmp_path) == ["projects/snake/game.py"]

    set_project_embedded("snake", False, tmp_path)
    assert _walked(tmp_path) == []


def test_only_the_first_component_holds_a_file_out(tmp_path, whole_root_walk):
    # A substring match would catch `subprojects/` anywhere in the tree.
    (tmp_path / "docs" / "subprojects").mkdir(parents=True)
    (tmp_path / "docs" / "subprojects" / "a.md").write_text("kept\n")
    assert _walked(tmp_path) == ["docs/subprojects/a.md"]


def test_a_broken_record_embeds_nothing(tmp_path):
    _tree(tmp_path)
    (tmp_path / "projects" / "embedded.json").write_text("{not json")
    assert embedded_projects(tmp_path) == set()
    assert _walked(tmp_path) == []


def test_list_projects_reports_files_and_state(tmp_path):
    _tree(tmp_path)
    set_project_embedded("chess", True, tmp_path)
    assert [(p["name"], p["files"], p["embedded"]) for p in list_projects(tmp_path)] == [
        ("chess", 1, True),
        ("snake", 1, False),
    ]


@pytest.mark.parametrize("bad", ["", "..", "../x", "a/b", ".hidden", "-x", 3])
def test_a_project_name_is_one_plain_component(bad):
    assert project_name_error(bad)


@pytest.mark.parametrize("taken", ["embedded.json", "Embedded.JSON", "embedded.json.tmp"])
def test_a_project_cannot_take_the_name_of_the_opt_in_record(taken):
    """A run into project 'embedded.json' made a directory where the record
    lives, and the next rebuild pruned every opted-in project from the corpus."""
    assert "reserved" in (project_name_error(taken) or "")


@pytest.mark.parametrize("name", ["", "..", "../x", "a/b", ".hidden", "-x", "a..b",
                                  "chess", "my-app_2.0", "x" * 64, "x" * 65,
                                  "embedded.json", "Embedded.JSON"])
def test_the_rule_the_console_is_sent_is_the_one_the_server_applies(name):
    """The page checks a name with `project_name_rule` before sending it; any
    name the two judge apart is a goal refused after the transcript took it."""
    rule = project_name_rule()
    pattern = re.compile(rule["pattern"])
    passes = bool(pattern.match(name)) and name.lower() not in rule["reserved"]
    assert passes == (project_name_error(name) is None)


def test_a_project_s_file_count_skips_what_the_walk_skips(tmp_path):
    """It counted a project's own .venv and node_modules: tens of thousands of
    files, on every refresh of the Corpus tab, that the corpus never reads."""
    project = tmp_path / "projects" / "app"
    (project / "src").mkdir(parents=True)
    (project / "src" / "main.py").write_text("print(1)\n", encoding="utf-8")
    for skipped in (".venv/lib", "node_modules/pkg", "build", "app.egg-info"):
        (project / skipped).mkdir(parents=True)
        (project / skipped / "f.txt").write_text("x", encoding="utf-8")

    assert [(p["name"], p["files"]) for p in list_projects(tmp_path)] == [("app", 1)]


def test_the_builder_may_write_only_inside_the_chosen_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _outside_output_dir("projects/snake/game.py", "projects/snake") is None
    assert _outside_output_dir("main.py", "projects/snake")
    assert _outside_output_dir("projects/snake/../../main.py", "projects/snake")
    assert _outside_output_dir("projects/chess/game.py", "projects/snake")


def test_the_tool_loop_refuses_a_write_outside_the_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class _Writes:
        def __init__(self):
            self.turn = 0

        def invoke(self, _messages):
            self.turn += 1
            if self.turn > 1:
                return SimpleNamespace(content="done", tool_calls=[])
            return SimpleNamespace(content="", tool_calls=[
                {"name": "filesystem_write", "id": "1",
                 "args": {"path": "main.py", "content": "z = 3\n"}},
                {"name": "filesystem_write", "id": "2",
                 "args": {"path": "projects/snake/game.py", "content": "z = 3\n"}},
            ])

    changed: list[str] = []
    _run_builder_tools(_Writes(), [], changed, [], _Deadline(60), output_dir="projects/snake")
    assert changed == ["projects/snake/game.py"]
    assert not (tmp_path / "main.py").exists()


def test_run_goal_refuses_a_bad_project_before_claiming_the_run():
    with pytest.raises(ValueError, match="not a project name"):
        serve.rpc_run_goal({"goal": "g", "project": "../escape"})
    assert not serve._run_progress["running"]


def test_embed_project_records_the_choice(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _tree(tmp_path)
    monkeypatch.setattr(serve, "REBUILD_CORPUS", False)
    r = serve.rpc_embed_project({"name": "snake"})
    assert r["embedded"] is True and r["rebuilding"] is False
    assert embedded_projects(tmp_path) == {"snake"}

    with pytest.raises(ValueError, match="no project"):
        serve.rpc_embed_project({"name": "missing"})


def test_embed_project_is_refused_mid_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _tree(tmp_path)
    monkeypatch.setitem(serve._run_progress, "running", True)
    with pytest.raises(ValueError, match="in flight"):
        serve.rpc_embed_project({"name": "snake"})
    assert embedded_projects(tmp_path) == set()


def test_a_project_s_files_are_verified_where_the_builder_was_told_to_run_them(
    tmp_path, monkeypatch
):
    """OUTPUT_DIR_NOTE tells the Builder to run its files with cwd set to the
    project; the proof ran them from the checkout root, so a script that opens
    its own data file failed verification on every cycle."""
    from langgraph_agent.nodes import _verify_written_files

    monkeypatch.chdir(tmp_path)
    project = tmp_path / "projects" / "demo"
    project.mkdir(parents=True)
    (project / "data.csv").write_text("a,b\n", encoding="utf-8")
    (project / "main.py").write_text("print(open('data.csv').read())\n", encoding="utf-8")

    in_project = _verify_written_files(["projects/demo/main.py"], [], None, cwd="projects/demo")
    from_root = _verify_written_files(["projects/demo/main.py"], [], None)

    assert [status for _, status, _ in in_project] == ["ok"]
    assert [status for _, status, _ in from_root] == ["failed"]


def test_the_proof_runs_in_the_run_s_project(monkeypatch):
    import langgraph_agent.nodes as nodes

    seen: list[str] = []
    monkeypatch.setattr(nodes, "_lint_written_files", lambda *a: ([], [], ""))
    monkeypatch.setattr(
        nodes, "_verify_written_files",
        lambda files, log, deadline, cwd="": seen.append(cwd) or [],
    )
    state = {"output_dir": "projects/demo", "failed_verification": [], "lint_failed": []}

    nodes._prove(state, [], [], nodes._Deadline(10))  # type: ignore[arg-type]
    nodes._prove({**state, "output_dir": ""}, [], [], nodes._Deadline(10))  # type: ignore[arg-type]

    assert seen == ["projects/demo", ""]
