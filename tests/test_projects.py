"""Generated projects: where a run writes, and what the corpus reads of it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import serve
from langgraph_agent.graphrag_server import iter_project_files
from langgraph_agent.nodes import _Deadline, _outside_output_dir, _run_builder_tools
from langgraph_agent.projects import (
    embedded_projects,
    list_projects,
    project_name_error,
    set_project_embedded,
)


def _tree(root):
    (root / "main.py").write_text("x = 1\n")
    for name in ("snake", "chess"):
        (root / "projects" / name).mkdir(parents=True)
        (root / "projects" / name / "game.py").write_text("y = 2\n")
    (root / "projects" / "loose.md").write_text("belongs to no project\n")


def _walked(root):
    return sorted(str(p.relative_to(root)) for p in iter_project_files(str(root)))


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
    monkeypatch.setattr(serve, "INDEX_PROJECT_BEFORE_RUN", False)
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
