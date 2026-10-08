"""The dwell tool's pin: it only ever saves this folder.

The pipeline itself is exercised against real repositories in test_git_dwell.py;
what the tool adds on top is the promise that holds however it is invoked:
the working directory is this checkout's root and no path escapes it. Those
refusals are what is pinned here, plus one read-only survey against the real
checkout to prove the pair agree on what this repository is.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("dwell_tool", ROOT / "scripts" / "dwell.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("dwell_tool", module)
    spec.loader.exec_module(module)
    return module


dwell = _load()


def test_the_tool_knows_this_folder_is_its_own_repository():
    assert dwell.this_repository() == ""


def test_a_path_outside_the_folder_is_refused():
    _, refusal = dwell.confined_paths(["..", "/etc/passwd"])
    assert refusal, "paths leaving the folder must be refused"
    assert "only saves this folder" in refusal


def test_a_dotted_path_that_stays_inside_is_repo_relative():
    paths, refusal = dwell.confined_paths(["src/../README.md", "."])
    assert refusal == ""
    assert paths == ["README.md", "."]


def test_run_refuses_an_escaping_path_before_git_is_asked():
    code, result = dwell.run(["msg", "--paths", "../../outside.txt", "--stages", "survey"])
    assert code == 2
    assert result["success"] is False
    assert "only saves this folder" in result["error"]


def test_a_survey_of_this_checkout_succeeds(capsys):
    code, result = dwell.run(["msg", "--stages", "survey"])
    capsys.readouterr()
    assert code == 0
    assert result["success"] is True
    stages = {entry["stage"] for entry in result["stages"]}
    assert stages == {"survey"}
