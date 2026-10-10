"""What scripts/network_check.sh asks, and how it reads what came back.

Run against a stand-in `curl` on PATH that answers from a table of origins, so
nothing here touches the network: each test says which origins answer, runs a
group, and reads the verdict and the requests the check made.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "network_check.sh"
# By its path, so a test can hand the script a PATH without it.
BASH = shutil.which("bash") or "/bin/bash"

# What the stand-in prints for `-w '%{http_code} %{http_connect}'`, and its exit
# code, per URL; anything not listed is a refused connection.
CURL = """#!/bin/sh
url=""
for arg in "$@"; do url="$arg"; done
echo "$url" >> "{log}"
line="$(grep -F "$url " "{answers}" | head -n 1)"
if [ -z "$line" ]; then printf '000 000'; exit 7; fi
set -- $line
printf '%s 000' "$2"
exit "$3"
"""


class _Net:
    """A PATH whose `curl` answers from `answers` and logs every URL it is asked."""

    def __init__(self, tmp_path: Path) -> None:
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.answers = tmp_path / "answers"
        self.answers.write_text("")
        self.log = tmp_path / "curl.log"
        curl = self.bin / "curl"
        curl.write_text(CURL.format(log=self.log, answers=self.answers))
        curl.chmod(0o755)
        self.mirrorlist = tmp_path / "mirrorlist"
        self.mirrorlist.write_text("")
        self.env = {
            key: value for key, value in os.environ.items()
            if key not in ("PIP_INDEX_URL", "UV_DEFAULT_INDEX", "UV_INDEX_URL")
        }
        self.env.update(PATH=f"{self.bin}:{os.environ['PATH']}", MIRRORLIST=str(self.mirrorlist))

    def answer(self, url: str, code: str = "200", rc: int = 0) -> None:
        with self.answers.open("a") as handle:
            handle.write(f"{url} {code} {rc}\n")

    def asked(self) -> list[str]:
        return self.log.read_text().splitlines() if self.log.exists() else []

    def run(self, *groups: str, **extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([BASH, str(SCRIPT), *groups], env={**self.env, **extra},
                              capture_output=True, text=True, timeout=60)


@pytest.fixture
def net(tmp_path):
    return _Net(tmp_path)


def test_a_table_host_is_asked_on_https(net):
    net.answer("https://pypi.org/")
    net.answer("https://files.pythonhosted.org/", code="404")

    done = net.run("pypi")

    assert done.returncode == 0, done.stdout
    assert sorted(net.asked()) == ["https://files.pythonhosted.org/", "https://pypi.org/"]
    assert "pypi.org files.pythonhosted.org" in done.stdout


def test_an_index_on_plain_http_and_its_own_port_is_asked_there(net):
    """A devpi at http://devpi.lan:3141 was asked at https://devpi.lan/, read
    as blocked, and failed the install."""
    net.answer("http://devpi.lan:3141/")

    done = net.run("pypi", PIP_INDEX_URL="http://user:secret@devpi.lan:3141/root/pypi/+simple/")

    assert done.returncode == 0, done.stdout
    assert net.asked() == ["http://devpi.lan:3141/"]
    assert "http://devpi.lan:3141" in done.stdout
    assert "secret" not in done.stdout


def test_a_mirrorlist_of_lan_mirrors_is_asked_where_they_serve(net):
    net.mirrorlist.write_text(
        "# Server = https://commented.out/$repo/os/$arch\n"
        "Server = http://192.168.1.5:8080/archlinux/$repo/os/$arch\n"
        "Server = http://192.168.1.5:8080/other/$repo/os/$arch\n"
        "Server = https://mirror.example.org/$repo/os/$arch\n"
    )
    net.answer("http://192.168.1.5:8080/")

    done = net.run("arch")

    assert done.returncode == 0, done.stdout
    assert sorted(net.asked()) == ["http://192.168.1.5:8080/", "https://mirror.example.org/"]


def test_a_blocked_origin_is_named_and_its_entry_is_the_host(net):
    net.answer("https://github.com/")
    net.answer("https://api.github.com/")

    done = net.run("pypi", "github", PIP_INDEX_URL="http://devpi.lan:3141/simple/")

    assert done.returncode == 1
    assert "http://devpi.lan:3141 (connection refused)" in done.stdout
    assert "      devpi.lan\n" in done.stdout


def test_a_curl_that_cannot_run_is_not_an_answer(net):
    """An empty probe used to shift the exit code into the status column, so
    127 read as an HTTP status and every host as reachable."""
    (net.bin / "curl").write_text("#!/bin/sh\nexit 127\n")

    done = net.run("pypi")

    assert done.returncode == 1
    assert "curl could not be run" in done.stdout


def test_without_curl_the_check_fails_rather_than_passes(net, tmp_path):
    bare = tmp_path / "bare"
    bare.mkdir()
    for tool in ("awk", "sed", "tr", "mktemp", "rm", "head", "cat", "grep"):
        found = shutil.which(tool)
        assert found, tool
        (bare / tool).symlink_to(found)

    done = net.run("pypi", "--optional", "cloud", PATH=str(bare))
    assert done.returncode == 1
    assert "curl is not installed" in done.stdout

    # With nothing required, there is nothing to fail.
    assert net.run("--optional", "cloud", PATH=str(bare)).returncode == 0
