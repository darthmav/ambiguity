"""The daemon is kept running: what scripts/ollama_keepalive.sh asks of systemd.

Run against stand-ins for `sudo` and `systemctl` on PATH, so nothing here needs
root or a systemd: the drop-in it writes is read back, and the calls it makes
are recorded.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ollama_keepalive.sh"


@pytest.fixture
def host(tmp_path):
    """A PATH whose `sudo` runs the command as is and whose `systemctl` logs."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "systemctl.log"
    (bin_dir / "sudo").write_text('#!/bin/sh\nexec "$@"\n')
    (bin_dir / "systemctl").write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> "{log}"\n'
        'case "$*" in\n'
        '  "is-enabled ollama.service") echo enabled ;;\n'
        '  "show ollama.service -p Restart --value") echo always ;;\n'
        '  "show ollama.service -p StartLimitIntervalUSec --value") echo 0 ;;\n'
        "esac\n"
    )
    for tool in ("sudo", "systemctl"):
        (bin_dir / tool).chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "OLLAMA_KEEPALIVE_DROPIN": str(tmp_path / "ollama.service.d" / "keep-alive.conf"),
    }
    return env, log, tmp_path / "ollama.service.d" / "keep-alive.conf"


def _run(env, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True,
                          text=True, timeout=30)


def test_install_makes_systemd_restart_the_daemon_on_every_exit(host):
    env, log, dropin = host

    done = _run(env, "install")

    assert done.returncode == 0, done.stdout + done.stderr
    body = dropin.read_text()
    assert "Restart=always" in body
    # Empty, so an exit status the packaged unit exempts is restarted too.
    assert "RestartPreventExitStatus=\n" in body + "\n"
    # Under [Unit], where systemd reads it: no start limit to give up at.
    assert body.index("StartLimitIntervalSec=0") < body.index("[Service]")
    calls = log.read_text()
    assert "daemon-reload" in calls
    assert "enable --now ollama.service" in calls


def test_a_second_install_rewrites_nothing(host):
    env, log, dropin = host
    _run(env, "install")
    log.write_text("")

    assert _run(env, "install").returncode == 0
    assert "daemon-reload" not in log.read_text()


def test_both_installers_keep_the_daemon_running():
    for installer in ("install.sh", "docker/install.sh"):
        text = (ROOT / installer).read_text(encoding="utf-8")
        assert "scripts/ollama_keepalive.sh install" in text, installer
