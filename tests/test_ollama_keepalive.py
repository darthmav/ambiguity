"""The daemon is kept running: what scripts/ollama_keepalive.sh asks of systemd.

Run against stand-ins for `sudo`, `systemctl` and `curl` on PATH, so nothing
here needs root or a systemd: the drop-in and units it writes are read back,
the calls it makes are recorded, and the watchdog's probe is driven through a
daemon that answers, hangs, was stopped, or has only just started.
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
    """A PATH whose `sudo` runs the command as is, and whose `systemctl` and
    `curl` answer from files under `state/` and log every call."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    log = tmp_path / "systemctl.log"
    (state / "active").write_text("yes")
    (state / "entered").write_text("1000000")      # started 1s after boot, in µs
    (state / "environment").write_text("OLLAMA_MAX_LOADED_MODELS=1")
    (state / "curl").write_text("0")
    (tmp_path / "uptime").write_text("100000.00 1.00\n")
    (bin_dir / "sudo").write_text('#!/bin/sh\nexec "$@"\n')
    (bin_dir / "systemctl").write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> "{log}"\n'
        'case "$*" in\n'
        '  "is-enabled ollama.service"|"is-enabled ollama-watchdog.timer") echo enabled ;;\n'
        '  "is-active ollama-watchdog.timer") echo active ;;\n'
        '  "show ollama.service -p Restart --value") echo always ;;\n'
        '  "show ollama.service -p StartLimitIntervalUSec --value") echo 0 ;;\n'
        f'  "show ollama.service -p ActiveEnterTimestampMonotonic --value") cat "{state}/entered" ;;\n'
        f'  "show ollama.service -p Environment --value") cat "{state}/environment" ;;\n'
        f'  "is-active --quiet ollama.service") [ "$(cat "{state}/active")" = yes ] ;;\n'
        "esac\n"
    )
    (bin_dir / "curl").write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> "{tmp_path}/curl.log"\n'
        f'exit "$(cat "{state}/curl")"\n'
    )
    for tool in ("sudo", "systemctl", "curl"):
        (bin_dir / tool).chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "OLLAMA_KEEPALIVE_DROPIN": str(tmp_path / "ollama.service.d" / "keep-alive.conf"),
        "OLLAMA_WATCHDOG_UNIT_DIR": str(tmp_path / "units"),
        "OLLAMA_WATCHDOG_BIN": str(tmp_path / "lib" / "ollama-keepalive"),
        "OLLAMA_WATCHDOG_STATE": str(tmp_path / "run"),
        "OLLAMA_WATCHDOG_UPTIME_FILE": str(tmp_path / "uptime"),
    }

    class Host:
        pass

    h = Host()
    h.env, h.log, h.state, h.tmp = env, log, state, tmp_path
    h.dropin = tmp_path / "ollama.service.d" / "keep-alive.conf"
    h.units = tmp_path / "units"
    h.probe_bin = tmp_path / "lib" / "ollama-keepalive"
    return h


def _run(host, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], env=host.env, capture_output=True,
                          text=True, timeout=30)


def _restarts(host):
    if not host.log.exists():
        return 0
    return host.log.read_text().splitlines().count("restart ollama.service")


def test_install_makes_systemd_restart_the_daemon_on_every_exit(host):
    done = _run(host, "install")

    assert done.returncode == 0, done.stdout + done.stderr
    body = host.dropin.read_text()
    assert "Restart=always" in body
    # Empty, so an exit status the packaged unit exempts is restarted too.
    assert "RestartPreventExitStatus=\n" in body + "\n"
    # Under [Unit], where systemd reads it: no start limit to give up at.
    assert body.index("StartLimitIntervalSec=0") < body.index("[Service]")
    calls = host.log.read_text()
    assert "daemon-reload" in calls
    assert "enable --now ollama.service" in calls


def test_install_sets_the_watchdog_ticking(host):
    done = _run(host, "install")

    assert done.returncode == 0, done.stdout + done.stderr
    service = (host.units / "ollama-watchdog.service").read_text()
    timer = (host.units / "ollama-watchdog.timer").read_text()
    # The timer runs a copy outside the checkout, so moving it breaks nothing.
    assert f"ExecStart={host.probe_bin} probe" in service
    assert "Type=oneshot" in service
    assert host.probe_bin.read_bytes() == SCRIPT.read_bytes()
    assert os.access(host.probe_bin, os.X_OK)
    assert "OnUnitActiveSec=1min" in timer and "WantedBy=timers.target" in timer
    assert "enable --now ollama-watchdog.timer" in host.log.read_text()
    assert "restarts the daemon when it stops answering" in done.stdout


def test_a_second_install_rewrites_nothing(host):
    _run(host, "install")
    host.log.write_text("")

    assert _run(host, "install").returncode == 0
    assert "daemon-reload" not in host.log.read_text()


def test_remove_watchdog_takes_the_timer_and_its_copy_away(host):
    _run(host, "install")

    done = _run(host, "remove-watchdog")

    assert done.returncode == 0, done.stdout + done.stderr
    assert not (host.units / "ollama-watchdog.timer").exists()
    assert not (host.units / "ollama-watchdog.service").exists()
    assert not host.probe_bin.exists()
    assert "disable --now ollama-watchdog.timer" in host.log.read_text()
    # The keep-alive itself stays.
    assert host.dropin.exists()


def test_a_daemon_that_answers_is_left_alone(host):
    for _ in range(5):
        assert _run(host, "probe").returncode == 0
    assert _restarts(host) == 0
    assert "http://127.0.0.1:11434/api/ps" in (host.tmp / "curl.log").read_text()


def test_a_hung_daemon_is_restarted_after_three_unanswered_asks(host):
    (host.state / "curl").write_text("28")   # curl's own timeout

    for expected in (0, 0, 1):
        assert _run(host, "probe").returncode == 0
        assert _restarts(host) == expected
    # And the count starts again after the restart.
    assert _run(host, "probe").returncode == 0
    assert _restarts(host) == 1


def test_one_answer_resets_the_count(host):
    (host.state / "curl").write_text("28")
    _run(host, "probe")
    _run(host, "probe")
    (host.state / "curl").write_text("0")
    _run(host, "probe")
    (host.state / "curl").write_text("28")
    _run(host, "probe")
    _run(host, "probe")

    assert _restarts(host) == 0


def test_a_daemon_someone_stopped_stays_stopped(host):
    (host.state / "active").write_text("no")
    (host.state / "curl").write_text("7")

    for _ in range(5):
        _run(host, "probe")

    assert _restarts(host) == 0
    assert not (host.tmp / "curl.log").exists()


def test_a_daemon_that_just_started_is_given_its_grace(host):
    # Started 60s ago (uptime 100000s): still within the 180s grace.
    (host.state / "entered").write_text(str((100000 - 60) * 1_000_000))
    (host.state / "curl").write_text("28")

    for _ in range(5):
        _run(host, "probe")

    assert _restarts(host) == 0


@pytest.mark.parametrize("environment, address", [
    ("OLLAMA_HOST=0.0.0.0:11500", "127.0.0.1:11500"),
    ("OLLAMA_HOST=http://127.0.0.1:11600", "127.0.0.1:11600"),
    ("OLLAMA_HOST=127.0.0.1", "127.0.0.1:11434"),
    ("OLLAMA_KEEP_ALIVE=5m", "127.0.0.1:11434"),
])
def test_the_probe_asks_where_the_daemon_listens(host, environment, address):
    (host.state / "environment").write_text(f"OLLAMA_NUM_PARALLEL=1 {environment}")

    _run(host, "probe")

    assert f"http://{address}/api/ps" in (host.tmp / "curl.log").read_text()


def test_both_installers_keep_the_daemon_running():
    for installer in ("install.sh", "docker/install.sh"):
        text = (ROOT / installer).read_text(encoding="utf-8")
        assert "scripts/ollama_keepalive.sh install" in text, installer
