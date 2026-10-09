#!/usr/bin/env bash
# Keep the Ollama daemon up: started at boot, and restarted whenever it exits.
#
# Every default seat and the embedder are served by the one local daemon, so a
# daemon that is down is a console that can do nothing -- and the console can
# only report it, since starting a system service is root's. This asks systemd,
# which already runs the daemon, to keep it running: a drop-in for
# ollama.service sets Restart=always (any exit, clean or not), clears any exit
# status the packaged unit exempts from restarting, and switches the start
# limit off, so a crash loop -- a driver not yet ready at boot -- is retried for
# as long as it lasts instead of given up on after five tries. The service is
# enabled at boot.
#
# A daemon that hangs without exiting is not something Restart= can see; the
# console reports it (the ollama-daemon circuit) and `sudo systemctl restart
# ollama` clears it.
#
# Usage:
#   scripts/ollama_keepalive.sh install    write the drop-in and enable (sudo)
#   scripts/ollama_keepalive.sh check      whether both are in place
#
# install.sh and docker/install.sh run `install`, the way they write the
# one-model drop-in beside it.

set -uo pipefail

# Overridable only so the tests can write somewhere harmless.
DROPIN="${OLLAMA_KEEPALIVE_DROPIN:-/etc/systemd/system/ollama.service.d/keep-alive.conf}"

# `RestartPreventExitStatus=` empty clears any list the packaged unit carries,
# and StartLimitIntervalSec=0 switches the start limit off.
DROPIN_BODY='[Unit]
StartLimitIntervalSec=0

[Service]
Restart=always
RestartSec=2
RestartPreventExitStatus='

install_keepalive() {
    if ! systemctl cat ollama.service >/dev/null 2>&1; then
        echo "  ✗ there is no ollama.service to keep up (sudo pacman -S ollama)"
        return 1
    fi
    # Written only when it differs, so a re-run touches nothing. Restart= and
    # the start limit are read at the daemon's next exit: no restart for them.
    if [ "$(cat "$DROPIN" 2>/dev/null)" != "$DROPIN_BODY" ]; then
        sudo install -D -m 0644 /dev/stdin "$DROPIN" <<<"$DROPIN_BODY" \
            && sudo systemctl daemon-reload || return 1
    fi
    sudo systemctl enable --now ollama.service >/dev/null 2>&1 || return 1
    check
}

check() {
    local bad=0
    if [ "$(systemctl is-enabled ollama.service 2>/dev/null)" = enabled ]; then
        echo "  ✓ ollama.service starts at boot"
    else
        echo "  ✗ ollama.service does not start at boot"; bad=1
    fi
    if [ "$(systemctl show ollama.service -p Restart --value 2>/dev/null)" = always ] \
            && [ "$(systemctl show ollama.service -p StartLimitIntervalUSec --value 2>/dev/null)" = 0 ]; then
        echo "  ✓ ollama.service is restarted whenever it exits, with no start limit"
    else
        echo "  ✗ ollama.service is not restarted on every exit ($DROPIN)"; bad=1
    fi
    return "$bad"
}

case "${1:-}" in
    install) install_keepalive ;;
    check) check ;;
    -h|--help) sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0" ;;
    *) echo "usage: $0 install|check (try --help)" >&2; exit 2 ;;
esac
