#!/usr/bin/env bash
# Keep the Ollama daemon up: started at boot, restarted whenever it exits, and
# restarted when it hangs.
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
# A daemon that hangs without exiting is not something Restart= can see, so a
# watchdog asks it once a minute: ollama-watchdog.timer runs this script's
# `probe`, which asks the daemon for its loaded models (/api/ps, which goes
# through its scheduler, where a hang shows) and restarts ollama.service after
# OLLAMA_WATCHDOG_FAILURES unanswered asks in a row -- three minutes by
# default, and none within OLLAMA_WATCHDOG_GRACE seconds of the daemon's own
# start, while it may still be loading. A daemon someone stopped is left
# stopped. The probe is a copy of this script outside the checkout, so moving
# or deleting the checkout never breaks the timer; `remove-watchdog` takes the
# timer, its service and that copy away again.
#
# Usage:
#   scripts/ollama_keepalive.sh install          drop-in, enable, watchdog (sudo)
#   scripts/ollama_keepalive.sh check            whether all of it is in place
#   scripts/ollama_keepalive.sh remove-watchdog  take the watchdog away (sudo)
#   scripts/ollama_keepalive.sh probe            one watchdog ask (the timer's)
#
# install.sh and docker/install.sh run `install`, the way they write the
# one-model drop-in beside it. `journalctl -u ollama-watchdog` shows every
# restart the watchdog made, and why.

set -uo pipefail

SELF="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"

# Overridable only so the tests can write somewhere harmless.
DROPIN="${OLLAMA_KEEPALIVE_DROPIN:-/etc/systemd/system/ollama.service.d/keep-alive.conf}"
UNIT_DIR="${OLLAMA_WATCHDOG_UNIT_DIR:-/etc/systemd/system}"
PROBE_BIN="${OLLAMA_WATCHDOG_BIN:-/usr/local/lib/ambiguity/ollama-keepalive}"
STATE_DIR="${OLLAMA_WATCHDOG_STATE:-/run/ambiguity-ollama-watchdog}"
UPTIME_FILE="${OLLAMA_WATCHDOG_UPTIME_FILE:-/proc/uptime}"

# How the watchdog judges: one ask may take this long, this many unanswered in
# a row restart the daemon, and none restarts it this soon after it started.
WATCHDOG_TIMEOUT="${OLLAMA_WATCHDOG_TIMEOUT:-20}"
WATCHDOG_FAILURES="${OLLAMA_WATCHDOG_FAILURES:-3}"
WATCHDOG_GRACE="${OLLAMA_WATCHDOG_GRACE:-180}"

# `RestartPreventExitStatus=` empty clears any list the packaged unit carries,
# and StartLimitIntervalSec=0 switches the start limit off.
DROPIN_BODY='[Unit]
StartLimitIntervalSec=0

[Service]
Restart=always
RestartSec=2
RestartPreventExitStatus='

WATCHDOG_SERVICE_BODY="[Unit]
Description=Restart the Ollama daemon when it stops answering (ambiguity)
After=ollama.service

[Service]
Type=oneshot
ExecStart=$PROBE_BIN probe"

# Every minute from three minutes after boot. A oneshot that is still asking
# when the next tick comes is not started twice.
WATCHDOG_TIMER_BODY='[Unit]
Description=Ask the Ollama daemon every minute whether it still answers (ambiguity)

[Timer]
OnBootSec=3min
OnUnitActiveSec=1min
AccuracySec=10s

[Install]
WantedBy=timers.target'

# Writes `body` to `path` as root, only when it differs: 0 when it wrote,
# 1 when the file already said it, 2 when the write failed.
write_if_changed() {
    local path="$1" body="$2"
    [ "$(cat "$path" 2>/dev/null)" = "$body" ] && return 1
    sudo install -D -m 0644 /dev/stdin "$path" <<<"$body" || return 2
}

# A daemon answering where ollama.service would listen while the unit is not
# running is someone else's -- `ollama serve` in a terminal, a user unit, a
# container. Starting the unit beside it only fails to bind the port, and with
# no start limit that failure would repeat every two seconds, at every boot.
foreign_daemon() {
    ! systemctl is-active --quiet ollama.service \
        && curl -fsS -m 3 -o /dev/null "http://$(daemon_address)/api/version" 2>/dev/null
}

install_keepalive() {
    if ! systemctl cat ollama.service >/dev/null 2>&1; then
        echo "  ✗ there is no ollama.service to keep up (sudo pacman -S ollama)"
        return 1
    fi
    if foreign_daemon; then
        echo "  ! a daemon ollama.service did not start answers at $(daemon_address); keeping"
        echo "    it running is up to whatever started it, and ollama.service is left"
        echo "    alone, not started beside it to fight for the port"
        return 0
    fi
    local changed=0 path body
    # Written only when it differs, so a re-run touches nothing. Restart= and
    # the start limit are read at the daemon's next exit: no restart for them.
    for path in "$DROPIN" "$UNIT_DIR/ollama-watchdog.service" "$UNIT_DIR/ollama-watchdog.timer"; do
        case "$path" in
            "$DROPIN") body="$DROPIN_BODY" ;;
            *.service) body="$WATCHDOG_SERVICE_BODY" ;;
            *) body="$WATCHDOG_TIMER_BODY" ;;
        esac
        write_if_changed "$path" "$body"
        case $? in 0) changed=1 ;; 2) return 1 ;; esac
    done
    # The probe's own copy: the timer must not depend on where the checkout is.
    if ! cmp -s "$SELF" "$PROBE_BIN"; then
        sudo install -D -m 0755 "$SELF" "$PROBE_BIN" || return 1
    fi
    if [ "$changed" -eq 1 ]; then
        sudo systemctl daemon-reload || return 1
    fi
    sudo systemctl enable --now ollama.service >/dev/null 2>&1 || return 1
    sudo systemctl enable --now ollama-watchdog.timer >/dev/null 2>&1 || return 1
    check
}

remove_watchdog() {
    sudo systemctl disable --now ollama-watchdog.timer >/dev/null 2>&1
    sudo rm -f "$UNIT_DIR/ollama-watchdog.timer" "$UNIT_DIR/ollama-watchdog.service" "$PROBE_BIN"
    sudo systemctl daemon-reload
    echo "  ✓ the watchdog is gone; the daemon is still restarted whenever it exits"
}

check() {
    local bad=0
    if foreign_daemon; then
        echo "  ! the daemon at $(daemon_address) is not ollama.service's; what follows is the unit's"
    fi
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
    if [ "$(systemctl is-enabled ollama-watchdog.timer 2>/dev/null)" = enabled ] \
            && [ "$(systemctl is-active ollama-watchdog.timer 2>/dev/null)" = active ] \
            && [ -x "$PROBE_BIN" ]; then
        echo "  ✓ ollama-watchdog.timer restarts the daemon when it stops answering"
    else
        echo "  ✗ no watchdog restarts a daemon that hangs (ollama-watchdog.timer)"; bad=1
    fi
    return "$bad"
}

# Where the daemon listens, from its own unit's OLLAMA_HOST: what it binds to,
# asked on loopback when it binds every interface.
daemon_address() {
    local env host="127.0.0.1:11434" word
    env="$(systemctl show ollama.service -p Environment --value 2>/dev/null)"
    for word in $env; do
        case "$word" in OLLAMA_HOST=*) host="${word#OLLAMA_HOST=}" ;; esac
    done
    host="${host#http://}"
    host="${host%/}"
    case "$host" in
        "") host="127.0.0.1:11434" ;;
        *:*) ;;
        *) host="$host:11434" ;;
    esac
    case "$host" in 0.0.0.0:*) host="127.0.0.1:${host#0.0.0.0:}" ;; esac
    echo "$host"
}

# Seconds since the daemon's current start, from systemd's monotonic stamp.
seconds_since_start() {
    local entered now
    entered="$(systemctl show ollama.service -p ActiveEnterTimestampMonotonic --value 2>/dev/null)"
    now="$(cut -d. -f1 "$UPTIME_FILE" 2>/dev/null)"
    case "$entered$now" in *[!0-9]*|"") echo 0; return ;; esac
    echo $(( now - entered / 1000000 ))
}

probe() {
    local failures_file="$STATE_DIR/failures" failures address
    mkdir -p "$STATE_DIR"
    # Stopped on purpose, or not running yet: systemd's Restart= is for exits,
    # and a stop someone asked for is not this script's to undo.
    if ! systemctl is-active --quiet ollama.service; then
        rm -f "$failures_file"
        return 0
    fi
    if [ "$(seconds_since_start)" -lt "$WATCHDOG_GRACE" ]; then
        rm -f "$failures_file"
        return 0
    fi
    address="$(daemon_address)"
    if curl -fsS -m "$WATCHDOG_TIMEOUT" -o /dev/null "http://$address/api/ps" 2>/dev/null; then
        rm -f "$failures_file"
        return 0
    fi
    failures=$(( $(cat "$failures_file" 2>/dev/null || echo 0) + 1 ))
    if [ "$failures" -lt "$WATCHDOG_FAILURES" ]; then
        echo "$failures" >"$failures_file"
        echo "the daemon at $address did not answer within ${WATCHDOG_TIMEOUT}s ($failures of $WATCHDOG_FAILURES)"
        return 0
    fi
    rm -f "$failures_file"
    echo "the daemon at $address did not answer $failures times in a row: restarting ollama.service"
    systemctl restart ollama.service
}

case "${1:-}" in
    install) install_keepalive ;;
    check) check ;;
    remove-watchdog) remove_watchdog ;;
    probe) probe ;;
    -h|--help) sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$SELF" ;;
    *) echo "usage: $0 install|check|remove-watchdog|probe (try --help)" >&2; exit 2 ;;
esac
