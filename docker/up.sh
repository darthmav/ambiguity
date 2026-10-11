#!/usr/bin/env bash
# Start (or restart) the Docker-only console, as you.
#
# What the container needs from your login is read here, at every start, and
# handed to it through the environment -- never written to a file:
#   * your GitHub token, from gh's login (kept in your keyring), so git_dwell
#     can push, open the pull request, read CI and merge;
#   * your git name and email, so its commits are yours.
# That is why the console is started through this script rather than a bare
# `docker compose up`, which would start it with neither.
#
# Usage:
#   ./docker/up.sh            start; builds the image only if there is none
#   ./docker/up.sh --build    rebuild the image from this checkout first (after
#                             a git pull: the image runs the code it was built with)
#   ./docker/up.sh --down     stop it (the run in flight writes its snapshot first)

set -euo pipefail

cd "$(dirname "$0")/.."
COMPOSE_FILE=docker/compose.yml

BUILD=0 DOWN=0
for arg in "$@"; do
    case "$arg" in
        --build) BUILD=1 ;;
        --down)  DOWN=1 ;;
        -h|--help) sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

if [ ! -f docker/.env ]; then
    echo "docker/.env does not exist: run ./docker/install.sh first" >&2
    exit 1
fi

# Docker directly when the socket is ours, through sudo while the docker group
# has not applied yet (it does after a reboot). sudo resets the environment, so
# the variables compose interpolates are named for it to keep.
PASSED=(GH_TOKEN GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL AMBIGUITY_UID AMBIGUITY_GID)
docker_cmd=(docker)
if [ -e /var/run/docker.sock ] && [ ! -w /var/run/docker.sock ]; then
    docker_cmd=(sudo "--preserve-env=$(IFS=,; echo "${PASSED[*]}")" docker)
fi
compose=("${docker_cmd[@]}" compose -f "$COMPOSE_FILE")

if [ "$DOWN" -eq 1 ]; then
    exec "${compose[@]}" stop
fi

export AMBIGUITY_UID AMBIGUITY_GID
AMBIGUITY_UID="$(id -u)"
AMBIGUITY_GID="$(id -g)"

export GH_TOKEN=""
if command -v gh >/dev/null && gh auth status --hostname github.com >/dev/null 2>&1; then
    GH_TOKEN="$(gh auth token --hostname github.com 2>/dev/null || true)"
fi
if [ -n "$GH_TOKEN" ]; then
    echo "  github: passing your gh login to the console"
else
    echo "  github: gh is not signed in, so runs commit locally and never push"
    echo "          (gh auth login, then ./docker/up.sh again)"
fi

export GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL
GIT_AUTHOR_NAME="$(git config --global user.name || true)"
GIT_AUTHOR_EMAIL="$(git config --global user.email || true)"
if [ -z "$GIT_AUTHOR_NAME" ] || [ -z "$GIT_AUTHOR_EMAIL" ]; then
    echo "  git: no global name/email, so commits carry the container's fallback identity"
fi

up=(up -d)
[ "$BUILD" -eq 1 ] && up+=(--build)
"${compose[@]}" "${up[@]}"

port="$(sed -n 's/^[[:space:]]*AMBIGUITY_PORT=//p' docker/.env | tail -n 1)"
port="${port:-8081}"
for _ in $(seq 1 60); do
    if curl -sf -m 2 "http://127.0.0.1:$port/api/status" | grep -q '"indexes_on_run"'; then
        echo "  console: http://127.0.0.1:$port"
        exit 0
    fi
    sleep 1
done
echo "  console did not answer on $port within a minute: ${compose[*]} logs console" >&2
exit 1
