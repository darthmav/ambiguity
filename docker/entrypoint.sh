#!/bin/bash
# The container's entry point: make the checkout usable, say what the console
# can and cannot reach, then exec the command.
#
# It probes rather than waits. Neither dependency is fatal -- the console
# starts without Ollama and reports each seat's real state itself, and nothing
# in the app reads DATABASE_URL at all -- so a probe that blocked the start
# would turn a warning into an outage. What a probe must not do is stay quiet:
# with no daemon behind them every default seat fails its first call and the
# corpus cannot be embedded, and the log a container leaves is the first place
# anyone looks for why.

set -e

# 1. .env ------------------------------------------------------------------
# Same courtesy launch_console.sh does, and for the same reason: a checkout
# with no .env otherwise starts with every timeout at its code default rather
# than the values this project is actually run with.
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env 2>/dev/null && echo "  no .env: copied .env.example"
fi

# 2. git -------------------------------------------------------------------
# The checkout is bind-mounted from the host, so its .git is owned by whoever
# owns it there. When that is not the uid inside the container, git refuses the
# repository outright ("dubious ownership") and every git_ tool -- the whole of
# git_dwell -- fails on a repository that is perfectly fine.
#
# GIT_CONFIG_* rather than `git config --global`, because the usual way to give
# this container an identity is to mount the host's ~/.gitconfig read-only, and
# writing to that file is then impossible. These variables are supplementary:
# a mounted config still wins on the keys it sets.
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0='*'

# An identity only if the mounted config (or the environment) has not supplied
# one. Without it `git_dwell`'s commit stage fails at the last moment, after
# branching and staging -- the half-finished sequence that tool exists to
# avoid. With the host's ~/.gitconfig mounted this is skipped and the commits
# carry the operator's own name.
if [ -z "$(git config user.email 2>/dev/null)" ] && [ -z "${GIT_AUTHOR_EMAIL:-}" ]; then
    export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-Ambiguity Builder}"
    export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-builder@ambiguity.local}"
    export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
    export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
fi

# 3. the Ollama daemon -----------------------------------------------------
# Every default seat is an ollama seat and the one embedding model is served by
# the same daemon, so this is the dependency that decides whether the image can
# do anything at all. It lives on the host: under `network_mode: host` the
# default URL reaches it unchanged.
ollama_url="${OLLAMA_BASE_URL:-http://localhost:11434}"
if curl -fsS --max-time 3 "${ollama_url%/}/api/tags" >/dev/null 2>&1; then
    echo "  ollama: ${ollama_url} answers"
else
    echo "  ollama: ${ollama_url} does not answer -- every Ollama seat will fail"
    echo "          its runs (the console shows OFFLINE) and the corpus cannot"
    echo "          be embedded."
    echo "          On the host: systemctl status ollama. Off host networking,"
    echo "          the daemon must listen beyond loopback (OLLAMA_HOST=0.0.0.0)"
    echo "          and OLLAMA_BASE_URL must name it."
fi

# 4. PostgreSQL ------------------------------------------------------------
# Nothing in the app reads DATABASE_URL -- the corpus is Chroma, under
# knowledge/ -- but the console exports .env to everything it runs, so this is
# the URL a script the Builder writes will use. That is exactly the kind of
# configuration nobody finds out is wrong until an agent is halfway through a
# task, so it is proved here the way install.sh proves it on the host: with a
# query, through psql.
if [ -n "${DATABASE_URL:-}" ]; then
    if PGCONNECT_TIMEOUT=5 psql "$DATABASE_URL" -tAc 'select 1' >/dev/null 2>&1; then
        echo "  postgres: ${DATABASE_URL%%\?*} answers"
    else
        echo "  postgres: ${DATABASE_URL%%\?*} does not answer"
        case "$DATABASE_URL" in
            *127.0.0.1*|*localhost*)
                # The postgres18 container Omarchy's installer runs publishes
                # 5432 on 127.0.0.1 only, so loopback is the host's loopback
                # and nothing else: reachable on host networking, and never
                # from a bridge, where the gateway address the container would
                # use is not an address that port is published on.
                echo "            A loopback URL only resolves under network_mode: host."
                echo "            Check the container is up (docker ps | grep postgres18),"
                echo "            or see README.md > Running in a container for the"
                echo "            shared-network alternative."
                ;;
        esac
    fi
fi

# 5. the tokenizer ---------------------------------------------------------
# The chunker cuts passages with the embedding model's own tokenizer, baked in
# when the image was built. A build that could not reach huggingface.co still
# succeeds -- the console fetches the tokenizer on first use -- but BuildKit
# caches that step and folds its warning away, and every rebuild reuses it, so
# this is where it gets said. Offline, such an image cannot chunk at all.
# Snapshots only: .no_exist/ holds empty markers for files the hub lacks.
if [ -n "$(find "${HF_HOME:-/opt/hf-cache}" -path '*/snapshots/*' -name 'tokenizer*.json' -print -quit 2>/dev/null)" ]; then
    echo "  tokenizer: baked into the image"
else
    echo "  tokenizer: not in this image -- the console fetches it from"
    echo "             huggingface.co the first time it chunks a document,"
    echo "             and with no network that fails. The build could not"
    echo "             reach the hub; once it can, rebuild without the cache:"
    echo "             docker compose build --no-cache"
fi

exec "$@"
