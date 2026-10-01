# syntax=docker/dockerfile:1

# The console as an Arch image.
#
# Arch rather than a python:* base because this project is an Arch/Omarchy
# project: `install.sh` names pacman packages, the interpreter it resolves
# against is Arch's current one (3.14), and the versions this repo is actually
# run on -- numpy 2.5, scipy 1.18, chromadb 1.5, transformers 5 -- are the ones
# that stack resolves to there. A Debian-based image would be a second,
# untested resolution of the same requirements.
#
# Three stages: `base` holds the system packages both halves need, `build`
# compiles the virtualenv (and only it needs base-devel, ~700 MB of toolchain),
# `runtime` takes the finished venv and none of the toolchain.

# ---------------------------------------------------------------------------
# base: the system packages
# ---------------------------------------------------------------------------
FROM archlinux:base AS base

# The runtime half of install.sh's REQUIRED list. What is deliberately absent:
#   ollama      -- the daemon stays on the host. It holds the weights every
#                  default seat runs, owns the embedding model's placement on
#                  the GPU, and holds the ollama.com credentials for any
#                  `:cloud` tag; this image only ever speaks HTTP to it
#                  (OLLAMA_BASE_URL).
#   xdg-utils   -- a container has no browser to open.
#   base-devel  -- build stage only.
# postgresql-libs is here for the reason install.sh puts it on the host: psql
# is how the database configuration is *proved* rather than assumed, and it is
# the client a script the Builder writes will reach for.
RUN pacman -Syu --noconfirm --needed \
        python \
        git \
        github-cli \
        curl \
        postgresql-libs \
    && rm -rf /var/cache/pacman/pkg/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/hf-cache \
    PYTHONUNBUFFERED=1

# The installed tree ships no __pycache__: 285 MB of a 990 MB venv, and
# unwritable anyway once the image runs as a user who does not own /opt/venv --
# so every process would recompile the same modules and throw the result away.
# Pointed at a writable directory instead, the first process in a container
# pays the ~1.4s and every one after it reads the cache, which is the speed of
# shipping the bytecode without the image carrying it.
ENV PYTHONPYCACHEPREFIX=/home/agent/.cache/pycache

# ---------------------------------------------------------------------------
# build: the virtualenv
# ---------------------------------------------------------------------------
FROM base AS build

# base-devel for any wheel with no binary for this interpreter (and for
# `strip`, below). uv is the installer rather than pip: it resolves and
# installs this tree in a fraction of the time, which is the difference between
# a rebuild being something you do and something you avoid. pip is still in the
# venv -- `python -m venv` puts it there -- so the image can install a package
# at runtime the ordinary way.
RUN pacman -Syu --noconfirm --needed base-devel uv && rm -rf /var/cache/pacman/pkg/*

# A virtualenv rather than pip into the system interpreter: Arch marks its
# python externally-managed (PEP 668), and a venv is also what `install.sh`
# and `launch_console.sh` expect on a host, so the two layouts stay the same
# shape.
RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /app

# Only what the install reads, so an edit to serve.py or the frontend does not
# re-resolve the dependency tree. `[dev]` rather than the bare install because
# two of those extras are runtime requirements of the Builder's own belt, not
# developer conveniences: `_lint_written_files` runs `ruff check` over every
# Python file a pass wrote (a machine without ruff blocks nothing and says so,
# which would quietly retire the lint gate), and the `test_` tools run pytest.
COPY pyproject.toml ./pyproject.toml
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e ".[dev]"

# Two dependencies of chromadb that this project cannot reach, removed after
# the fact because there is no way to ask for the tree without them: 148 MB of
# 990. `onnxruntime` backs Chroma's own default embedding function, and nothing
# here uses it -- there is exactly one embedding model and it is served by the
# Ollama daemon over HTTP. `kubernetes` is its client for a Chroma running as a
# server, and this one runs embedded, as files under knowledge/.
# Measured rather than assumed: the whole suite (755 tests) passes in the image
# without them. Undo by deleting this line if a chromadb upgrade starts
# reaching for either.
RUN uv pip uninstall onnxruntime kubernetes

# Debug symbols from 415 compiled extensions: 80 MB, and nothing reads them
# here -- a segfault in scipy is not a thing this project debugs from inside
# its own container. Suite green after it, the same way.
RUN find /opt/venv -name "*.so" -exec strip --strip-unneeded {} + 2>/dev/null || true

# The chunker cuts passages with the embedding model's own tokenizer, loaded
# cache-first and fetched from Hugging Face when the cache misses. Baking it in
# means a container on a machine with no network still chunks -- and that the
# first run does not pay for a download at the moment it is indexing.
# Only an unreachable hub is forgiven: the runtime path already falls back to
# the network, so a build behind a proxy that cannot reach huggingface.co still
# produces a working image, one that fetches on first use. Anything else -- an
# import that moved, a renamed repository, a transformers API change -- fails
# the build, where a blanket `|| true` would ship a green image with no
# tokenizer in it. A forgiven step is cached like any other and its warning
# folds away in BuildKit's output, so the entrypoint says at every start
# whether the image has one. HF_HOME is made first because the runtime stage
# copies it, and a download that never connected leaves nothing there.
RUN mkdir -p "$HF_HOME" && python - <<'PY'
import httpx
from huggingface_hub.errors import LocalEntryNotFoundError
from transformers import AutoTokenizer

from langgraph_agent.graphrag_server import EMBEDDING_TOKENIZER_NAME


def unreachable(exc):
    # huggingface_hub reports a hub it could not reach -- refused, unresolvable,
    # timed out, or down -- as LocalEntryNotFoundError, and re-raises a proxy's
    # refusal as the httpx error itself; transformers wraps either in OSError.
    while exc is not None:
        if isinstance(exc, (LocalEntryNotFoundError, httpx.TransportError)):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


try:
    AutoTokenizer.from_pretrained(EMBEDDING_TOKENIZER_NAME)
except Exception as exc:
    if not unreachable(exc):
        raise
    print(f"WARNING: no tokenizer baked in, the hub is unreachable ({exc}); "
          f"the image fetches {EMBEDDING_TOKENIZER_NAME}'s on first use")
else:
    print(f"cached the tokenizer for {EMBEDDING_TOKENIZER_NAME}")
PY

# ---------------------------------------------------------------------------
# runtime: the console
# ---------------------------------------------------------------------------
FROM base AS runtime

COPY --from=build /opt/venv /opt/venv
COPY --from=build $HF_HOME $HF_HOME
# Readable and lockable by whatever uid the container is given: transformers
# takes a file lock beside a cached repo even when it reads it.
RUN chmod -R a+rwX "$HF_HOME"

# uid 1000 because that is the first human account on an Arch/Omarchy install,
# so the bind-mounted checkout is owned by the uid writing into it and the
# files a run produces are not left root-owned on the host. The home directory
# is world-writable so `--user` with some other uid still has somewhere to put
# a git config; nothing secret lives there.
RUN useradd --create-home --uid 1000 --shell /bin/bash agent \
    && chmod 0777 /home/agent

COPY docker/entrypoint.sh /usr/local/bin/ambiguity-entrypoint
COPY --chown=1000:1000 . /app

WORKDIR /app
ENV HOME=/home/agent \
    PORT=8080
EXPOSE 8080
USER 1000:1000

# The same readiness check launch_console.sh waits on, for the same reason:
# /api/status answers as soon as the server is up, and building a corpus does
# not block it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["sh", "-c", "curl -fsS \"http://127.0.0.1:${PORT}/api/status\" >/dev/null || exit 1"]

ENTRYPOINT ["/usr/local/bin/ambiguity-entrypoint"]
CMD ["python", "serve.py"]
