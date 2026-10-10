#!/usr/bin/env bash
# Which of the hosts an install downloads from does this network let through?
#
# On an open network the list is invisible. Behind an egress allowlist -- a
# company proxy, or a Claude Code cloud environment, whose Network access is
# set per environment -- each blocked host fails one step at a time, worded by
# whichever tool met it: pacman's refusal reads as a stale package database,
# and a tokenizer that never arrived as a chunking error at the first upload.
# This asks every host the named groups need, all at once, and prints the ones
# that do not answer as the allowlist entries that would let them through.
#
# It changes nothing. No script can allow a host: the allowlist belongs to the
# network, not to this machine.
#
# Usage:
#   scripts/network_check.sh GROUP... [--optional GROUP...]
#
#   Groups: arch pypi ollama hf tokenizer dockerhub github research cloud
#           npm claude playwright
#   A group after --optional is reported but never fails the check. Exit
#   status: 0 when every required group answers, 1 when one does not, 2 on a
#   group it does not know.
#
# install.sh and docker/install.sh run it as their first step, and the
# dockerfile before its first download.

set -uo pipefail

# group | hosts probed | allowlist entries | what needs them
#
# The entries are wider than the probes where a download is redirected to a
# CDN whose host is not fixed: Ollama's blobs come from Cloudflare R2, Hugging
# Face's from its Xet bridge under hf.co, Docker Hub's from Cloudflare. arch's
# hosts are whatever pacman's mirrorlist names. Claude Code's installer at
# claude.ai/install.sh redirects to its bootstrap on downloads.claude.ai, and
# the CLI, once installed, signs in through claude.ai and talks to
# api.anthropic.com. Playwright's own browser builds are needed only where no
# system chromium is.
TABLE='
arch|||system packages (pacman)
pypi|pypi.org files.pythonhosted.org|pypi.org files.pythonhosted.org|Python packages (pip, uv)
ollama|registry.ollama.ai|registry.ollama.ai *.r2.cloudflarestorage.com|library models (ollama pull)
hf|hf.co|hf.co *.hf.co huggingface.co *.huggingface.co|hf.co/ models (ollama pull)
tokenizer|huggingface.co|huggingface.co *.huggingface.co *.hf.co|the embedding tokenizer (the chunker)
dockerhub|registry-1.docker.io auth.docker.io|registry-1.docker.io auth.docker.io production.cloudflare.docker.com|Docker Hub images
github|github.com api.github.com|github.com api.github.com *.githubusercontent.com|GitHub (gh, git_dwell)
research|html.duckduckgo.com|duckduckgo.com *.duckduckgo.com|online research without a SearxNG
cloud|ollama.com|ollama.com|Ollama Cloud tags and ollama signin
npm|registry.npmjs.org|registry.npmjs.org|Node packages (npx @playwright/mcp)
claude|claude.ai downloads.claude.ai|claude.ai *.claude.ai claude.com *.claude.com api.anthropic.com|the Claude Code installer, sign-in and updates
playwright|cdn.playwright.dev|cdn.playwright.dev playwright.download.prss.microsoft.com|browser builds from Playwright (only without a system chromium)
'

MIRRORLIST="${MIRRORLIST:-/etc/pacman.d/mirrorlist}"

field() { awk -F'|' -v g="$1" -v n="$2" '$1 == g {print $n}' <<<"$TABLE"; }
known() { [ -n "$(field "$1" 1)" ]; }

# What is asked is an origin -- scheme, host and port -- not a host: a LAN
# mirror or a devpi on plain http, or on a port of its own, is asked where it
# serves. Asked on https:443 instead, it read as blocked and failed the check.
origin_of() {
    sed -nE 's#^[[:space:]]*([A-Za-z][A-Za-z0-9+.-]*)://([^/@]*@)?([^/?#[:space:]]+).*#\1://\3#p' <<<"$1" \
        | tr '[:upper:]' '[:lower:]'
}

# An origin as the report prints it: https is the default, so only the others
# keep their scheme.
shown() { echo "${1#https://}"; }

# An origin as an allowlist takes it: the host alone.
host_of() { sed -E 's#^[a-z0-9+.-]+://##; s#:[0-9]+$##' <<<"$1"; }

# Every server pacman would try, in its order; it moves to the next when one
# fails, so the group answers when any of them does.
arch_origins() {
    local url
    sed -n 's|^[[:space:]]*Server[[:space:]]*=[[:space:]]*||p' "$MIRRORLIST" 2>/dev/null \
        | while read -r url; do origin_of "$url"; done | awk 'NF && !seen[$0]++'
}

# A pip or uv pointed at another index downloads from it instead of PyPI.
index_url="${PIP_INDEX_URL:-${UV_DEFAULT_INDEX:-${UV_INDEX_URL:-}}}"
index_origin="$(origin_of "$index_url")"

origins_of() {
    case "$1" in
        arch) arch_origins ;;
        pypi) if [ -n "$index_origin" ]; then echo "$index_origin"; else field pypi 2 | tr ' ' '\n' | sed 's#^#https://#'; fi ;;
        *) field "$1" 2 | tr ' ' '\n' | sed '/^$/d; s#^#https://#' ;;
    esac
}

entries_of() {
    case "$1" in
        arch) host_of "$(arch_origins | head -n 1)" ;;
        pypi) if [ -n "$index_origin" ]; then host_of "$index_origin"; else field pypi 3 | tr ' ' '\n'; fi ;;
        *) field "$1" 3 | tr ' ' '\n' ;;
    esac
}

required=() optional=() after_optional=0
for arg in "$@"; do
    case "$arg" in
        --optional) after_optional=1 ;;
        -h|--help) sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"; exit 0 ;;
        *)
            known "$arg" || { echo "network_check: no group '$arg' (try --help)" >&2; exit 2; }
            if [ "$after_optional" -eq 1 ]; then optional+=("$arg"); else required+=("$arg"); fi
            ;;
    esac
done
[ "$(( ${#required[@]} + ${#optional[@]} ))" -gt 0 ] || { echo "network_check: name a group (try --help)" >&2; exit 2; }

# Without curl nothing can be asked -- and every probe's empty answer used to
# read as an answer, so the check passed on exactly the machines it was for.
if ! command -v curl >/dev/null 2>&1; then
    echo "  ✗ curl is not installed, so no host could be asked (sudo pacman -S curl)"
    [ "${#required[@]}" -eq 0 ]
    exit $?
fi

# Every origin at once, so the check takes as long as the slowest probe rather
# than the sum. Each probe keeps curl's exit code, which says why nothing came
# back -- and each reason has a different fix -- then the HTTP status (any
# status means the host answered) and the proxy's answer to CONNECT. The exit
# code comes first, so a curl that printed nothing cannot shift it into the
# status.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mapfile -t all_origins < <(for g in "${required[@]}" "${optional[@]}"; do origins_of "$g"; done | awk 'NF && !seen[$0]++')
declare -A probe=()
for origin in "${all_origins[@]}"; do
    probe[$origin]="$tmp/${#probe[@]}"
    {
        out="$(curl -s -o /dev/null -w '%{http_code} %{http_connect}' --connect-timeout 5 \
            --max-time 10 "$origin/" 2>/dev/null)"
        echo "$? $out"
    } >"${probe[$origin]}" &
done
wait

# Why an origin did not answer; nothing when it did.
why() {
    local rc code connect
    read -r rc code connect 2>/dev/null <"${probe[$1]:-/nonexistent}" || { echo "never asked"; return; }
    case "$code" in ""|000) ;; *) return ;; esac
    case "$rc" in
        56) if [ "$connect" != "000" ]; then echo "refused by the proxy, $connect"; else echo "dropped"; fi ;;
        5|6) echo "does not resolve" ;;
        7) echo "connection refused" ;;
        28) echo "timed out" ;;
        60) echo "certificate not trusted" ;;
        35) echo "TLS handshake failed" ;;
        126|127) echo "curl could not be run" ;;
        *) echo "curl exit $rc" ;;
    esac
}

# reached: something answered -- a host, a proxy's refusal, a TLS handshake.
failed=0 blocked_entries=() reached=0 untrusted=0 research_blocked=0
report() {  # group, 1 when optional
    local group="$1" is_optional="$2" what origins origin reason down=() up=() listed=0
    what="$(field "$group" 4)"
    mapfile -t origins < <(origins_of "$group")
    if [ "${#origins[@]}" -eq 0 ]; then
        echo "  - $what: no $MIRRORLIST here, so nothing to ask"
        return
    fi
    for origin in "${origins[@]}"; do
        reason="$(why "$origin")"
        if [ -z "$reason" ]; then
            up+=("$(shown "$origin")")
            reached=1
            continue
        fi
        down+=("$(shown "$origin") ($reason)")
        case "$reason" in
            "refused by the proxy"*) reached=1; listed=1 ;;
            # The machine's to fix, not the allowlist's.
            "certificate not trusted") reached=1; untrusted=1 ;;
            "TLS handshake failed") reached=1; listed=1 ;;
            *) listed=1 ;;
        esac
    done
    # pacman needs one mirror that answers; everything else needs each host.
    if [ "${#down[@]}" -eq 0 ] || { [ "$group" = arch ] && [ "${#up[@]}" -gt 0 ]; }; then
        echo "  ✓ $what: ${up[*]}"
        return
    fi
    if [ "$is_optional" -eq 1 ]; then
        echo "  - $what (optional): ${down[*]}"
    else
        echo "  ✗ $what: ${down[*]}"
        failed=1
    fi
    [ "$group" = research ] && research_blocked=1
    [ "$listed" -eq 1 ] && mapfile -t -O "${#blocked_entries[@]}" blocked_entries < <(entries_of "$group")
}
for g in "${required[@]}"; do report "$g" 0; done
for g in "${optional[@]}"; do report "$g" 1; done

if [ "$untrusted" -eq 1 ]; then
    echo "  A certificate this machine does not trust is a proxy that inspects TLS:"
    echo "  install that proxy's CA certificate here; allowing hosts will not help."
fi
if [ "${#blocked_entries[@]}" -gt 0 ]; then
    if [ "$reached" -eq 0 ]; then
        echo "  Nothing answered at all: is this machine online, or behind a proxy it"
        echo "  has not been told about (HTTPS_PROXY)?"
    else
        echo "  Nothing on this machine can let a host through; the network's allowlist"
        echo "  can (a company proxy, or a Claude Code cloud environment: Network access >"
        echo "  Custom > Allowed domains). The entries for what did not answer:"
        printf '      %s\n' "${blocked_entries[@]}" | awk '!seen[$0]++'
        if [ "$research_blocked" -eq 1 ]; then
            echo "  Online research also reads whatever pages a search finds, which no"
            echo "  allowlist can name: it needs open access."
        fi
    fi
fi
exit "$failed"
