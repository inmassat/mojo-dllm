#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

binary=build/mojo-dllm
if [ ! -x "$binary" ]; then
    echo "check_build_id: build $binary first" >&2
    exit 1
fi

sha=unknown
if [ -e .git ] \
    && head="$(git rev-parse --short=12 HEAD 2>/dev/null)" \
    && changes="$(git status --porcelain -- src 2>/dev/null)"; then
    sha="$head${changes:+-dirty}"
fi
version="$(tr -d '\r' < src/main.mojo | sed -n 's/^comptime VERSION = "\([^"]*\)"$/\1/p')"
if [ -z "$version" ]; then
    echo "check_build_id: could not read VERSION from src/main.mojo" >&2
    exit 1
fi

expected="mojo-dllm $version"$'\n'"commit: $sha"
actual="$("$binary" --version)"
if [ "$actual" != "$expected" ]; then
    printf 'check_build_id: expected:\n%s\nactual:\n%s\n' "$expected" "$actual" >&2
    exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
MOJO="${MOJO:-uv run --frozen mojo}"
# shellcheck disable=SC2086
if ! $MOJO build -Werror --target-accelerator sm_80 -I src src/main.mojo -o "$tmp/mojo-dllm"; then
    exit 1
fi

actual="$("$tmp/mojo-dllm" --version)"
expected="mojo-dllm $version"$'\n'"commit: unknown"
if [ "$actual" != "$expected" ]; then
    printf 'check_build_id: no-define build expected:\n%s\nactual:\n%s\n' "$expected" "$actual" >&2
    exit 1
fi

echo "check_build_id: stamped and default version output passed"
