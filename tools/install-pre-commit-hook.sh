#!/bin/bash
# Bootstrap the per-clone git pre-commit hook that runs the fast pytest
# suite on every commit. Run this once per clone, per
# Harmony_Test_Suite_Plan.md §5.3.
#
#   ./tools/install-pre-commit-hook.sh
#
# This symlinks .git/hooks/pre-commit to tools/pre-commit-pytest.sh so that
# updates to the hook script flow in via normal git pulls.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
src="$repo_root/tools/pre-commit-pytest.sh"
dst="$repo_root/.git/hooks/pre-commit"

if [ ! -x "$src" ]; then
  echo "error: $src is missing or not executable." >&2
  exit 1
fi

if [ -e "$dst" ] && [ ! -L "$dst" ]; then
  backup="$dst.bak.$(date +%Y%m%d-%H%M%S)"
  echo "existing non-symlink pre-commit hook at $dst, backing up to $backup" >&2
  mv "$dst" "$backup"
fi

ln -sf "$src" "$dst"
echo "installed: $dst -> $src"
