#!/bin/bash
# Git pre-commit hook: run the fast pytest suite (`pytest tests/ -m "not slow" -q`)
# before allowing the commit through. Designed to live at .git/hooks/pre-commit;
# kept under tools/ so it can be version-controlled and installed per clone via
# `tools/install-pre-commit-hook.sh` (see tests/README.md).
#
# Wired up by Harmony_Test_Suite_Plan.md §5.3. If pytest is unavailable, the
# hook is a no-op — it does not block commits on machines without the lsl env.
#
# Bypass: `git commit --no-verify`. NEVER bypass on main / master.

set -u

# Pick the lsl conda env's pytest if available; otherwise PATH.
PYTEST_BIN="/home/arman-admin/opt/miniconda/envs/lsl/bin/pytest"
if [ ! -x "$PYTEST_BIN" ]; then
  PYTEST_BIN="$(command -v pytest || true)"
fi
if [ -z "${PYTEST_BIN:-}" ]; then
  echo "pre-commit-pytest: pytest not on PATH and lsl env missing — skipping." >&2
  exit 0
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "${repo_root}" ]; then
  exit 0
fi

cd "$repo_root" || exit 0

if [ ! -d "tests" ]; then
  exit 0
fi

echo "pre-commit-pytest: running fast suite (-m 'not slow') ..." >&2
"$PYTEST_BIN" tests/ -m "not slow" -q --tb=line
rc=$?

if [ "$rc" -ne 0 ]; then
  echo "" >&2
  echo "pre-commit-pytest: commit BLOCKED — fix the failing tests above and retry." >&2
  echo "  To bypass for an emergency (NEVER on main/master): git commit --no-verify" >&2
  exit "$rc"
fi

exit 0
