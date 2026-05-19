# Git hooks (optional)

Harmony ships a `pre-commit` hook with two pieces:

1. **Soft hints** when you stage changes under common config/runtime
   paths but do **not** stage `CHANGELOG.md` (and sometimes `README.md`).
   These never block the commit.
2. **Hard pytest gate** (via `tools/pre-commit-pytest.sh`) — runs
   `pytest tests/ -m "not slow" -q` and blocks the commit if any test
   fails. Bypass with `git commit --no-verify` (never on `main` /
   `master`).

See `Documents/SoftwareDocs/Harmony_Test_Suite_Plan.md` §5.3 for the
test gate's history and policy.

Enable for this repository:

```bash
git config core.hooksPath .githooks
```

Disable (use default `.git/hooks`):

```bash
git config --unset core.hooksPath
```

**Why not auto-update CHANGELOG/README?** Reliable summaries need human judgment (and often break CI if a hook edits tracked files mid-commit). A reminder keeps the repo clean while nudging discipline.

**Cursor:** For agent sessions, use `.cursor/rules/finalize-documentation.md` so the agent updates docs when you ask to commit or push (Cursor has no separate server-side push hook for this).
