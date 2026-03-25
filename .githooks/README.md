# Git hooks (optional)

Harmony ships a **non-destructive** `pre-commit` hook: it only prints **hints** when you stage changes under common config/runtime paths but do **not** stage `CHANGELOG.md` (and sometimes `README.md`). It does **not** rewrite files or append commits.

Enable for this repository:

```bash
git config core.hooksPath .githooks
```

Disable (use default `.git/hooks`):

```bash
git config --unset core.hooksPath
```

**Why not auto-update CHANGELOG/README?** Reliable summaries need human judgment (and often break CI if a hook edits tracked files mid-commit). A reminder keeps the repo clean while nudging discipline.
