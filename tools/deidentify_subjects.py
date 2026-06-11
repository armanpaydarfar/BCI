#!/usr/bin/env python3
"""De-identify healthy-subject names in the BCI study data directory.

Some healthy-subject pseudonyms encode an identifying cohort/semester/initial:
``CLASS_SUBJ_*`` (a class cohort), ``F25CLASS_SUBJ_*`` (Fall-2025 class),
``S26CLASS_SUBJ_*`` (Spring-2026 class) and ``LAB_SUBJ_*`` (lab members by
initial). Combined with a small cohort that is a weak re-identification vector.
This tool replaces those tokens with flat, randomized pseudonyms ``SUBJ_NNN``.

Left untouched: ``CLIN_*`` (clinical cohort, already generic), ``PILOT*``,
``P503``, ``TESTING_BP_NOISE`` and every non-subject token.

The mapping is frozen once into ``deid_key.csv`` (written under DATA_DIR/_deid,
i.e. OUTSIDE the git repo) and is the single source of truth for every later
run -- including the Linux machine. Same key + same script => identical
renaming on every host. The script body contains no PII; the key does, and the
key is never committed.

Subcommands:
  discover  scan DATA_DIR, derive the identifying token set, generate the key
            if absent, and write a dry-run change manifest. Touches nothing.
  snapshot  hash every (non-zip) file under DATA_DIR into pre_manifest.json,
            recording each file's mapped post-rename path and (for edited text
            files) the expected post-edit hash. The pre-rename audit baseline.
  apply     drive the rename from the frozen key: rewrite text-log contents,
            then rename files and folders deepest-first, logging every step to
            apply_rollback.jsonl. Run snapshot first.
  validate  re-hash the post-rename tree and assert every file matches its
            expected mapped path + hash from pre_manifest.json (byte-identical
            payload proof), plus a zero-residual-token scan.

Rename is metadata-only on a single volume, so binary payloads (.xdf/.pkl) are
byte-identical by construction; the hash manifest proves it empirically. Only
text-log contents are rewritten, and those are validated by forward-applying
the same mapping to the original bytes recorded at snapshot time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field

# --- repo config (DATA_DIR) -------------------------------------------------
# config.py imports config_local.py for the machine-local DATA_DIR. We add the
# repo root to sys.path so this works regardless of the invoking cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # noqa: E402

# Identifying subject tokens carry one of these cohort prefixes. CLIN and PILOT
# are deliberately absent: they are not renamed. FS25CLASS / FSCLASS are typo
# spellings of F25CLASS observed in the data; they are canonicalized below so a
# misspelled file maps to the SAME new id as its correctly-spelled sibling.
_IDENTIFYING_TOKEN_RE = re.compile(
    r"^(CLASS|F25CLASS|S26CLASS|FS25CLASS|FSCLASS|LAB)_SUBJ_([A-Za-z0-9]+)$"
)
_PREFIX_NORMALIZE = {"FS25CLASS": "F25CLASS", "FSCLASS": "F25CLASS"}
# Deterministic cohort ordering for the pre-shuffle canonical list.
_PREFIX_ORDER = {"CLASS": 0, "F25CLASS": 1, "S26CLASS": 2, "LAB": 3}

# Fixed seed: the shuffle only needs to break the number<->cohort correlation;
# recording the seed keeps key generation reproducible and auditable.
_SHUFFLE_SEED = 20260610

# Text file types whose CONTENT may name a subject. Everything else (.xdf,
# .pkl, .png, .zip, .npy ...) is treated as opaque binary and never read.
_TEXT_SUFFIXES = {".json", ".txt", ".csv", ".log", ".py", ".md", ".tsv", ".yaml", ".yml"}
_TEXT_NAME_PREFIXES = ("notes",)  # extensionless session notes
_MAX_TEXT_BYTES = 64 * 1024 * 1024  # skip anything larger; logs are far below

_DEID_DIRNAME = "_deid"  # output dir under DATA_DIR; excluded from scans


def _is_text_file(name: str) -> bool:
    lower = name.lower()
    if os.path.splitext(lower)[1] in _TEXT_SUFFIXES:
        return True
    return any(lower.startswith(p) for p in _TEXT_NAME_PREFIXES)


def _canonical(spelling: str) -> str:
    """Map any identifying spelling (incl. typos) to its canonical subject id."""
    m = _IDENTIFYING_TOKEN_RE.match(spelling)
    if not m:
        raise ValueError(f"not an identifying token: {spelling!r}")
    prefix, num = m.group(1), m.group(2)
    prefix = _PREFIX_NORMALIZE.get(prefix, prefix)
    return f"{prefix}_SUBJ_{num}"


def _iter_path_tokens(name: str):
    """Yield identifying spellings embedded in a file/dir name.

    Tokens appear as ``sub-<TOKEN>`` (folders) or ``sub-<TOKEN>_ses-...``
    (recordings). We extract the segment after ``sub-`` up to the next ``_ses``
    boundary or end-of-name, then test it against the identifying pattern.
    """
    for m in re.finditer(r"sub-([A-Za-z0-9]+_SUBJ_[A-Za-z0-9]+)", name):
        tok = m.group(1)
        if _IDENTIFYING_TOKEN_RE.match(tok):
            yield tok


def _iter_content_tokens(text: str):
    """Yield (spelling, count) for identifying tokens in text content.

    Matches the bare ``<PREFIX>_SUBJ_<n>`` token with non-alphanumeric
    boundaries so that, e.g., ``CLASS_SUBJ_<n>`` is NOT matched inside
    ``F25CLASS_SUBJ_<n>`` (the ``5`` before ``CLASS`` blocks it).
    """
    counts: dict[str, int] = defaultdict(int)
    for m in re.finditer(
        r"(?<![A-Za-z0-9])(CLASS|F25CLASS|S26CLASS|FS25CLASS|FSCLASS|LAB)_SUBJ_([A-Za-z0-9]+)(?![A-Za-z0-9])",
        text,
    ):
        counts[m.group(0)] += 1
    return counts


@dataclass
class Scan:
    spellings: set[str] = field(default_factory=set)
    name_hits: list[tuple[str, list[str]]] = field(default_factory=list)  # (relpath, tokens)
    content_hits: list[tuple[str, dict[str, int]]] = field(default_factory=list)  # (relpath, {tok:count})
    n_files: int = 0
    n_dirs: int = 0


def scan_tree(root: str, *, read_content: bool = True) -> Scan:
    """Walk ``root`` collecting identifying-token occurrences in names + content."""
    s = Scan()
    for dirpath, dirnames, filenames in os.walk(root):
        # Never descend into the de-id output dir or zip-extracted scratch we
        # are told to leave alone is handled by name (zips aren't dirs anyway).
        dirnames[:] = [d for d in dirnames if d != _DEID_DIRNAME]
        for d in dirnames:
            s.n_dirs += 1
            toks = list(_iter_path_tokens(d))
            if toks:
                rel = os.path.relpath(os.path.join(dirpath, d), root)
                s.name_hits.append((rel, toks))
                s.spellings.update(toks)
        for fn in filenames:
            s.n_files += 1
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            toks = list(_iter_path_tokens(fn))
            if toks:
                s.name_hits.append((rel, toks))
                s.spellings.update(toks)
            if read_content and _is_text_file(fn):
                try:
                    if os.path.getsize(full) > _MAX_TEXT_BYTES:
                        continue
                    with open(full, "r", encoding="utf-8", errors="replace") as fh:
                        text = fh.read()
                except OSError as e:
                    print(f"  [warn] cannot read {rel}: {e}", file=sys.stderr)
                    continue
                counts = _iter_content_tokens(text)
                if counts:
                    s.content_hits.append((rel, dict(counts)))
                    s.spellings.update(counts.keys())
    return s


def build_key(spellings: set[str]) -> dict:
    """Generate the frozen mapping: canonical subject -> SUBJ_NNN (randomized).

    Returns a dict with ``seed``, ``subjects`` (list of records) and a flat
    ``spelling_to_new`` map covering every observed spelling incl. typos.
    """
    canon_to_spellings: dict[str, set[str]] = defaultdict(set)
    for sp in spellings:
        canon_to_spellings[_canonical(sp)].add(sp)

    canonicals = sorted(
        canon_to_spellings,
        key=lambda c: (
            _PREFIX_ORDER[c.split("_SUBJ_")[0]],
            # numeric where possible, else lexical (handles non-numeric suffixes)
            (0, int(c.split("_SUBJ_")[1])) if c.split("_SUBJ_")[1].isdigit() else (1, c.split("_SUBJ_")[1]),
        ),
    )
    order = list(canonicals)
    random.Random(_SHUFFLE_SEED).shuffle(order)
    new_for_canon = {c: f"SUBJ_{i:03d}" for i, c in enumerate(order, start=1)}

    subjects = []
    spelling_to_new: dict[str, str] = {}
    for c in canonicals:
        new = new_for_canon[c]
        sps = sorted(canon_to_spellings[c])
        for sp in sps:
            spelling_to_new[sp] = new
        subjects.append({
            "new_id": new,
            "canonical_old": c,
            "cohort_prefix": c.split("_SUBJ_")[0],
            "spellings": sps,
        })
    subjects.sort(key=lambda r: r["new_id"])
    return {"seed": _SHUFFLE_SEED, "subjects": subjects, "spelling_to_new": spelling_to_new}


def _rename_target(name: str, spelling_to_new: dict[str, str]) -> str:
    """Apply the mapping to a file/dir name (boundary-aware, longest-first)."""
    out = name
    for sp in sorted(spelling_to_new, key=len, reverse=True):
        out = re.sub(
            r"(?<![A-Za-z0-9])" + re.escape(sp) + r"(?![A-Za-z0-9])",
            spelling_to_new[sp],
            out,
        )
    return out


def write_key_csv(path: str, key: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["new_id", "canonical_old", "cohort_prefix", "spellings", "seed"])
        for r in key["subjects"]:
            w.writerow([r["new_id"], r["canonical_old"], r["cohort_prefix"],
                        ";".join(r["spellings"]), key["seed"]])


def discover(data_dir: str, repo_root: str) -> None:
    out_dir = os.path.join(data_dir, _DEID_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)
    key_path = os.path.join(out_dir, "deid_key.csv")

    print(f"Scanning DATA_DIR: {data_dir}")
    data_scan = scan_tree(data_dir, read_content=True)
    print(f"  {data_scan.n_files} files, {data_scan.n_dirs} dirs scanned")
    print(f"Scanning REPO (report-only): {repo_root}")
    repo_scan = scan_tree(repo_root, read_content=True)

    if not data_scan.spellings:
        print("No identifying tokens found. Nothing to do.")
        return

    key = build_key(data_scan.spellings)
    s2n = key["spelling_to_new"]

    if os.path.exists(key_path):
        print(f"\n[!] Key already exists at {key_path} -- NOT regenerating. "
              f"The dry-run manifest below uses the freshly-derived mapping; "
              f"reconcile before apply if they differ.")
    else:
        write_key_csv(key_path, key)
        print(f"\nProposed key written: {key_path}")

    # --- assemble dry-run manifest ---
    dir_renames, file_renames = [], []
    for rel, toks in sorted(data_scan.name_hits):
        base = os.path.basename(rel)
        new_base = _rename_target(base, s2n)
        if new_base != base:
            new_rel = os.path.join(os.path.dirname(rel), new_base)
            (dir_renames if os.path.isdir(os.path.join(data_dir, rel)) else file_renames).append((rel, new_rel))

    content_edits = []
    for rel, counts in sorted(data_scan.content_hits):
        mapped = {tok: s2n[tok] for tok in counts}
        content_edits.append({"path": rel, "tokens": counts, "new": mapped})

    repo_occurrences = []
    for rel, toks in sorted(repo_scan.name_hits):
        repo_occurrences.append({"path": rel, "kind": "name", "tokens": sorted(set(toks))})
    for rel, counts in sorted(repo_scan.content_hits):
        repo_occurrences.append({"path": rel, "kind": "content", "tokens": counts})

    manifest = {
        "data_dir": data_dir,
        "n_subjects": len(key["subjects"]),
        "key": key["subjects"],
        "dir_renames": [{"from": a, "to": b} for a, b in dir_renames],
        "file_renames": [{"from": a, "to": b} for a, b in file_renames],
        "content_edits": content_edits,
        "repo_occurrences": repo_occurrences,
    }
    manifest_path = os.path.join(out_dir, "dry_run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # --- human summary ---
    print("\n" + "=" * 70)
    print(f"PROPOSED KEY  ({len(key['subjects'])} subjects, seed={key['seed']})")
    print("=" * 70)
    for r in key["subjects"]:
        sp = ", ".join(r["spellings"])
        print(f"  {r['new_id']}  <=  {sp}")
    print("\nCHANGE SUMMARY (DATA_DIR)")
    print(f"  directory renames : {len(dir_renames)}")
    print(f"  file renames      : {len(file_renames)}")
    print(f"  content edits     : {len(content_edits)} files "
          f"({sum(sum(c.values()) for _, c in data_scan.content_hits)} token occurrences)")
    print("\nREPO OCCURRENCES (report-only, NOT auto-edited)")
    for o in repo_occurrences:
        toks = o["tokens"] if isinstance(o["tokens"], list) else sorted(o["tokens"])
        print(f"  [{o['kind']}] {o['path']}  ->  {', '.join(toks)}")
    print(f"\nFull manifest: {manifest_path}")
    # sanity: every subject should be SUBJ_001..NNN contiguous
    ids = sorted(r["new_id"] for r in key["subjects"])
    expected = [f"SUBJ_{i:03d}" for i in range(1, len(ids) + 1)]
    if ids != expected:
        print(f"\n[!] new-id set is not contiguous SUBJ_001..{len(ids):03d}: {ids}")


# ---------------------------------------------------------------------------
# snapshot / apply / validate  (driven by the frozen key)
# ---------------------------------------------------------------------------

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_key(key_path: str) -> dict[str, str]:
    """Load the frozen mapping: every observed spelling -> SUBJ_NNN.

    Also merges deid_overrides.csv (old_substring,new_substring) from the same
    directory if present. Overrides resolve rename collisions where two source
    spellings of one subject would map to the same path (e.g. a typo-spelled
    legacy model alongside the canonical one); they are longer, more specific
    strings and win because _rename_target applies replacements longest-first.
    The override file is part of the private key and must travel with it.
    """
    if not os.path.exists(key_path):
        raise SystemExit(f"key not found: {key_path} -- run `discover` first")
    s2n: dict[str, str] = {}
    with open(key_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for sp in filter(None, row["spellings"].split(";")):
                s2n[sp] = row["new_id"]
    ov_path = os.path.join(os.path.dirname(key_path), "deid_overrides.csv")
    if os.path.exists(ov_path):
        with open(ov_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                s2n[row["old_substring"]] = row["new_substring"]
    return s2n


def _iter_data_files(data_dir: str):
    """Yield (fullpath, relpath) for every file under data_dir except _deid."""
    for dirpath, dirnames, filenames in os.walk(data_dir):
        dirnames[:] = [d for d in dirnames if d != _DEID_DIRNAME]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            yield full, os.path.relpath(full, data_dir)


def _edited_text(raw: bytes, name: str, s2n: dict[str, str]) -> bytes | None:
    """Return rewritten bytes if `raw` is editable text containing a token.

    Reads/writes raw bytes with explicit UTF-8 so Windows newline translation
    never perturbs the payload. Returns None when nothing changes.
    """
    if not _is_text_file(name):
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None  # text suffix but not UTF-8 -> treat as opaque, never edit
    new = _rename_target(text, s2n)
    return new.encode("utf-8") if new != text else None


def snapshot(data_dir: str) -> None:
    out_dir = os.path.join(data_dir, _DEID_DIRNAME)
    key_path = os.path.join(out_dir, "deid_key.csv")
    s2n = load_key(key_path)
    with open(key_path, "rb") as fh:
        key_sha = _sha256_bytes(fh.read())

    # Hash cache: on a re-run (e.g. after adding an override), reuse the source
    # hash for any binary file whose (rel, size, mtime) is unchanged from the
    # prior manifest, instead of re-reading 100+ GB. Text files are always read
    # fresh -- they are small and their post-edit hash depends on the mapping.
    cache: dict[str, tuple[int, int, str]] = {}
    path = os.path.join(out_dir, "pre_manifest.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for e in json.load(fh)["entries"]:
                if "mtime" in e:
                    cache[e["rel"]] = (e["size"], e["mtime"], e["sha"])

    entries, zips, nbytes = [], [], 0
    for full, rel in _iter_data_files(data_dir):
        name = os.path.basename(rel)
        if name.lower().endswith(".zip"):
            zips.append({"path": rel, "size": os.path.getsize(full)})
            continue
        st = os.stat(full)
        size, mtime = st.st_size, int(st.st_mtime)
        is_text = _is_text_file(name)
        cached = cache.get(rel)
        if cached and cached[0] == size and cached[1] == mtime and not is_text:
            sha, post_bytes = cached[2], None  # binary, unchanged -> reuse hash
        else:
            with open(full, "rb") as fh:
                raw = fh.read()
            sha = _sha256_bytes(raw)
            post_bytes = _edited_text(raw, name, s2n)
        nbytes += size
        entries.append({
            "rel": rel,
            "size": size,
            "mtime": mtime,
            "sha": sha,
            "post_rel": _rename_target(rel, s2n),
            "post_sha": _sha256_bytes(post_bytes) if post_bytes is not None else sha,
            "edited": post_bytes is not None,
        })

    manifest = {
        "data_dir": data_dir, "key_sha": key_sha,
        "n_files": len(entries), "n_bytes": nbytes,
        "zips": zips, "entries": entries,
    }
    path = os.path.join(out_dir, "pre_manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    n_ren = sum(1 for e in entries if e["post_rel"] != e["rel"])
    n_edit = sum(1 for e in entries if e["edited"])
    # collision guard: two distinct sources must not map to one post path.
    post_paths = [e["post_rel"] for e in entries]
    dupes = len(post_paths) - len(set(post_paths))
    print(f"snapshot: {len(entries)} files ({nbytes/1024**3:.1f} GB), "
          f"{len(zips)} zips skipped")
    print(f"  path remaps: {n_ren}   content edits: {n_edit}   key_sha={key_sha[:12]}")
    if dupes:
        print(f"  [!] {dupes} post-path collisions -- ABORT before apply")
    print(f"  wrote {path}")


def apply(data_dir: str) -> None:
    out_dir = os.path.join(data_dir, _DEID_DIRNAME)
    s2n = load_key(os.path.join(out_dir, "deid_key.csv"))

    # Refuse to run unless a clean snapshot exists with no path collisions; the
    # manifest is also the validate baseline, so it must predate any change.
    man_path = os.path.join(out_dir, "pre_manifest.json")
    if not os.path.exists(man_path):
        raise SystemExit("no pre_manifest.json -- run `snapshot` first")
    with open(man_path, encoding="utf-8") as fh:
        posts = [e["post_rel"] for e in json.load(fh)["entries"]]
    if len(posts) != len(set(posts)):
        raise SystemExit("snapshot has post-path collisions -- resolve before apply")

    # Guard for cross-machine runs: the key is frozen from one host, so a second
    # host could hold an identifying subject the key never saw. Abort rather than
    # silently leave it un-renamed (which validate would only catch afterwards).
    unmapped = set()
    for full, rel in _iter_data_files(data_dir):
        name = os.path.basename(rel)
        for tok in _iter_path_tokens(name):
            if tok not in s2n:
                unmapped.add(tok)
        if _is_text_file(name):
            try:
                text = open(full, "rb").read().decode("utf-8")
            except UnicodeDecodeError:
                continue
            for tok in _iter_content_tokens(text):
                if tok not in s2n:
                    unmapped.add(tok)
    if unmapped:
        raise SystemExit(
            "identifying tokens not in deid_key.csv -- extend the key (append "
            f"rows, do NOT regenerate) before apply: {sorted(unmapped)}")

    rollback_path = os.path.join(out_dir, "apply_rollback.jsonl")
    rb = open(rollback_path, "a", encoding="utf-8")
    backup_dir = os.path.join(out_dir, "pre_text_backup")

    # 1. content edits first, while paths are still stable. Each original is
    #    copied verbatim to pre_text_backup/<rel> so `revert` can restore it
    #    byte-for-byte -- the forward mapping is many-spellings-to-one and so
    #    not exactly invertible from the edited bytes alone.
    n_edit = 0
    for full, rel in _iter_data_files(data_dir):
        with open(full, "rb") as fh:
            raw = fh.read()
        post = _edited_text(raw, os.path.basename(rel), s2n)
        if post is not None:
            bpath = os.path.join(backup_dir, rel)
            os.makedirs(os.path.dirname(bpath), exist_ok=True)
            with open(bpath, "wb") as fh:
                fh.write(raw)
            with open(full, "wb") as fh:
                fh.write(post)
            rb.write(json.dumps({"op": "edit", "rel": rel,
                                 "old_sha": _sha256_bytes(raw)}) + "\n")
            n_edit += 1

    # 2. rename files+dirs deepest-first so a child's basename is rewritten
    #    while its parent still has the old name, then the parent moves it.
    paths = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        dirnames[:] = [d for d in dirnames if d != _DEID_DIRNAME]
        for n in list(dirnames) + filenames:
            paths.append(os.path.join(dirpath, n))
    paths.sort(key=lambda p: p.count(os.sep), reverse=True)
    n_ren = 0
    for old in paths:
        d, n = os.path.split(old)
        nn = _rename_target(n, s2n)
        if nn != n:
            new = os.path.join(d, nn)
            os.rename(old, new)
            rb.write(json.dumps({"op": "rename",
                                 "from": os.path.relpath(old, data_dir),
                                 "to": os.path.relpath(new, data_dir)}) + "\n")
            n_ren += 1
    rb.close()
    print(f"apply: {n_edit} files content-edited, {n_ren} paths renamed")
    print(f"  rollback log: {rollback_path}")


def revert(data_dir: str) -> None:
    """Undo an apply: reverse renames (newest-first), then restore text backups."""
    out_dir = os.path.join(data_dir, _DEID_DIRNAME)
    rollback_path = os.path.join(out_dir, "apply_rollback.jsonl")
    if not os.path.exists(rollback_path):
        raise SystemExit("no apply_rollback.jsonl -- nothing to revert")
    ops = [json.loads(line) for line in open(rollback_path, encoding="utf-8")]
    n_ren = n_edit = 0
    for op in reversed(ops):  # reverse order: renames undone before edits
        if op["op"] == "rename":
            src = os.path.join(data_dir, op["to"])
            dst = os.path.join(data_dir, op["from"])
            if os.path.exists(src):
                os.rename(src, dst)
                n_ren += 1
        elif op["op"] == "edit":
            backup = os.path.join(out_dir, "pre_text_backup", op["rel"])
            dst = os.path.join(data_dir, op["rel"])
            if os.path.exists(backup):
                with open(backup, "rb") as fh:
                    raw = fh.read()
                with open(dst, "wb") as fh:
                    fh.write(raw)
                n_edit += 1
    print(f"revert: {n_ren} renames undone, {n_edit} text files restored")
    print("  delete apply_rollback.jsonl + pre_text_backup/ once confirmed.")


def validate(data_dir: str) -> None:
    out_dir = os.path.join(data_dir, _DEID_DIRNAME)
    s2n = load_key(os.path.join(out_dir, "deid_key.csv"))
    with open(os.path.join(out_dir, "pre_manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)

    fails, ok = [], 0
    expected = {}
    for e in man["entries"]:
        expected[e["post_rel"]] = e
        full = os.path.join(data_dir, e["post_rel"])
        if not os.path.exists(full):
            fails.append(("missing", e["post_rel"]))
            continue
        with open(full, "rb") as fh:
            sha = _sha256_bytes(fh.read())
        if sha != e["post_sha"]:
            fails.append(("sha-mismatch", e["post_rel"]))
        else:
            ok += 1

    # no unexpected files (excluding the zips we never touched and _deid)
    for full, rel in _iter_data_files(data_dir):
        if os.path.basename(rel).lower().endswith(".zip"):
            continue
        if rel not in expected:
            fails.append(("unexpected", rel))

    # zero residual identifying tokens, in names or text content
    residual = []
    for full, rel in _iter_data_files(data_dir):
        name = os.path.basename(rel)
        if list(_iter_path_tokens(name)):
            residual.append(("name", rel))
        if _is_text_file(name):
            try:
                text = open(full, "rb").read().decode("utf-8")
            except UnicodeDecodeError:
                continue
            if _iter_content_tokens(text):
                residual.append(("content", rel))

    print(f"validate: {ok}/{man['n_files']} files match expected path+hash")
    if fails:
        print(f"  [FAIL] {len(fails)} discrepancies:")
        for kind, p in fails[:40]:
            print(f"    {kind}: {p}")
        if len(fails) > 40:
            print(f"    ... +{len(fails) - 40} more")
    if residual:
        print(f"  [FAIL] {len(residual)} residual identifying tokens:")
        for kind, p in residual[:40]:
            print(f"    {kind}: {p}")
    if not fails and not residual:
        print("  PASS: every payload byte-identical under its mapped name; "
              "no residual identifying tokens.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("discover", help="scan + propose key + dry-run manifest (no changes)")
    sub.add_parser("snapshot", help="hash every file into pre_manifest.json (no changes)")
    sub.add_parser("apply", help="rename + content-edit per the frozen key")
    sub.add_parser("revert", help="undo the last apply from the rollback log")
    sub.add_parser("validate", help="re-hash post-rename tree vs pre_manifest.json")
    args = ap.parse_args()

    data_dir = config.DATA_DIR
    if not data_dir or not os.path.isdir(data_dir):
        raise SystemExit(f"DATA_DIR not set or missing: {data_dir!r} (see config_local.py)")

    {"discover": lambda: discover(data_dir, _REPO_ROOT),
     "snapshot": lambda: snapshot(data_dir),
     "apply": lambda: apply(data_dir),
     "revert": lambda: revert(data_dir),
     "validate": lambda: validate(data_dir)}[args.cmd]()


if __name__ == "__main__":
    main()
