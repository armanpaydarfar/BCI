#!/usr/bin/env python3
"""Empirical equivalence check for the subject de-identification rename.

The only published-analysis computation whose *inputs* change under the rename
is the shared expert-decoder pool in ``Analyze_eds_topoplot_CLIN``: six OG_Right
training XDFs are renamed and the machine-local basename list
(``DATA_DIR/_deid/expert_pool_basenames.txt``) points at the new names.
Because the XDF payloads are byte-identical and read in fixed list order, the
expert EDS vector -- and the topomap rendered from it -- must be identical
before and after. The expert figure title carries only ``n_trials`` (no subject
name), so nothing in the plot itself should move either. This tool proves that.

  capture --tag pre|post   recompute the expert EDS vector and render the real
                           expert_eds_topoplot_mu panel via the analysis code.
  compare                  assert pre == post: exact array equality on the EDS
                           vector + channels + n_trials, and a pixel-identical
                           PNG.

Workflow: ``capture --tag pre`` BEFORE apply; run the rename; point
``expert_pool_basenames.txt`` at the new names; ``capture --tag post``; then
``compare``. Must run in the ``lsl`` conda env (matplotlib/mne/pyxdf).
"""

import argparse
import os
import sys

# The analysis module prints non-ASCII (≈, em dash); when stdout is redirected
# on Windows it defaults to cp1252 and those prints raise UnicodeEncodeError
# inside the per-file try/except, masking real loads as failures. Force UTF-8
# (no-op where stdout is already UTF-8, e.g. Linux).
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
import config  # noqa: E402
import Analyze_eds_topoplot_CLIN as eds  # noqa: E402

_OUT = os.path.join(config.DATA_DIR, "_deid", "equivalence")


def _zscore(v):
    """Per-channel z-score, matching the closure in Analyze_eds_topoplot_CLIN."""
    v = np.asarray(v)
    sd = v.std(ddof=1)
    return np.zeros_like(v) if sd == 0 else (v - v.mean()) / sd


def _expert():
    """Recompute the expert EDS using the analysis defaults (motor15/drop_fp/hjorth)."""
    return eds.expert_eds_shared(
        (eds.MU_LO, eds.MU_HI), eds.MOTOR_CHANNEL_NAMES, "mu",
        blink_removal=eds._DEFAULT_BLINK, spatial_filter=eds._DEFAULT_SPATIAL,
    )


def capture(tag: str) -> None:
    os.makedirs(_OUT, exist_ok=True)
    vec, channels, n = _expert()
    if vec is None:
        raise SystemExit("expert EDS returned None -- expert pool failed to load")
    png = os.path.join(_OUT, f"expert_eds_topoplot_mu_{tag}.png")
    eds._plot_topomap_panel(
        _zscore(vec), channels,
        f"Expert EDS (shared OG_Right pool, n_trials={n}) — mu", png,
    )
    np.savez(os.path.join(_OUT, f"expert_{tag}.npz"),
             vec=np.asarray(vec), channels=np.asarray(channels, dtype=object), n=n)
    print(f"capture[{tag}]: n_trials={n}, n_channels={len(channels)}, "
          f"vec[:3]={np.asarray(vec).ravel()[:3]}")
    print(f"  saved {png}")


def _load(tag: str):
    d = np.load(os.path.join(_OUT, f"expert_{tag}.npz"), allow_pickle=True)
    return np.asarray(d["vec"]), list(d["channels"]), int(d["n"])


def compare() -> None:
    import matplotlib.image as mpimg
    v0, c0, n0 = _load("pre")
    v1, c1, n1 = _load("post")
    vec_eq = v0.shape == v1.shape and np.array_equal(v0, v1)
    max_diff = float(np.max(np.abs(v0 - v1))) if v0.shape == v1.shape else float("nan")
    ch_eq = c0 == c1
    n_eq = n0 == n1
    p0 = mpimg.imread(os.path.join(_OUT, "expert_eds_topoplot_mu_pre.png"))
    p1 = mpimg.imread(os.path.join(_OUT, "expert_eds_topoplot_mu_post.png"))
    px_eq = p0.shape == p1.shape and np.array_equal(p0, p1)

    print(f"EDS vector exact-equal : {vec_eq}  (max|Δ|={max_diff:.2e})")
    print(f"channels equal         : {ch_eq}")
    print(f"n_trials equal         : {n_eq}  ({n0} vs {n1})")
    print(f"PNG pixel-identical    : {px_eq}")
    if vec_eq and ch_eq and n_eq and px_eq:
        print("PASS: expert pool + rendered figure identical before/after rename.")
    else:
        print("FAIL: divergence detected -- do NOT merge the repo edit.")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture", help="recompute + render expert EDS for a tag")
    cap.add_argument("--tag", required=True, choices=("pre", "post"))
    sub.add_parser("compare", help="assert pre == post (arrays + pixels)")
    args = ap.parse_args()
    if args.cmd == "capture":
        capture(args.tag)
    else:
        compare()


if __name__ == "__main__":
    main()
