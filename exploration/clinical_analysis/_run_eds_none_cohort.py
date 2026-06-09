"""One-off: regenerate the CLIN-cohort per-class EDS with spatial_filter="none"
(raw, online-matched), alongside the existing Hjorth default, for comparison.

The canonical CLI only exposes {car, csd, hjorth}; "none" is supported at the
function layer once apply_spatial_filter is patched to pass through (the same
patch the oneshot wrapper uses). Writes variant-tagged "_none" outputs to
~/Pictures/clin_analysis/eds/ so the Hjorth (un-suffixed) files are untouched.
Then re-renders the _none cohort MI map in the manuscript publication style
(matching make_publication_figures.fig_eds_topomap_mi) to the preprint figures/
dir as fig_eds_topomap_mi_none.png, keeping fig_eds_topomap_mi.png (Hjorth).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

_REPO = Path(r"C:\Users\arman\Projects\BCI")
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import Analyze_eds_topoplot_CLIN as eds

# Pass-through for the un-referenced (online-matched) covariance path.
_orig_asf = eds.apply_spatial_filter
eds.apply_spatial_filter = (
    lambda epochs, method: epochs if method == "none" else _orig_asf(epochs, method))

out_dir = eds.clin_pictures_root() / "eds"
out_dir.mkdir(parents=True, exist_ok=True)
cohort = list(eds.CLIN_PRIMARY_SUBJECTS)          # n=6, excludes CLIN_SUBJ_002
band = (eds.MU_LO, eds.MU_HI)
variant_tag = eds._build_variant_tag("motor15", "drop_fp", "none")  # -> "_none"
print(f"cohort={cohort}  variant_tag='{variant_tag}'")

eds.run_for_band_per_class(
    "mu", band, cohort, out_dir,
    include_clin002=False, channel_set="motor15",
    blink_removal="drop_fp", spatial_filter="none", variant_tag=variant_tag,
)

# ---- Publication-style render of the _none cohort MI map ----
csv = out_dir / f"eds_per_class_cohort_summary_mu{variant_tag}.csv"
df = pd.read_csv(csv)
d = df[(df["band"] == "mu") & (df["class"] == "mi")].copy()
ch_names = list(d["channel"])
values = d["cohort_z_score"].to_numpy(dtype=float)
info = mne.create_info(ch_names, sfreq=512.0, ch_types="eeg")
info.set_montage("standard_1020", match_case=False)
fig, ax = plt.subplots(figsize=(5.2, 4.6))
vmax = float(np.nanmax(np.abs(values)))
im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap="viridis",
                             vlim=(-vmax, vmax), names=ch_names, contours=6,
                             sensors=True)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
cbar.set_label("EDS (z-score)")
ax.set_title("Cohort MI EDS - no spatial filter (online-matched)", fontsize=10)
dst = Path(r"C:\Users\arman\Documents\harmony-als-preprint\figures") / "fig_eds_topomap_mi_none.png"
fig.savefig(dst, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"wrote pub-style none map: {dst}")
print("DRIVER_DONE")
