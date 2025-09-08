import os
import numpy as np
import pickle
import mne
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from pyriemann.estimation import Shrinkage
from pyriemann.classification import MDM, FgMDM
from pyriemann.estimation import Covariances, XdawnCovariances
import config
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from Utils.stream_utils import load_xdf, get_channel_names_from_xdf
from Utils.preprocessing import select_channels
import glob  # Required for multi-file loading
from scipy.stats import zscore
from pyriemann.utils.mean import mean_riemann
from scipy.linalg import sqrtm
import seaborn as sns
from sklearn.covariance import LedoitWolf
from pyriemann.preprocessing import Whitening
from Utils.preprocessing import (
    get_valid_channel_mask_and_metadata,
    initialize_filter_bank,
    apply_streaming_filters
)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load trigger mappings from config
TRIGGERS = config.TRIGGERS

# Define Relevant Markers for Classification (Exclude Robot Move - 300 & 320)
EPOCHS_START_END = {
    config.TRIGGERS["REST_BEGIN"]: config.TRIGGERS["REST_END"],  # 100 → 120
    config.TRIGGERS["MI_BEGIN"]: config.TRIGGERS["MI_END"],      # 200 → 220
}



def plot_posterior_probabilities(posterior_probs):
    """
    Plots the histogram of posterior probabilities for each class,
    with a dotted vertical line at the class mean.

    Parameters:
        posterior_probs (dict): Dictionary containing posterior probabilities for each class.
    """
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 20)  # Set bins for histogram

    # Convert numerical labels to "Rest" and "MI"
    label_map = {100: "Rest", 200: "MI"}
    renamed_probs = {label_map.get(int(label), str(label)): probs
                     for label, probs in posterior_probs.items()}

    # Define class colors (feel free to adjust)
    palette = {"Rest": "skyblue", "MI": "darkorange"}

    for label, probs in renamed_probs.items():
        probs = np.array(probs).flatten()
        color = palette.get(label, None)

        sns.histplot(probs, bins=bins, alpha=0.6, label=f"{label} Probability",
                     kde=True, color=color)

        # Add mean line in same color
        mean_val = np.mean(probs)
        plt.axvline(mean_val, color=color, linestyle="--", linewidth=1.5,
                    label=f"{label} Mean = {mean_val:.2f}")

    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Posterior Probability Distribution Across Classes")
    plt.legend(title="True Class")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def validate_trial_pairs(marker_values, marker_timestamps, eeg_timestamps, sfreq, EPOCHS_START_END, min_duration=1.0):
    """
    Validates trial start/end pairs and prints duration and skip-safety checks.

    Parameters:
        marker_values: np.array of int
        marker_timestamps: np.array of float (seconds)
        eeg_timestamps: np.array of float (seconds)
        sfreq: float sampling frequency
        EPOCHS_START_END: dict of {start_marker: end_marker}
        min_duration: float, minimum seconds required to allow 1s skip
    """
    print("\n🔍 Pre-validating trial start/end pairs...")
    
    for start_marker, end_marker in EPOCHS_START_END.items():
        start_indices = np.where(marker_values == int(start_marker))[0]
        end_indices = np.where(marker_values == int(end_marker))[0]

        print(f"\n🔹 Validating marker pair {start_marker} → {end_marker}")
        print(f"   Found {len(start_indices)} start markers, {len(end_indices)} end markers")

        if len(start_indices) != len(end_indices):
            print("   ⚠️ Mismatch in marker counts — trimming to shortest length")
            min_len = min(len(start_indices), len(end_indices))
            start_indices = start_indices[:min_len]
            end_indices = end_indices[:min_len]

        for i, (s_idx, e_idx) in enumerate(zip(start_indices, end_indices)):
            t_start = marker_timestamps[s_idx]
            t_end = marker_timestamps[e_idx]
            duration = t_end - t_start
            safe_to_skip = duration > min_duration

            if not safe_to_skip:
                print(f"   ❌ Trial {i}: {duration:.2f}s < {min_duration}s → will be invalid if 1s is skipped")
            else:
                print(f"   ✅ Trial {i}: {duration:.2f}s → OK to skip 1s")

        print(f"   Finished validating {len(start_indices)} trials for marker {start_marker}")

def segment_and_label_one_run(eeg_stream, marker_stream):
    """
    Preprocesses and segments EEG data for one run, closely replicating online preprocessing logic.
    Includes causal filtering with state tracking, baseline subtraction, and sliding window segmentation.

    Parameters:
        eeg_stream: dict containing EEG data and timestamps
        marker_stream: dict containing marker data and timestamps

    Returns:
        segments_all: np.ndarray of shape (n_segments, n_channels, n_samples)
        labels_all: np.ndarray of shape (n_segments,)
    """
    marker_values = np.array([int(m[0]) for m in marker_stream["time_series"]])
    marker_timestamps = np.array([float(m[1]) for m in marker_stream["time_series"]])
    eeg_timestamps = np.array(eeg_stream["time_stamps"])
    eeg_data = np.array(eeg_stream["time_series"]).T

    # Select valid EEG channels and metadata cleanup
    all_channel_names = get_channel_names_from_xdf(eeg_stream)  # raw labels from XDF

    valid_channel_names, valid_raw, valid_indices = get_valid_channel_mask_and_metadata(
        eeg_data, all_channel_names, fs=config.FS, drop_mastoids=True
    )

    # Map EEG data to the "valid" coordinate system (original -> valid)
    eeg_data = eeg_data[valid_indices, :]

    # Keep names in the same (normalized) space as valid_raw
    current_channel_names = list(valid_channel_names)
    print("channel names after metadata extraction:", current_channel_names)

    # --- Explicit motor subset (if enabled) ---
    if config.SELECT_MOTOR_CHANNELS:
        sel_raw = select_channels(valid_raw, keep_channels=config.MOTOR_CHANNEL_NAMES)
        subset_indices_valid = [current_channel_names.index(ch) for ch in sel_raw.ch_names]
        eeg_data = eeg_data[subset_indices_valid, :]
        current_channel_names = list(sel_raw.ch_names)

    # === Filter setup ===
    filter_bank = initialize_filter_bank(
        fs=config.FS,
        lowcut=config.LOWCUT,
        highcut=config.HIGHCUT,
        notch_freqs=[60],
        notch_q=30
    )
    filter_state = {}  # Stateful tracking


    # === Segmentation configuration ===
    window_size = config.CLASSIFY_WINDOW / 1000
    step_size = 1 / 16
    window_samples = int(window_size * config.FS)
    step_samples = int(step_size * config.FS)
    chunk_samples = int(window_size * config.FS)  # based on config.py parameters (256 for a 512Hz window)

    segments_all = []
    labels_all = []

    # === Precompute all trial windows ===
    trial_windows = []
    for start_marker, end_marker in EPOCHS_START_END.items():
        start_indices = np.where(marker_values == int(start_marker))[0]
        end_indices = np.where(marker_values == int(end_marker))[0]
        if len(start_indices) != len(end_indices):
            min_len = min(len(start_indices), len(end_indices))
            start_indices = start_indices[:min_len]
            end_indices = end_indices[:min_len]
        for s_idx, e_idx in zip(start_indices, end_indices):
            ts_start = marker_timestamps[s_idx]
            ts_end = marker_timestamps[e_idx]
            if ts_end - ts_start > 1.0:
                trial_windows.append((ts_start, ts_end, int(start_marker)))

    # === Sort windows and get min/max bounds ===
    trial_windows.sort()
    filter_warmup = 1.0 
    trial_bounds = [(start - 1.0, end) for (start, end, _) in trial_windows]
    valid_start = trial_bounds[0][0] - filter_warmup
    valid_end = trial_bounds[-1][1]

    # === Extract only relevant data segment ===
    global_start = np.searchsorted(eeg_timestamps, valid_start) 
    global_end = np.searchsorted(eeg_timestamps, valid_end)
    raw_global = eeg_data[:, global_start:global_end]
    rel_timestamps = eeg_timestamps[global_start:global_end]

    # === Stream through global segment with filter continuity ===
    filter_state = {}
    filtered_global = np.zeros_like(raw_global)

    for chunk_start in range(0, raw_global.shape[1], chunk_samples):
        chunk_end = min(chunk_start + chunk_samples, raw_global.shape[1])
        chunk = raw_global[:, chunk_start:chunk_end]
        filtered_chunk, filter_state = apply_streaming_filters(chunk, filter_bank, filter_state)
        filtered_global[:, chunk_start:chunk_end] = filtered_chunk

    # === For each trial, extract and label segments ===
    for trial_num, (ts_start, ts_end, label) in enumerate(trial_windows):
        rel_start = np.searchsorted(rel_timestamps, ts_start)
        rel_end = np.searchsorted(rel_timestamps, ts_end)
        baseline_end = np.searchsorted(rel_timestamps, ts_start)
        decision_start = np.searchsorted(rel_timestamps, ts_start + 1.0)

        if rel_end <= decision_start or baseline_end <= 0:
            continue

        baseline = filtered_global[:, :baseline_end].mean(axis=1, keepdims=True)
        trial_data = filtered_global[:, decision_start:rel_end] - baseline
        n_samples = trial_data.shape[1]

        if n_samples < window_samples:
            continue

        for i in range(0, n_samples - window_samples + 1, step_samples):
            segment = trial_data[:, i:i + window_samples]
            segments_all.append(segment)
            labels_all.append(label)


    return np.array(segments_all), np.array(labels_all)



def compute_processed_covariances(segments, labels):
    """
    Computes regularized and optionally whitened covariance matrices from EEG segments.

    Args:
        segments (np.ndarray): EEG data segments, shape (n_trials, n_channels, n_timepoints).
        labels (np.ndarray): Labels corresponding to each segment.
        fs (float): Sampling frequency, used for future flexibility (not required now).

    Returns:
        np.ndarray: Processed covariance matrices.
    """
    print("Computing raw covariance matrices...")
    cov_matrices = np.array([
        (seg @ seg.T) / np.trace(seg @ seg.T) for seg in segments
    ])

    print(f"Covariance shape: {cov_matrices.shape}")
    print("Label distribution:", dict(zip(*np.unique(labels, return_counts=True))))

    if config.LEDOITWOLF:
        print("Applying Ledoit-Wolf shrinkage...")
        cov_matrices = np.array([
            LedoitWolf().fit(cov).covariance_ for cov in cov_matrices
        ])
    else:
        print(f"Applying shrinkage (λ={config.SHRINKAGE_PARAM})...")
        shrinker = Shrinkage(shrinkage=config.SHRINKAGE_PARAM)
        cov_matrices = shrinker.fit_transform(cov_matrices)

    if config.RECENTERING:
        print("Applying Riemannian whitening...")
        whitener = Whitening(metric="riemann")
        cov_matrices = whitener.fit_transform(cov_matrices)

    print(f"Processed covariance matrices shape: {cov_matrices.shape}")
    return cov_matrices



# ---------- plotting helpers -------------------------------------------------
def _plot_scores_hist_with_thresholds(scores, y_bin, t_low, t_high, bins=30):
    s0 = scores[y_bin == 0]
    s1 = scores[y_bin == 1]
    plt.figure(figsize=(9, 5))
    plt.hist(s0, bins=bins, alpha=0.6, density=True, label="True REST")
    plt.hist(s1, bins=bins, alpha=0.6, density=True, label="True MI")
    plt.axvline(t_low,  color="black", ls="--", lw=1.5, label=f"t_low={t_low:.3f}")
    plt.axvline(t_high, color="black", ls="--", lw=1.5, label=f"t_high={t_high:.3f}")
    yl = plt.ylim()
    plt.fill_betweenx(yl, t_low, t_high, color="gray", alpha=0.12, label="Ambiguous")
    plt.xlabel("Score (P[MI])"); plt.ylabel("Density")
    plt.title("Score Distributions with Dual Thresholds")
    plt.legend(); plt.grid(True, ls=":", alpha=0.6); plt.tight_layout(); plt.show()


def _plot_risk_coverage(scores, y_bin, center, widths, c_fp, c_fn, c_reject):
    cov, acc, cost = [], [], []
    for w in widths:
        tl, th = np.clip(center - w / 2, 0, 1), np.clip(center + w / 2, 0, 1)
        pred = np.full_like(y_bin, -1)
        pred[scores >= th] = 1
        pred[scores <= tl] = 0
        U  = (pred == -1).sum()
        TP = ((pred == 1) & (y_bin == 1)).sum()
        TN = ((pred == 0) & (y_bin == 0)).sum()
        FP = ((pred == 1) & (y_bin == 0)).sum()
        FN = ((pred == 0) & (y_bin == 1)).sum()
        decided = TP + TN + FP + FN
        cov.append(decided / len(y_bin))
        acc.append((TP + TN) / decided if decided else np.nan)
        cost.append(c_fp * FP + c_fn * FN + c_reject * U)
    cov, acc, cost = np.array(cov), np.array(acc), np.array(cost)

    plt.figure(figsize=(9, 4))
    plt.plot(cov * 100, acc, marker="o")
    plt.xlabel("Coverage (%)"); plt.ylabel("Decided-only Accuracy")
    plt.title("Risk–Coverage: Accuracy vs Coverage")
    plt.grid(True, ls=":", alpha=0.6); plt.tight_layout(); plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(cov * 100, cost, marker="o")
    plt.xlabel("Coverage (%)"); plt.ylabel(f"Cost (c_fp={c_fp}, c_fn={c_fn}, c_reject={c_reject})")
    plt.title("Risk–Coverage: Cost vs Coverage")
    plt.grid(True, ls=":", alpha=0.6); plt.tight_layout(); plt.show()


def _plot_roc_with_point(scores, y_bin, t_high):
    fpr, tpr, thr = roc_curve(y_bin, scores)
    auc = roc_auc_score(y_bin, scores)
    idx = np.argmin(np.abs(thr - t_high))
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], ls="--", alpha=0.7, label="Chance")
    plt.scatter(fpr[idx], tpr[idx], s=60, label=f"operating @ t_high={t_high:.3f}")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC (Positive = MI)"); plt.legend()
    plt.grid(True, ls=":", alpha=0.6); plt.tight_layout(); plt.show()


def _plot_confusion_with_reject(true_all, pred_all, rest_label, mi_label):
    """
    Heatmap-like view for confusion-with-reject.
    Rows: true class (REST, MI)
    Cols: predicted (REST, AMB, MI)
    pred_all uses -1 for ambiguous.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Ensure ndarray
    true_all = np.asarray(true_all)
    pred_all = np.asarray(pred_all)

    REST, MI, AMB = rest_label, mi_label, -1

    mat = np.array([
        np.sum((true_all == REST) & (pred_all == REST)),
        np.sum((true_all == REST) & (pred_all == AMB)),
        np.sum((true_all == REST) & (pred_all == MI)),
        np.sum((true_all == MI)   & (pred_all == REST)),
        np.sum((true_all == MI)   & (pred_all == AMB)),
        np.sum((true_all == MI)   & (pred_all == MI)),
    ], dtype=int).reshape(2, 3)

    plt.figure(figsize=(6.5, 4.8))
    im = plt.imshow(mat, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([0, 1, 2], ["REST", "AMB", "MI"])
    plt.yticks([0, 1], ["REST (true)", "MI (true)"])

    # annotate counts
    for i in range(2):
        for j in range(3):
            plt.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=11)

    plt.title("Confusion with Reject (counts)")
    plt.tight_layout()
    plt.show()


def pick_dual_thresholds_target_ambiguity(
    y_true_bin, pos_scores,
    target_ambig=0.20,
    c_fp=1.0, c_fn=1.0,               # tie-break costs among feasible pairs
    n_grid=401, min_gap=0.0,
    # --- optional performance constraints on the decided subset ---
    tpr_min=None,                     # e.g., 0.85  (MI recall ≥ 85%)
    fpr_max=None,                     # e.g., 0.05  (false MI rate ≤ 5%)
    ppv_min=None,                     # e.g., 0.80  (MI precision ≥ 80%)
    npv_min=None,                     # e.g., 0.90  (REST precision ≥ 90%)
    # --- optional "make it look sane" constraint ---
    require_center_around_half=False  # if True, force t_low <= 0.5 <= t_high
):
    """
    Choose (t_low, t_high) on P(MI) so that ambiguous fraction U/N ≲ target_ambig,
    constraints on decided-only metrics are satisfied, and among feasible pairs
    we minimize (c_fp*FP + c_fn*FN). If infeasible, return the nearest-violation pair.
    """
    y = np.asarray(y_true_bin, dtype=int).ravel()
    s = np.asarray(pos_scores, dtype=float).ravel()
    N = len(y)
    grid = np.linspace(0.0, 1.0, n_grid)

    best_feas = None    # (cost, Urate, tpr, fpr, ppv, npv, tl, th)
    best_violation = None  # (total_slack, Urate_gap, cost, tl, th, tpr, fpr, ppv, npv)

    for tl in grid:
        for th in grid:
            if th < max(tl, tl + min_gap):
                continue
            if require_center_around_half and not (tl <= 0.5 <= th):
                continue

            # classify with reject
            pred = np.full_like(y, -1)
            pred[s >= th] = 1
            pred[s <= tl] = 0

            U = np.sum(pred == -1)
            Urate = U / N

            # decided subset
            decided = (pred != -1)
            if not decided.any():
                continue
            yd, pd = y[decided], pred[decided]

            TP = np.sum((pd == 1) & (yd == 1))
            TN = np.sum((pd == 0) & (yd == 0))
            FP = np.sum((pd == 1) & (yd == 0))
            FN = np.sum((pd == 0) & (yd == 1))

            tpr = TP / (TP + FN) if (TP + FN) else 0.0
            fpr = FP / (FP + TN) if (FP + TN) else 0.0
            ppv = TP / (TP + FP) if (TP + FP) else 0.0
            npv = TN / (TN + FN) if (TN + FN) else 0.0

            # constraint slacks (0 if satisfied, >0 if violated)
            slack_tpr = max(0.0, (tpr_min - tpr)) if tpr_min is not None else 0.0
            slack_fpr = max(0.0, (fpr - fpr_max)) if fpr_max is not None else 0.0
            slack_ppv = max(0.0, (ppv_min - ppv)) if ppv_min is not None else 0.0
            slack_npv = max(0.0, (npv_min - npv)) if npv_min is not None else 0.0
            total_slack = slack_tpr + slack_fpr + slack_ppv + slack_npv

            cost = c_fp * FP + c_fn * FN

            # Feasible if ambiguity and all constraints satisfied
            if (Urate <= target_ambig) and (total_slack == 0.0):
                if (best_feas is None or
                    cost < best_feas[0] or
                    (cost == best_feas[0] and Urate < best_feas[1])):
                    best_feas = (cost, Urate, tpr, fpr, ppv, npv, tl, th)
            else:
                # Keep closest infeasible pair (smallest total_slack),
                # tie-break by |Urate - target_ambig|, then by cost
                Urate_gap = abs(Urate - target_ambig)
                cand = (total_slack, Urate_gap, cost, tl, th, tpr, fpr, ppv, npv)
                if (best_violation is None or
                    cand < best_violation):
                    best_violation = cand

    if best_feas is not None:
        cost, Urate, tpr, fpr, ppv, npv, tl, th = best_feas
        info = dict(feasible=True, Urate=Urate, tpr=tpr, fpr=fpr, ppv=ppv, npv=npv, cost=cost)
        return tl, th, info
    else:
        # return the least-violating pair and tell the caller what it achieved
        total_slack, Urate_gap, cost, tl, th, tpr, fpr, ppv, npv = best_violation
        info = dict(feasible=False, Urate=None, tpr=tpr, fpr=fpr, ppv=ppv, npv=npv,
                    cost=cost, total_slack=total_slack)
        return tl, th, info


# ---------- threshold learner (unchanged) ------------------------------------
def _dual_thresholds(y_true_bin, scores, c_fp=1.0, c_fn=1.0, c_reject=0.3,
                     n_grid=201, min_gap=0.0):
    grid = np.linspace(0, 1, n_grid)
    best = (np.inf, 0.5, 0.5)  # cost, tl, th
    for tl in grid:
        for th in grid:
            if th < max(tl, tl + min_gap):
                continue
            pred = np.full_like(y_true_bin, -1)
            pred[scores >= th] = 1
            pred[scores <= tl] = 0
            U  = (pred == -1).sum()
            FP = ((pred == 1) & (y_true_bin == 0)).sum()
            FN = ((pred == 0) & (y_true_bin == 1)).sum()
            cost = c_fp * FP + c_fn * FN + c_reject * U
            if cost < best[0]:
                best = (cost, tl, th)
    return best[1], best[2]


# ---------- main training function -------------------------------------------
def train_riemannian_model(
    cov_matrices,
    labels,
    n_splits=8,
    use_dual_thresholds=True,
    c_fp=1.0, c_fn=1.0, c_reject=0.3,
    n_grid=201,
    min_gap=0.0,
    target_ambig=0.3,          # keep ~20% ambiguity by default
    # --- NEW constraint knobs (decided-only metrics) ---
    tpr_min=None,              # e.g., 0.85 for MI recall >= 85%
    fpr_max=None,              # e.g., 0.10 for false MI rate <= 10%
    ppv_min=None,              # e.g., 0.80 for MI precision >= 80%
    npv_min=None,              # e.g., 0.90 for REST precision >= 90%
    require_center_around_half=False,  # force t_low <= 0.5 <= t_high (optional aesthetics)
):
    """
    Trains MDM with k-fold CV, learns dual thresholds to mirror online driver,
    prints aggregated report, and generates diagnostic plots.
    """

    print("\n🚀 Starting K-Fold Cross Validation with Riemannian MDM...\n")

    # Informational note: when target_ambig is set, c_reject is ignored during selection
    if target_ambig is not None and (c_reject not in (None, 0.0)):
        print("[Note] target_ambig is set; c_reject is ignored for threshold SELECTION "
              "(still used only in the reported overall cost below).")

    classes = np.sort(np.unique(labels))
    if len(classes) != 2:
        raise ValueError("Function expects binary classes.")
    rest_label, mi_label = classes[0], classes[1]

    kf  = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mdm = MDM()

    # Aggregation containers
    acc_argmax    = []
    t_lows, t_highs = [], []
    all_true, all_pred, all_scores, all_true_bin = [], [], [], []
    posterior_probs = {lbl: [] for lbl in classes}

    for fold_idx, (tr, te) in enumerate(kf.split(cov_matrices), 1):
        X_tr, X_te = cov_matrices[tr], cov_matrices[te]
        y_tr, y_te = labels[tr], labels[te]

        mdm.fit(X_tr, y_tr)
        prob_tr = mdm.predict_proba(X_tr)
        prob_te = mdm.predict_proba(X_te)
        pos_idx = np.where(mdm.classes_ == mi_label)[0][0]
        scr_tr  = prob_tr[:, pos_idx]
        scr_te  = prob_te[:, pos_idx]

        # baseline argmax accuracy
        y_pred_argmax = mdm.predict(X_te)
        acc = accuracy_score(y_te, y_pred_argmax)
        acc_argmax.append(acc)
        print(f"✅ Fold {fold_idx} Argmax Accuracy: {acc:.4f}")

        # dual thresholds on TRAIN split
        if use_dual_thresholds:
            y_tr_bin = (y_tr == mi_label).astype(int)
            if target_ambig is not None:
                # --- NEW: pass constraint knobs into the ambiguity-target picker
                tl, th, diag = pick_dual_thresholds_target_ambiguity(
                    y_true_bin=y_tr_bin,
                    pos_scores=scr_tr,
                    target_ambig=target_ambig,
                    c_fp=c_fp, c_fn=c_fn,
                    n_grid=n_grid, min_gap=min_gap,
                    tpr_min=tpr_min, fpr_max=fpr_max,
                    ppv_min=ppv_min, npv_min=npv_min,
                    require_center_around_half=require_center_around_half
                )
                print(f"   ↳ thresholds@ambig={target_ambig:.2f}: "
                      f"t_low={tl:.3f}, t_high={th:.3f}, feasible={diag.get('feasible', False)}, "
                      f"train_TPR={diag.get('tpr', float('nan')):.3f}, "
                      f"train_FPR={diag.get('fpr', float('nan')):.3f}, "
                      f"train_PPV={diag.get('ppv', float('nan')):.3f}, "
                      f"train_NPV={diag.get('npv', float('nan')):.3f}")
            else:
                # cost-based picker (uses c_reject)
                tl, th = _dual_thresholds(y_tr_bin, scr_tr, c_fp, c_fn, c_reject,
                                          n_grid, min_gap)
        else:
            tl = th = 0.5

        t_lows.append(tl); t_highs.append(th)

        # apply thresholds to TEST
        pred = np.full_like(y_te, -1)
        pred[scr_te >= th] = mi_label
        pred[scr_te <= tl] = rest_label

        # aggregate
        all_true.extend(y_te); all_pred.extend(pred)
        all_scores.extend(scr_te)
        all_true_bin.extend((y_te == mi_label).astype(int))

        # posterior for histogram
        for idx, true_lbl in enumerate(y_te):
            posterior_probs[true_lbl].append(
                prob_te[idx, np.where(mdm.classes_ == true_lbl)[0][0]]
            )

    # ---------- aggregated report ----------
    all_true     = np.asarray(all_true)
    all_pred     = np.asarray(all_pred)
    all_scores   = np.asarray(all_scores)
    all_true_bin = np.asarray(all_true_bin)

    decided = all_pred != -1
    cm_decided = confusion_matrix(all_true[decided], all_pred[decided],
                                  labels=[rest_label, mi_label])
    TN, FP, FN, TP = cm_decided.ravel()
    U  = (all_pred == -1).sum()
    coverage = decided.mean()
    decided_acc = (TP + TN) / (TP + TN + FP + FN) if decided.any() else np.nan
    cost = c_fp * FP + c_fn * FN + c_reject * U

    # (Optional) decided-only diagnostic metrics
    TPR = TP / (TP + FN) if (TP + FN) else np.nan   # MI recall
    FPR = FP / (FP + TN) if (FP + TN) else np.nan   # false MI rate
    PPV = TP / (TP + FP) if (TP + FP) else np.nan   # MI precision
    NPV = TN / (TN + FN) if (TN + FN) else np.nan   # REST precision

    tl_star, th_star = float(np.median(t_lows)), float(np.median(t_highs))

    print("\n====== Aggregated Report ======")
    print(f"Argmax Accuracy (mean): {np.mean(acc_argmax):.4f}")
    print(f"Learned thresholds (medians): t_low*={tl_star:.3f}, t_high*={th_star:.3f}")
    print(f"Coverage (decided %): {coverage*100:.2f}% (Ambiguity {(1.0-coverage)*100:.2f}%)")
    print(f"Decided-only Accuracy: {decided_acc:.4f}")
    print(f"Decided-only: TPR(MI)={TPR:.3f}, FPR(MI)={FPR:.3f}, PPV(MI)={PPV:.3f}, NPV(REST)={NPV:.3f}")
    print(f"Overall Cost (c_fp={c_fp}, c_fn={c_fn}, c_reject={c_reject}): {cost:.1f}")

    print("\nConfusion (decided-only; rows=true [REST, MI], cols=pred [REST, MI]):")
    print(cm_decided)

    print("\nAmbiguous counts by true class:")
    print(f"U_rest={(all_pred == -1)[all_true == rest_label].sum()}, "
          f"U_mi={(all_pred == -1)[all_true == mi_label].sum()}, "
          f"Total U={U}")

    # Mode B thresholds for your online config
    THRESHOLD_REST = 1.0 - tl_star  # compare to P(REST)
    THRESHOLD_MI   = th_star        # compare to P(MI)
    print("\nMode B thresholds for config:")
    print(f"  THRESHOLD_REST (use with P(REST) ≥ ...): {THRESHOLD_REST:.3f}")
    print(f"  THRESHOLD_MI   (use with P(MI)   ≥ ...): {THRESHOLD_MI:.3f}")

    # ---------- plots ----------
    _plot_scores_hist_with_thresholds(all_scores, all_true_bin, tl_star, th_star)

    center = (tl_star + th_star) / 2.0
    widths = np.linspace(0.0, 0.9, 35)
    _plot_risk_coverage(all_scores, all_true_bin, center, widths, c_fp, c_fn, c_reject)

    _plot_roc_with_point(all_scores, all_true_bin, th_star)

    _plot_confusion_with_reject(all_true, all_pred, rest_label, mi_label)

    # posterior histograms (your original helper)
    for lbl in posterior_probs:
        posterior_probs[lbl] = np.array(posterior_probs[lbl])
    plot_posterior_probabilities(posterior_probs)

    # ---------- fit final model (unchanged) ----------
    mdm.fit(cov_matrices, labels)
    return mdm




def main():
    """
    Main function to generate a Riemannian-based EEG decoder.
    """
    mne.set_log_level("WARNING")

    print("Loading XDF data...")
    eeg_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "training_data")
    print(f"Script is looking for XDF files in: {eeg_dir}")

    xdf_files = [
        os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir)
        if f.endswith(".xdf") and "OBS" not in f
    ]

    if not xdf_files:
        raise FileNotFoundError(f"No XDF files found in: {eeg_dir}")
    print(f"Found XDF files: {xdf_files}")

    all_cov_matrices = []
    all_labels = []

    for xdf_path in xdf_files:
        print(f"\n📂 Processing file: {xdf_path}")
        eeg_stream, marker_stream = load_xdf(xdf_path)

        segments, labels = segment_and_label_one_run(eeg_stream, marker_stream)

        # Print summary
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_summary = ", ".join([f"{int(lbl)}: {cnt}" for lbl, cnt in zip(unique_labels, counts)])
        print("\n📊 Segmentation Summary:")
        print(f"🔹 Total segments: {len(segments)}")
        print(f"🔹 Segment shape: {segments.shape} (n_segments, n_channels, n_timepoints)")
        print(f"🔹 Class distribution: {label_summary}")


    
        # === HARD REJECTION BASED ON PEAK AMPLITUDE ===
        REJECTION_THRESHOLD_UV = 30  # μV

        # Compute max abs amplitude per segment
        max_vals = np.max(np.abs(segments), axis=(1, 2))  # shape: (n_segments,)
        keep_mask = max_vals <= REJECTION_THRESHOLD_UV

        # Apply rejection
        segments = segments[keep_mask]
        labels = labels[keep_mask]

        print(f"Retained {len(segments)} segments after rejecting {np.sum(~keep_mask)} high-amplitude artifacts.")

        
        cov_matrices = compute_processed_covariances(segments, labels)
        
        #print(mean_riemann(cov_matrices))
        all_cov_matrices.append(cov_matrices)
        all_labels.append(labels)

    all_labels = np.concatenate(all_labels)
    all_cov_matrices = np.concatenate(all_cov_matrices)
    print("Training Riemannian Classifier...")
    Reimans_model = train_riemannian_model(all_cov_matrices, all_labels)

    #  Save the trained model
    # Define model save path (subject-level, not session-specific)
    subject_model_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "models")
    os.makedirs(subject_model_dir, exist_ok=True)

    subject_model_path = os.path.join(subject_model_dir, f"sub-{config.TRAINING_SUBJECT}_model.pkl")

    
    # Save the trained model
    with open(subject_model_path, 'wb') as f:
        pickle.dump(Reimans_model, f)

    print(f"✅ Trained model saved at: {subject_model_path}")
    #np.save(Training_mean_path, training_mean)
    #np.save(Training_std_path, training_std)
    #print(" Saved precomputed training mean and std for real-time use.")


if __name__ == "__main__":
    main()
