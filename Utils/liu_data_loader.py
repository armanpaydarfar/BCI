"""
Utils/liu_data_loader.py — Data loader for the Liu et al. (2025) ErrP dataset.

Provides load_liu_epochs() for loading combinedEpochs_v2.mat or
eegEpochs_16subs_chanInterp_control.mat.  Both files are MATLAB v7.3
(HDF5) format and must be read with h5py, not scipy.io.loadmat.

Data notes:
  - Epochs are already theta-band filtered (1–10 Hz) by the authors.
  - Do not re-filter.
  - 32 EEG channels ordered as per Liu et al. Methods (ANT Neuro eego system).
  - 768 samples per epoch at 512 Hz → −0.498 s to +1.0 s relative to event onset.
  - label: 0 = correct (0° rotation), 1 = error (3/6/9/12° rotation).

Reference: Liu, Iwane et al. (2025). Brain-computer interface training fosters
perceptual skills to detect errors. bioRxiv. doi:10.1101/2025.04.26.650792
"""

from __future__ import annotations

import numpy as np
import h5py

# =============================================================================
# Channel layout (Liu et al. Methods, ANT Neuro eego system, 10-20 positions).
# Index 0 = rotation_data[:, 0, :] in the mat file.
# =============================================================================
LIU_CHANNEL_NAMES: list[str] = [
    "AF3", "AF4",
    "F3", "F1", "Fz", "F2", "F4",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C3", "C1", "Cz", "C2", "C4",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P3", "P1", "Pz", "P2", "P4",
    "PO3", "POz", "PO4",
    "O1", "O2",
]
assert len(LIU_CHANNEL_NAMES) == 32

_CH_IDX: dict[str, int] = {name: i for i, name in enumerate(LIU_CHANNEL_NAMES)}

# Harmony ErrP channel names (from config.ERRP_CHANNEL_NAMES).
# Hard-coded here so the loader can operate without importing config.
HARMONY_ERRP_CHANNELS: list[str] = ["F3", "Fz", "F4", "FC1", "FC2", "Cz"]

# Sampling rate (confirmed from params.fsamp in the mat file).
LIU_FS: int = 512

# Epoch time axis: 768 samples at 512 Hz, from sample −255 to +512.
LIU_TIME_S: np.ndarray = np.arange(-255, 513) / LIU_FS


def load_liu_epochs(
    mat_path: str,
    subjects: list[int] | None = None,
    channel_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Load epochs from a Liu et al. .mat file.

    Parameters
    ----------
    mat_path : str
        Path to combinedEpochs_v2.mat or eegEpochs_16subs_chanInterp_control.mat.
    subjects : list[int] | None
        1-based subject indices to load (e.g. [1, 2, 3]).  None = all subjects.
    channel_names : list[str] | None
        Channel names to retain (must be in LIU_CHANNEL_NAMES).  None = all 32.

    Returns
    -------
    X : ndarray, shape (n_epochs, n_channels, n_samples)
        EEG epochs (already filtered, µV).
    y : ndarray, shape (n_epochs,), int
        Labels: 0 = correct, 1 = error.
    mag : ndarray, shape (n_epochs,), float
        Rotation magnitude in degrees: 0, 3, 6, 9, or 12.
    sub_id : ndarray, shape (n_epochs,), int
        1-based subject index for each epoch.
    time_s : ndarray, shape (n_samples,)
        Time axis in seconds relative to event onset.
    loaded_subjects : list[int]
        1-based subject indices that were actually loaded.
    """
    with h5py.File(mat_path, "r") as f:
        if "combinedEpochs" in f:
            struct = f["combinedEpochs"]
        elif "eegEpochs" in f:
            struct = f["eegEpochs"]
        else:
            raise ValueError(
                f"No recognised struct in {mat_path}. Keys: {list(f.keys())}"
            )

        n_subjects_in_file = struct["label"].shape[0]
        if subjects is None:
            subjects = list(range(1, n_subjects_in_file + 1))

        for s in subjects:
            if s < 1 or s > n_subjects_in_file:
                raise ValueError(
                    f"Subject {s} out of range [1, {n_subjects_in_file}]."
                )

        if channel_names is None:
            ch_idx = list(range(32))
        else:
            unknown = [c for c in channel_names if c not in _CH_IDX]
            if unknown:
                raise ValueError(
                    f"Unknown channel(s): {unknown}. "
                    f"Valid names: {LIU_CHANNEL_NAMES}"
                )
            ch_idx = [_CH_IDX[c] for c in channel_names]

        Xs, ys, mags, sids = [], [], [], []
        for s in subjects:
            row = s - 1
            eeg = f[struct["rotation_data"][row, 0]][()]   # (epochs, 32, 768)
            lbl = f[struct["label"][row, 0]][()].flatten().astype(int)
            mgn = f[struct["magnitude"][row, 0]][()].flatten()

            Xs.append(eeg[:, ch_idx, :])
            ys.append(lbl)
            mags.append(mgn)
            sids.append(np.full(len(lbl), s, dtype=int))

    X      = np.concatenate(Xs,    axis=0).astype(np.float64)
    y      = np.concatenate(ys)
    mag    = np.concatenate(mags)
    sub_id = np.concatenate(sids)

    # Rebuild time axis for the returned channel count.
    time_s = np.arange(-255, 513) / LIU_FS

    return X, y, mag, sub_id, time_s, subjects
