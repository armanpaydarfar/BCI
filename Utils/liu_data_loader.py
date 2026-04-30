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

from Utils.preprocessing import car_rereference as _car_rereference

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
# Must be kept in sync with config.ERRP_CHANNEL_NAMES.
HARMONY_ERRP_CHANNELS: list[str] = [
    "F3", "Fz", "F4", "FC1", "FC2",
    "C3", "Cz", "C4",
    "CP1", "CP2",
    "Pz", "POz",
    "O1", "O2",
]

# Sampling rate (confirmed from params.fsamp in the mat file).
LIU_FS: int = 512

# Epoch time axis: 768 samples at 512 Hz, from sample −255 to +512.
LIU_TIME_S: np.ndarray = np.arange(-255, 513) / LIU_FS


def load_liu_epochs(
    mat_path: str,
    subjects: list[int] | None = None,
    channel_names: list[str] | None = None,
    car: bool = True,
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
    car : bool
        If True (default), apply Common Average Reference across the retained
        channels after selection.  Matches the online Harmony path, which
        CAR-rereferences its 14-channel ErrP subset before spatial filtering.

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

        # First pass: read per-subject epoch counts so we can preallocate a
        # single contiguous output array.  Accumulating into a list of
        # per-subject arrays and concatenating at the end doubles peak RAM,
        # which OOMs a 16 GB box at 14 channels × 16 subjects.
        sizes = [int(f[struct["label"][s - 1, 0]].size) for s in subjects]
        total = int(sum(sizes))
        n_ch = len(ch_idx)
        n_samples = f[struct["rotation_data"][subjects[0] - 1, 0]].shape[-1]

        X      = np.empty((total, n_ch, n_samples), dtype=np.float64)
        y      = np.empty(total, dtype=int)
        mag    = np.empty(total, dtype=np.float64)
        sub_id = np.empty(total, dtype=int)

        offset = 0
        for s, n in zip(subjects, sizes):
            row = s - 1
            eeg = f[struct["rotation_data"][row, 0]][()]   # (epochs, 32, 768)
            X[offset:offset + n] = eeg[:, ch_idx, :]
            del eeg
            y[offset:offset + n] = f[struct["label"][row, 0]][()].flatten().astype(int)
            mag[offset:offset + n] = f[struct["magnitude"][row, 0]][()].flatten()
            sub_id[offset:offset + n] = s
            offset += n

    # Common Average Reference across the retained channels.  Applied after
    # channel selection so the reference matches the online footprint rather
    # than the full 32-channel cap.
    if car:
        X = _car_rereference(X)

    # Rebuild time axis for the returned channel count.
    time_s = np.arange(-255, 513) / LIU_FS

    return X, y, mag, sub_id, time_s, subjects
