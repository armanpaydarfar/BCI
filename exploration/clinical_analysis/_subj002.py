"""Single source of truth for CLIN_SUBJ_002's protocol divergences.

Background. SUBJ_002 was recorded under the older runtime configuration —
the study expanded after she finished her sessions, so the post-002
subjects (CLIN_SUBJ_003..008) use different defaults across channel
count, shrinkage, decoder backend, classify window, and integrator
alpha. Five clinical-analysis scripts had been independently
maintaining their own copies of these divergence flags
(`Analyze_clinical_gr_ablation.py`, `Analyze_eds_topoplot_CLIN.py`,
`Analyze_clinical_bar_dynamics_longitudinal.py`,
`Analyze_clinical_decoder_longitudinal.py`,
`Analyze_clinical_confusion_matrices.py`), risking drift across the
pipeline. This module is the unified declaration. Scripts import from
here instead of hardcoding.

Notes-confirmed protocol (from `sub-CLIN_SUBJ_002/notes_06_0{4,5}`):
  - S001ONLINE: LEFT-arm MI vs rest, 1 online run (operator's decoder,
    transfer learning). MI/REST marker semantics flip vs the canonical
    cohort, so S001 is EXCLUDED from cohort analyses.
  - S002ONLINE: RIGHT-arm MI vs rest, 6 runs (operator's decoder,
    transfer learning). INCLUDED.

S003ONLINE and S004ONLINE (both recorded 2025-06-12) lack written notes
in `sub-CLIN_SUBJ_002/notes_*`. Existing scripts have treated them as
right-arm transfer-learning sessions based on file-structure inference
alone. They are INCLUDED by default per the historical convention but
flagged here for future verification.

To restrict cohort analyses to the two notes-confirmed sessions, set
`SUBJ002_INCLUDE_S003_S004 = False` below. All downstream scripts will
honor the choice via the helper functions in this module.
"""

# Whether to include the two unverified-by-notes right-arm sessions
# (S003/S004) alongside the notes-confirmed S002. Default True to
# preserve the historical gr_ablation convention until the user verifies
# the S003/S004 protocol; flip to False to use only S002.
SUBJ002_INCLUDE_S003_S004 = True

# All right-arm online sessions per the historical gr_ablation
# convention. S001 is always excluded (left-arm).
_SUBJ002_RIGHT_ARM_ALL = ("S002ONLINE", "S003ONLINE", "S004ONLINE")
_SUBJ002_NOTES_CONFIRMED = ("S002ONLINE",)

# Motor channel set she was recorded with (older
# `select_motor_channels(keep_prefixes=("CP","P","C"))` convention,
# verified at git c5d2886). From the 30-channel cap after drop_fp, the
# surviving channels starting with C/CP/P are these 13:
SUBJ002_MOTOR_CHANNELS_13 = (
    "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "Pz", "P4", "P8", "POz",
)

# Shrinkage method. SUBJ_002 used LedoitWolf with adaptive λ fit on the
# raw window (`Utils/runtime_common.py:283`) at runtime; post-002
# subjects use a fixed Shrinkage(λ=0.02) on the trace-normalised
# covariance. Documented divergence — both are honest given the
# respective runtime configs.
SUBJ002_USE_LEDOITWOLF = True

# Older bar-dynamics integration constants. Post-002 subjects use
# CLASSIFY_WINDOW=1000 ms and INTEGRATOR_ALPHA=0.98.
SUBJ002_CLASSIFY_WINDOW_MS = 500
SUBJ002_INTEGRATOR_ALPHA = 0.95

# Lateralization expectation for the rubric (`evaluate_erd_quality.py`
# _LATERALIZATION_EXPECTED). Different from post-002 subjects because
# her S001 left-arm session contaminates any single-side expectation.
SUBJ002_LATERALIZATION = "bilateral"


def subj002_valid_sessions() -> list[str]:
    """Right-arm online sessions for CLIN_SUBJ_002.

    Returns the notes-confirmed set if `SUBJ002_INCLUDE_S003_S004` is
    False, else the full historical right-arm set (S002/S003/S004).
    """
    if SUBJ002_INCLUDE_S003_S004:
        return list(_SUBJ002_RIGHT_ARM_ALL)
    return list(_SUBJ002_NOTES_CONFIRMED)


def is_subj002(subject: str) -> bool:
    return subject == "CLIN_SUBJ_002"
