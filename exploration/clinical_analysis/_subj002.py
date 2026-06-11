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

Per-session protocol (verified 2026-06-04 from `config_snapshot.json`
`ARM_SIDE`, the event-log `training data:` line, and `notes_06_0{4,5}`):

  - S001ONLINE (2025-06-04, 1 run): LEFT-arm MI vs rest. Calibration
    failed (software-trigger issue), so the operator's overnight decoder
    was used (transfer learning). MI/REST marker semantics flip vs the
    right-arm cohort. EXCLUDED — left arm.
  - S002ONLINE (2025-06-05, 6 runs): RIGHT-arm, operator's decoder
    ("MY decoder", notes_06_05) — expert transfer learning. INCLUDED.
  - S003ONLINE (2025-06-12, 2 runs): RIGHT-arm, but the decoder was
    trained on her OWN `S002ONLINE` data (event-log `training data:`
    line) — within-subject transfer, NOT expert→patient transfer.
    EXCLUDED — different paradigm from the rest of the cohort (decision
    2026-06-04). It remains a valid right-arm MI recording, just not an
    expert-TL session.
  - S004ONLINE (2025-06-12, 5 runs): RIGHT-arm, decoder trained on
    the shared operator expert pool used cohort-wide — expert
    transfer learning. INCLUDED.

S001 (left arm) is excluded everywhere. S003's treatment then SPLITS by
analysis family (see the two constant blocks below):
  - DECODER family: S003 excluded; sessions S002/S004 -> idx 1/2.
  - FEATURE family: S003 included and POOLED with S004 (same day) into one
    longitudinal timepoint; S002 -> 1, {S003, S004} -> 2.
Downstream scripts pick the matching `subj002_{decoder,feature}_sessions`
and `subj002_{decoder,feature}_idx` helpers.
"""

# Two session policies, because S003's status depends on the analysis:
#
# DECODER family (gr_ablation, decoder-perf, bar-dynamics): the deployed
#   decoder's source matters, and S003 used a decoder trained on her OWN
#   S002 data (within-subject), not the expert/operator pool. So S003 is
#   EXCLUDED; the two expert-TL sessions S002/S004 are renumbered 1/2.
#
# FEATURE family (ERD/neuromod, erd_refined, FD): decoder-independent —
#   they measure her mu desync / covariance separability during right-arm
#   MI, which is valid in S003 regardless of decoder source. S003 and S004
#   are the same day (2025-06-12) and POOL into one longitudinal timepoint
#   (session 2); S002 is session 1. Hence both S003 and S004 map to idx 2,
#   and feature scripts pool trials per session_idx.
_SUBJ002_DECODER_SESSIONS = ("S002ONLINE", "S004ONLINE")
_SUBJ002_DECODER_IDX = {"S002ONLINE": 1, "S004ONLINE": 2}

_SUBJ002_FEATURE_SESSIONS = ("S002ONLINE", "S003ONLINE", "S004ONLINE")
_SUBJ002_FEATURE_IDX = {"S002ONLINE": 1, "S003ONLINE": 2, "S004ONLINE": 2}

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


def subj002_decoder_sessions() -> list[str]:
    """DECODER-family sessions for CLIN_SUBJ_002 — the two expert-TL
    right-arm sessions (S002, S004). S001 (left arm) and S003
    (within-subject transfer) are excluded. See the module docstring."""
    return list(_SUBJ002_DECODER_SESSIONS)


def subj002_decoder_idx(session: str) -> int:
    """DECODER-family longitudinal index (S002 -> 1, S004 -> 2). Raises
    KeyError for a session outside `subj002_decoder_sessions()`."""
    return _SUBJ002_DECODER_IDX[session]


def subj002_feature_sessions() -> list[str]:
    """FEATURE-family sessions for CLIN_SUBJ_002 — S002, S003, S004. S003
    is included (decoder-independent ERD/FD); it pools with S004 into one
    longitudinal timepoint via `subj002_feature_idx`."""
    return list(_SUBJ002_FEATURE_SESSIONS)


def subj002_feature_idx(session: str) -> int:
    """FEATURE-family longitudinal index (S002 -> 1, S003 -> 2, S004 -> 2).
    S003 and S004 share idx 2 so feature scripts pool their trials into one
    same-day session-2 estimate. Raises KeyError for a session outside
    `subj002_feature_sessions()`."""
    return _SUBJ002_FEATURE_IDX[session]


def is_subj002(subject: str) -> bool:
    return subject == "CLIN_SUBJ_002"
