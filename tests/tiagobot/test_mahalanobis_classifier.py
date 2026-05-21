"""
test_mahalanobis_classifier.py

Guards the Mahalanobis classifier path added 2026-05-20 (see
`Utils/tiagobot_gaze.py:load_calibration_samples` and
`classify_gaze_mahalanobis`). The Mahalanobis classifier reuses the
full per-sample data stored in the NPZ's `G` + `labels` arrays — which
the centroid classifier throws away — so axes the calibration was
noisy on get down-weighted per-letter, instead of via a global
`depth_weight_cm_inv` scalar.

Pins:
  - load_calibration_samples returns one model per letter with finite
    mean + invertible cov_inv (regularization protects against
    degenerate calibrations).
  - The 3D model is omitted for letters with too few valid-depth
    samples; the 2D model is still produced.
  - classify_gaze_mahalanobis picks the calibrated centroid when fed
    that letter's own mean (sanity).
  - Per-letter fallback: when the runtime sample lacks depth, the
    classifier silently switches to 2D for every letter (doesn't error,
    doesn't return None).
  - When one letter's 3D model is None but another's exists, only the
    no-3D letter uses 2D; the others retain 3D.
  - NPZs lacking the `G` array (centroid-only schema) raise ValueError —
    fail-fast per CLAUDE.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from Utils.tiagobot_gaze import (
    LETTERS,
    classify_gaze_cloud_vote,
    classify_gaze_mahalanobis,
    load_calibration_samples,
    per_letter_distance_breakdown,
)


def _write_full_calibration_npz(
    path,
    samples_per_letter: int = 30,
    *,
    depth_for_letters=None,
    seed: int = 0,
) -> None:
    """Helper: synthesize a calibration NPZ with N samples per letter,
    each cluster tight around its nominal grid position. Optional
    per-letter depth (cm); letters omitted from `depth_for_letters` get
    NaN depth across all their samples.

    Layout: A..I at {0.25, 0.50, 0.75}^2, jittered with sigma=0.005.
    Depth jitter sigma=2 cm when set.
    """
    rng = np.random.default_rng(seed)
    nominal = {}
    for i, ch in enumerate(LETTERS):
        col = i % 3
        row = i // 3
        nominal[ch] = (0.25 + 0.25 * col, 0.25 + 0.25 * row)
    if depth_for_letters is None:
        depth_for_letters = {ch: 60.0 + 4 * i for i, ch in enumerate(LETTERS)}

    rows = []
    labels = []
    for ch in LETTERS:
        cx, cy = nominal[ch]
        for _ in range(samples_per_letter):
            x = cx + rng.normal(0, 0.005)
            y = cy + rng.normal(0, 0.005)
            if ch in depth_for_letters:
                d = depth_for_letters[ch] + rng.normal(0, 2.0)
            else:
                d = float("nan")
            rows.append([x, y, 1.0, d])
            labels.append(ch)
    G = np.array(rows, dtype=np.float32)
    labels_arr = np.array(labels, dtype="S1")
    centroids = np.zeros((9, 3), dtype=np.float32)
    for i, ch in enumerate(LETTERS):
        mask = labels_arr == ch.encode("ascii")
        centroids[i, 0] = np.median(G[mask, 0])
        centroids[i, 1] = np.median(G[mask, 1])
        ds = G[mask, 3]
        valid = np.isfinite(ds)
        centroids[i, 2] = float(np.median(ds[valid])) if valid.any() else np.nan
    letters_arr = np.array(list(LETTERS), dtype="S1")
    np.savez_compressed(
        path,
        T=np.zeros(len(rows), dtype=np.float64),
        G=G,
        labels=labels_arr,
        centroids=centroids,
        letters=letters_arr,
        meta=dict(version=1, source="test_mahalanobis_classifier"),
    )


def test_loads_one_model_per_letter(tmp_path):
    """Healthy synthetic calibration -> 9 letter models, all with both
    3D and 2D cov_inv."""
    p = tmp_path / "cal.npz"
    _write_full_calibration_npz(p)
    models = load_calibration_samples(p)
    assert set(models.keys()) == set(LETTERS)
    for ch, m in models.items():
        assert m["cov_inv_2d"].shape == (2, 2)
        assert m["cov_inv_3d"].shape == (3, 3)
        # Means are finite for all axes (depth was provided for every letter).
        assert np.all(np.isfinite(m["mean"]))
        assert m["n_samples"] == 30
        assert m["n_samples_with_depth"] == 30


def test_letters_with_no_valid_depth_get_2d_only(tmp_path):
    """If a letter's calibration captured no finite depth samples, the
    loader emits only the 2D model (cov_inv_3d=None) but keeps the
    letter eligible at classify time."""
    p = tmp_path / "cal_no_depth_E.npz"
    # E gets no depth; all others do.
    depth_map = {ch: 60.0 + 4 * i for i, ch in enumerate(LETTERS) if ch != "E"}
    _write_full_calibration_npz(p, depth_for_letters=depth_map)
    models = load_calibration_samples(p)
    assert "E" in models
    assert models["E"]["cov_inv_3d"] is None
    assert models["E"]["cov_inv_2d"] is not None
    # Mean depth is NaN, mean x/y are finite.
    assert np.isfinite(models["E"]["mean"][0])
    assert np.isfinite(models["E"]["mean"][1])
    assert not np.isfinite(models["E"]["mean"][2])


def test_letters_with_few_samples_are_omitted(tmp_path):
    """A letter with fewer than the minimum sample count is dropped
    entirely. We hand-craft a tiny NPZ to verify."""
    p = tmp_path / "cal_tiny.npz"
    rng = np.random.default_rng(0)
    rows = []
    labels = []
    # Only A gets the minimum; B gets 2 (below threshold).
    for ch, n in [("A", 5), ("B", 2)]:
        for _ in range(n):
            rows.append([0.5 + rng.normal(0, 0.01), 0.5, 1.0, 60.0])
            labels.append(ch)
    np.savez_compressed(
        p,
        T=np.zeros(len(rows)),
        G=np.array(rows, dtype=np.float32),
        labels=np.array(labels, dtype="S1"),
        centroids=np.full((9, 3), np.nan, dtype=np.float32),
        letters=np.array(list(LETTERS), dtype="S1"),
        meta=dict(version=1),
    )
    models = load_calibration_samples(p)
    assert "A" in models
    assert "B" not in models  # below _MAHAL_MIN_SAMPLES


def test_classify_returns_letter_at_own_mean(tmp_path):
    """Feed each letter's mean back into the classifier -> the
    classifier returns that letter (Mahalanobis distance = 0)."""
    p = tmp_path / "cal.npz"
    _write_full_calibration_npz(p)
    models = load_calibration_samples(p)
    for ch, m in models.items():
        mx, my, md = m["mean"]
        got = classify_gaze_mahalanobis(
            mx, my, models, gaze_depth_cm=md
        )
        assert got == ch, f"expected {ch} at its own mean, got {got}"


def test_classify_2d_fallback_when_runtime_depth_missing(tmp_path):
    """Pass gaze_depth_cm=None; classifier uses 2D Mahalanobis for
    every letter and still returns a letter (the closest in xy)."""
    p = tmp_path / "cal.npz"
    _write_full_calibration_npz(p)
    models = load_calibration_samples(p)
    # E is at (0.5, 0.5).
    got = classify_gaze_mahalanobis(0.5, 0.5, models, gaze_depth_cm=None)
    assert got == "E"


def test_classify_depth_disambiguates_xy_clash(tmp_path):
    """Two letters at the SAME (x, y) but different depths. The
    classifier must use the depth axis to disambiguate."""
    p = tmp_path / "cal_depth_clash.npz"
    rng = np.random.default_rng(0)
    rows = []
    labels = []
    # X and Y both at (0.5, 0.5); X at 50 cm, Y at 80 cm.
    # (Borrow the 'X' / 'Y' name shape — use real letters A and I from LETTERS.)
    for ch, depth in [("A", 50.0), ("I", 80.0)]:
        for _ in range(20):
            rows.append([
                0.5 + rng.normal(0, 0.005),
                0.5 + rng.normal(0, 0.005),
                1.0,
                depth + rng.normal(0, 1.0),
            ])
            labels.append(ch)
    np.savez_compressed(
        p,
        T=np.zeros(len(rows)),
        G=np.array(rows, dtype=np.float32),
        labels=np.array(labels, dtype="S1"),
        centroids=np.full((9, 3), np.nan, dtype=np.float32),
        letters=np.array(list(LETTERS), dtype="S1"),
        meta=dict(version=1),
    )
    models = load_calibration_samples(p)
    # Runtime gaze at (0.5, 0.5) — x/y matches both A and I.
    # Depth 49 cm -> should pick A; depth 79 -> should pick I.
    assert classify_gaze_mahalanobis(0.5, 0.5, models, gaze_depth_cm=49.0) == "A"
    assert classify_gaze_mahalanobis(0.5, 0.5, models, gaze_depth_cm=79.0) == "I"


def test_classify_respects_available_letters(tmp_path):
    """When `available_letters` restricts the eligible set, only those
    letters can be returned."""
    p = tmp_path / "cal.npz"
    _write_full_calibration_npz(p)
    models = load_calibration_samples(p)
    # Gaze at E's mean -> E without restriction.
    mx, my, md = models["E"]["mean"]
    assert classify_gaze_mahalanobis(mx, my, models, gaze_depth_cm=md) == "E"
    # Restrict to A and I -> one of them wins.
    got = classify_gaze_mahalanobis(
        mx, my, models, gaze_depth_cm=md,
        available_letters=["A", "I"],
    )
    assert got in ("A", "I")


def test_classify_returns_none_when_no_eligible(tmp_path):
    """available_letters empty / no overlap -> None."""
    p = tmp_path / "cal.npz"
    _write_full_calibration_npz(p)
    models = load_calibration_samples(p)
    assert classify_gaze_mahalanobis(
        0.5, 0.5, models, available_letters=[]
    ) is None
    # No overlap.
    assert classify_gaze_mahalanobis(
        0.5, 0.5, models, available_letters=["Z"]
    ) is None


def test_classify_max_mahal_dist_skips_far_gaze(tmp_path):
    """A gaze sample very far from every centroid + max_mahal_dist
    set -> None (caller treats this as skip-GO)."""
    p = tmp_path / "cal.npz"
    _write_full_calibration_npz(p)
    models = load_calibration_samples(p)
    # Gaze at (5, 5) is ~100s of Mahalanobis units from every letter
    # (sigma ~0.005 in xy). max_mahal_dist=10 -> None.
    assert classify_gaze_mahalanobis(
        5.0, 5.0, models, gaze_depth_cm=60.0, max_mahal_dist=10.0
    ) is None
    # Without the ceiling, the nearest letter still comes back.
    got = classify_gaze_mahalanobis(
        5.0, 5.0, models, gaze_depth_cm=60.0, max_mahal_dist=None
    )
    assert got in LETTERS


@pytest.mark.parametrize("gx, gy", [
    (float("nan"), 0.5),
    (0.5, float("inf")),
])
def test_classify_rejects_nonfinite_gaze(tmp_path, gx, gy):
    """The classifier MUST reject non-finite gaze inputs loudly."""
    p = tmp_path / "cal.npz"
    _write_full_calibration_npz(p)
    models = load_calibration_samples(p)
    with pytest.raises(ValueError):
        classify_gaze_mahalanobis(gx, gy, models, gaze_depth_cm=60.0)


def test_load_rejects_npz_without_G(tmp_path):
    """An NPZ that has centroids but no `G` per-sample array can't be
    used with the Mahalanobis classifier -> ValueError."""
    p = tmp_path / "centroids_only.npz"
    np.savez_compressed(
        p,
        centroids=np.zeros((9, 3), dtype=np.float32),
        letters=np.array(list(LETTERS), dtype="S1"),
    )
    with pytest.raises(ValueError, match="missing 'G'"):
        load_calibration_samples(p)


def test_load_rejects_missing_file(tmp_path):
    """Per CLAUDE.md fail-fast: missing NPZ -> FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_calibration_samples(tmp_path / "does_not_exist.npz")


# ============================================================
# Cloud-vote classifier — per-sample majority vote across the runtime
# selection window (added 2026-05-21 to replace the single-median
# classify call in tools/gaze_to_tiago_test.py).
# ============================================================
def _make_sample(x, y, depth, *, conf=1.0, t=0.0):
    """Build one (t, x, y, conf_or_worn, depth) tuple in the runtime
    sample format that classify_gaze_cloud_vote expects."""
    return (float(t), float(x), float(y), float(conf), float(depth))


def test_cloud_vote_unanimous_returns_that_letter(tmp_path):
    """All samples land at E's calibration mean -> 100% E vote."""
    p = tmp_path / "cal.npz"
    _write_full_calibration_npz(p)
    models = load_calibration_samples(p)
    mx, my, md = models["E"]["mean"]
    samples = [_make_sample(mx, my, md) for _ in range(50)]
    letter, votes = classify_gaze_cloud_vote(samples, models)
    assert letter == "E"
    assert votes == {"E": 50}


def test_cloud_vote_outlier_does_not_flip_majority(tmp_path):
    """49 samples at E + 1 outlier at A -> still E. The single-median
    classifier was prone to outlier-driven misclassification; voting
    is robust to a handful of bad samples."""
    p = tmp_path / "cal.npz"
    _write_full_calibration_npz(p)
    models = load_calibration_samples(p)
    ex, ey, ed = models["E"]["mean"]
    ax, ay, ad = models["A"]["mean"]
    samples = [_make_sample(ex, ey, ed) for _ in range(49)]
    samples.append(_make_sample(ax, ay, ad))
    letter, votes = classify_gaze_cloud_vote(samples, models)
    assert letter == "E"
    assert votes.get("E", 0) == 49
    assert votes.get("A", 0) == 1


def test_cloud_vote_filters_below_conf_threshold(tmp_path):
    """Samples whose `conf_or_worn` is below the threshold cast no
    vote — used to drop worn=False rows."""
    p = tmp_path / "cal.npz"
    _write_full_calibration_npz(p)
    models = load_calibration_samples(p)
    ex, ey, ed = models["E"]["mean"]
    # 30 samples at conf=1.0 (kept), 70 at conf=0.0 (dropped).
    samples = (
        [_make_sample(ex, ey, ed, conf=1.0) for _ in range(30)]
        + [_make_sample(ex, ey, ed, conf=0.0) for _ in range(70)]
    )
    letter, votes = classify_gaze_cloud_vote(
        samples, models, conf_threshold=0.5
    )
    assert letter == "E"
    assert votes == {"E": 30}


def test_cloud_vote_empty_samples_returns_none(tmp_path):
    """No samples -> (None, empty dict)."""
    p = tmp_path / "cal.npz"
    _write_full_calibration_npz(p)
    models = load_calibration_samples(p)
    letter, votes = classify_gaze_cloud_vote([], models)
    assert letter is None
    assert votes == {}


def test_cloud_vote_all_filtered_returns_none(tmp_path):
    """All samples below conf_threshold -> no votes cast -> None."""
    p = tmp_path / "cal.npz"
    _write_full_calibration_npz(p)
    models = load_calibration_samples(p)
    ex, ey, ed = models["E"]["mean"]
    samples = [_make_sample(ex, ey, ed, conf=0.1) for _ in range(10)]
    letter, votes = classify_gaze_cloud_vote(
        samples, models, conf_threshold=0.5
    )
    assert letter is None
    assert votes == {}


def test_cloud_vote_per_sample_depth_fallback(tmp_path):
    """Sample with NaN depth still votes (via 2D fallback for that
    sample). Mixed-validity input doesn't drop those samples entirely."""
    p = tmp_path / "cal.npz"
    _write_full_calibration_npz(p)
    models = load_calibration_samples(p)
    ex, ey, ed = models["E"]["mean"]
    samples = (
        [_make_sample(ex, ey, ed) for _ in range(15)]  # valid depth
        + [_make_sample(ex, ey, float("nan")) for _ in range(15)]  # 2D fallback
    )
    letter, votes = classify_gaze_cloud_vote(samples, models)
    assert letter == "E"
    assert votes.get("E", 0) == 30  # both batches voted for E


def test_cloud_vote_max_mahal_skips_far_samples(tmp_path):
    """Samples too far from any centroid (max_mahal_dist exceeded)
    don't vote, but other samples still do."""
    p = tmp_path / "cal.npz"
    _write_full_calibration_npz(p)
    models = load_calibration_samples(p)
    ex, ey, ed = models["E"]["mean"]
    # 20 good samples + 20 way-off samples.
    samples = (
        [_make_sample(ex, ey, ed) for _ in range(20)]
        + [_make_sample(5.0, 5.0, 200.0) for _ in range(20)]
    )
    letter, votes = classify_gaze_cloud_vote(
        samples, models, max_mahal_dist=5.0
    )
    assert letter == "E"
    assert votes == {"E": 20}


def test_cloud_vote_respects_available_letters(tmp_path):
    """Restricting `available_letters` confines voting to that subset
    even when the data clearly points at an excluded letter."""
    p = tmp_path / "cal.npz"
    _write_full_calibration_npz(p)
    models = load_calibration_samples(p)
    ex, ey, ed = models["E"]["mean"]
    samples = [_make_sample(ex, ey, ed) for _ in range(20)]
    letter, votes = classify_gaze_cloud_vote(
        samples, models, available_letters=["A", "I"]
    )
    assert letter in ("A", "I")
    assert set(votes.keys()).issubset({"A", "I"})


# ============================================================
# Diagnostic helper: per_letter_distance_breakdown (added 2026-05-21
# for the --verbose path of tools/gaze_to_tiago_test.py).
# ============================================================
def test_breakdown_returns_one_entry_per_eligible_letter(tmp_path):
    """All 9 letters have 3D models -> breakdown has 9 entries with
    both mean_3d and mean_2d populated."""
    p = tmp_path / "cal.npz"
    _write_full_calibration_npz(p)
    models = load_calibration_samples(p)
    mx, my, md = models["E"]["mean"]
    samples = [_make_sample(mx, my, md) for _ in range(20)]
    out = per_letter_distance_breakdown(samples, models)
    assert set(out.keys()) == set(LETTERS)
    for ch, info in out.items():
        assert info["n_used"] == 20
        assert info["has_3d_model"] is True
        assert info["mean_3d"] >= 0.0
        assert info["mean_2d"] >= 0.0


def test_breakdown_winning_letter_has_smallest_3d_distance(tmp_path):
    """The letter the cloud-vote picks should also be (close to) the
    smallest-mean-3D-distance letter in the breakdown. They aren't
    identical metrics (vote counts vs mean distance), but for tight
    synthetic clusters they agree."""
    p = tmp_path / "cal.npz"
    _write_full_calibration_npz(p)
    models = load_calibration_samples(p)
    mx, my, md = models["E"]["mean"]
    samples = [_make_sample(mx, my, md) for _ in range(20)]
    winner, _ = classify_gaze_cloud_vote(samples, models)
    out = per_letter_distance_breakdown(samples, models)
    sorted_by_3d = sorted(out.items(), key=lambda kv: kv[1]["mean_3d"])
    assert sorted_by_3d[0][0] == winner == "E"


def test_breakdown_letter_without_3d_model_falls_back_to_2d(tmp_path):
    """A letter with no valid-depth calibration samples returns
    mean_3d == mean_2d and has_3d_model=False."""
    p = tmp_path / "cal.npz"
    depth_map = {ch: 60.0 + 4 * i for i, ch in enumerate(LETTERS) if ch != "E"}
    _write_full_calibration_npz(p, depth_for_letters=depth_map)
    models = load_calibration_samples(p)
    ex, ey, _ = models["E"]["mean"]
    samples = [_make_sample(ex, ey, 60.0) for _ in range(20)]
    out = per_letter_distance_breakdown(samples, models)
    assert "E" in out
    assert out["E"]["has_3d_model"] is False
    assert abs(out["E"]["mean_3d"] - out["E"]["mean_2d"]) < 1e-9


def test_breakdown_empty_samples_returns_empty(tmp_path):
    """No samples -> empty dict (consistent with cloud-vote)."""
    p = tmp_path / "cal.npz"
    _write_full_calibration_npz(p)
    models = load_calibration_samples(p)
    assert per_letter_distance_breakdown([], models) == {}


def test_breakdown_respects_available_letters(tmp_path):
    """available_letters subsets the returned breakdown."""
    p = tmp_path / "cal.npz"
    _write_full_calibration_npz(p)
    models = load_calibration_samples(p)
    samples = [_make_sample(0.5, 0.5, 60.0) for _ in range(5)]
    out = per_letter_distance_breakdown(
        samples, models, available_letters=["A", "B", "C"]
    )
    assert set(out.keys()) == {"A", "B", "C"}
