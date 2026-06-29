"""
test_registration_view.py — pure quality math for the live 3-D registration view
(WS-4, accuracy-roadmap).

The cv2 window + drawing in ``RegistrationView`` need a display and are HIL-gated;
these pin the hardware-free logic: viewpoint-diversity cone angle, the per-tag
unseen/weak/ok/good classifier, and the overall summary / accept condition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Utils.gaze.registration_view import (  # noqa: E402
    GOOD_DIVERSITY_DEG,
    GOOD_RESIDUAL_MM,
    MIN_VIEWS,
    classify_tags,
    cone_half_angle_deg,
    registration_summary,
)


def test_cone_angle_few_bearings_is_zero():
    assert cone_half_angle_deg([]) == 0.0
    assert cone_half_angle_deg([np.array([0.0, 0.0, 1.0])]) == 0.0


def test_cone_angle_grows_with_spread():
    straight = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 2.0])]
    spread = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0])]
    assert cone_half_angle_deg(straight) < 1.0
    # Two bearings 45° apart sit 22.5° each side of their mean.
    assert abs(cone_half_angle_deg(spread) - 22.5) < 1e-6


def test_cone_angle_is_scale_invariant():
    a = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0])]
    b = [v * 1000.0 for v in a]
    assert abs(cone_half_angle_deg(a) - cone_half_angle_deg(b)) < 1e-9


def test_classify_states():
    ids = [0, 1, 2, 3]
    views = {1: MIN_VIEWS + 5, 2: MIN_VIEWS + 5, 3: 3}  # 0 never seen
    residuals = {1: GOOD_RESIDUAL_MM - 2, 2: 40.0, 3: 5.0}
    diversity = {1: GOOD_DIVERSITY_DEG + 5, 2: GOOD_DIVERSITY_DEG + 5, 3: 30.0}
    q = classify_tags(ids, views, residuals, diversity)
    assert q[0].state == "unseen"          # never detected
    assert q[1].state == "good"            # views + diversity + low residual
    assert q[2].state == "weak"            # residual too high
    assert q[3].state == "weak"            # too few views despite low residual


def test_classify_no_residuals_yet_is_not_good():
    # Before the first fuse converges there is no residual map; a tag can be at most
    # weak (we never call it good without the residual it is gated on).
    ids = [0]
    q = classify_tags(ids, {0: MIN_VIEWS + 50}, None, {0: GOOD_DIVERSITY_DEG + 20})
    assert q[0].state == "weak"


def test_summary_all_good_accept_condition():
    ids = [0, 1]
    good = dict(views={0: MIN_VIEWS, 1: MIN_VIEWS},
                residuals={0: 5.0, 1: 6.0},
                diversity={0: GOOD_DIVERSITY_DEG + 1, 1: GOOD_DIVERSITY_DEG + 1})
    s = registration_summary(classify_tags(ids, **good))
    assert s["all_good"] is True
    assert s["n_good"] == 2 and s["n_total"] == 2
    assert s["weak_ids"] == []
    assert abs(s["mean_residual_mm"] - 5.5) < 1e-9


def test_summary_worklist_lists_weak_and_unseen():
    ids = [0, 1, 2]
    q = classify_tags(ids, views={1: MIN_VIEWS, 2: 2},
                      residuals={1: 5.0, 2: 5.0},
                      diversity={1: GOOD_DIVERSITY_DEG + 1, 2: 5.0})
    s = registration_summary(q)
    assert s["all_good"] is False
    assert s["weak_ids"] == [0, 2]  # 0 unseen, 2 too few views / low diversity
