"""The pitch landmark table exists twice and the two copies must agree.

``PF_PITCH_LANDMARKS`` in script.js is a hand-maintained mirror of ``LANDMARKS``
here. The browser fits the pitch with one and the server re-fits it with the
other, so any divergence quietly moves the pitch between what the user approved
on screen and what the pipeline actually uses.

It had already drifted: the right end of the pitch had no 6-yard box and no
goalposts while the left end had both, so a camera pointed at the right goal —
most of this footage — could not mark the crispest lines available to it.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

from polyfut_v2.pipeline.pitch_calibration import LANDMARKS, landmark_xy

L, W = 100.0, 64.0
_JS = Path(__file__).resolve().parents[2] / "script.js"


def _js_landmarks() -> dict[str, tuple[str, str]]:
    src = io.open(_JS, encoding="utf-8").read()
    blk = src.split("PF_PITCH_LANDMARKS = [", 1)[1].split("];", 1)[0]
    rows = re.findall(r"\[\s*'([^']+)'\s*,\s*'([^']*)'\s*,\s*'([^']*)'", blk)
    return {k: (x, y) for k, x, y in rows}


def test_the_two_tables_hold_the_same_landmarks():
    js = _js_landmarks()
    assert set(js) == set(LANDMARKS), {
        "only_python": sorted(set(LANDMARKS) - set(js)),
        "only_js": sorted(set(js) - set(LANDMARKS)),
    }


def test_the_two_tables_agree_on_every_position():
    js = _js_landmarks()
    differing = {k: (LANDMARKS[k], js[k]) for k in LANDMARKS
                 if js.get(k) != LANDMARKS[k]}
    assert not differing, differing


def test_every_landmark_evaluates_to_a_point_on_the_pitch():
    for key in LANDMARKS:
        p = landmark_xy(key, L, W)
        assert p is not None, key
        assert -1.0 <= p[0] <= L + 1.0, (key, p)
        assert -1.0 <= p[1] <= W + 1.0, (key, p)


@pytest.mark.parametrize("left", sorted(
    k for k in LANDMARKS if "_L_" in k or k.endswith("_L")))
def test_each_left_landmark_has_a_mirrored_right_one(left):
    """The asymmetry that prompted this: four goalarea_L_* and two post_L_*
    existed with no right-hand counterpart."""
    right = (left[:-2] + "_R") if left.endswith("_L") else left.replace("_L_", "_R_")
    assert right in LANDMARKS, f"{left} has no mirror {right}"
    lx, ly = landmark_xy(left, L, W)
    rx, ry = landmark_xy(right, L, W)
    assert rx == pytest.approx(L - lx), (left, right)
    assert ry == pytest.approx(ly), (left, right)


def test_both_goals_can_be_marked():
    """The concrete capability that was missing."""
    for key in ("goalarea_R_goalline_near", "goalarea_R_goalline_far",
                "goalarea_R_outer_near", "goalarea_R_outer_far",
                "post_R_near", "post_R_far"):
        assert key in LANDMARKS
        assert key in _js_landmarks()
