"""k-means splits one kit across lighting conditions, so the candidate list
reaches the model holding a shirt and its washed-out twin. That is a coin flip
the model cannot win: on a live run it chose the pale orange #bc9766 over the
truer #8e551c, which kept just as many of your own touches (98%) but removed
95% of the opponent's instead of 100%.

The merge is on HUE, not colour distance, and the reason is measured. On the
ISB/TAS clip the two shades of the orange kit sit dE76 45.5 apart while orange
and the referee's red sit 48.2 apart. Those populations overlap, so no distance
threshold can separate "one kit twice" from "two different things" — the same
failure that ruled out snapping a free-form hex.
"""

from __future__ import annotations

import pytest

from polyfut_video.pipeline.team_preview import (
    _hsv_of, _same_kit_different_light, merge_shades,
)


# --------------------------------------- one kit, two lightings -> merge
@pytest.mark.parametrize("a,b,why", [
    ("#af7a50", "#baaa9d", "orange kit sunlit vs shaded, hue 13 both"),
    ("#3f3b53", "#1e1832", "navy kit at two exposures, hue 125/127"),
    ("#8e551c", "#bc9766", "the live shade tie that cost 5% precision"),
])
def test_same_kit_under_different_light_is_merged(a, b, why):
    assert _same_kit_different_light(a, b), why


# ------------------------------------ genuinely different -> never merge
@pytest.mark.parametrize("a,b,why", [
    ("#af7a50", "#8a4535", "TAS orange vs referee red"),
    ("#3f3b53", "#556b89", "navy vs a lighter blue, hue 125 vs 107"),
    ("#8f561b", "#110a25", "the two real kits"),
    ("#87ceeb", "#0a1a3f", "sky blue vs navy: same hue, opposite brightness"),
    ("#cfcfcf", "#8f8f8f", "two greys: hue is noise there"),
])
def test_different_things_are_kept_apart(a, b, why):
    assert not _same_kit_different_light(a, b), why


def test_the_case_distance_cannot_solve():
    """Pins why this is a hue rule. The pair that must merge is FURTHER apart
    in Lab than the pair that must not, so any dE threshold gets one wrong."""
    import cv2
    import numpy as np

    def de(x, y):
        def lab(hx):
            h = hx.lstrip("#")
            bgr = np.uint8([[[int(h[4:6], 16), int(h[2:4], 16),
                              int(h[0:2], 16)]]])
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        return float(np.linalg.norm(lab(x) - lab(y)))

    must_merge = de("#af7a50", "#baaa9d")      # one orange kit, two lightings
    must_not = de("#af7a50", "#8a4535")        # orange vs the referee
    assert must_merge < must_not < must_merge + 5, (
        f"merge-pair {must_merge:.1f} vs keep-pair {must_not:.1f} — these "
        f"overlap, which is why the rule is on hue"
    )
    assert _same_kit_different_light("#af7a50", "#baaa9d")
    assert not _same_kit_different_light("#af7a50", "#8a4535")


# ------------------------------------------------ which one survives
def test_the_saturated_reading_survives():
    """The less washed-out reading is the closer one to the kit as worn."""
    for order in (["#baaa9d", "#af7a50"], ["#af7a50", "#baaa9d"]):
        assert merge_shades(order) == ["#af7a50"]


def test_hue_is_circular():
    """Red straddles the 0/180 wrap; two reds must not read as far apart."""
    assert _same_kit_different_light("#8f1a12", "#b31f16")   # hue ~2 and ~2
    a, b = "#a01005", "#a00520"
    ha, hb = _hsv_of(a)[0], _hsv_of(b)[0]
    assert min(abs(ha - hb), 180 - abs(ha - hb)) <= 180


# --------------------------------------------------- the whole palette
def test_the_real_palette_loses_only_its_duplicates():
    real = ["#3f3b53", "#af7a50", "#556b89", "#1e1832", "#8a4535", "#baaa9d"]
    merged = merge_shades(real)
    assert "#af7a50" in merged and "#baaa9d" not in merged   # kept the orange
    assert "#1e1832" in merged and "#3f3b53" not in merged   # kept the navy
    assert "#8a4535" in merged and "#556b89" in merged       # distinct survive
    assert len(merged) == 4


def test_merging_never_empties_or_reorders_beyond_dropping():
    assert merge_shades([]) == []
    assert merge_shades(["#8f561b"]) == ["#8f561b"]
    out = merge_shades(["#8f561b", "#110a25", "#358c42"])
    assert out == ["#8f561b", "#110a25", "#358c42"]          # nothing alike


def test_both_true_kits_survive_a_merge_of_the_buggy_palette():
    """The palette from the failing run: the two purples k-means picked, plus
    the real kits. Whatever else collapses, both real kits must remain
    choosable, or the vision read cannot fix the bug it exists for."""
    palette = ["#beb9e4", "#140f2b", "#8f561b", "#110a25", "#cfcfcf", "#3d7a2f"]
    merged = merge_shades(palette)
    assert "#8f561b" in merged
    assert "#110a25" in merged
