"""A colourless crop is not the same thing as a crop with no kit in it.

`_is_neutral_crop` used to be `s < 30 or (v > 210 and s < 50)`, which discarded
every unsaturated crop from the team picker. White is the least saturated thing
on a pitch, so that removed an entire white-kitted team — and black kits, which
are also near-zero saturation, and pale blues.

It barely showed on the footage to hand because turf contamination keeps
measured saturation high. That made it a latent bug that would get *worse* as
footage improved: the cleaner the exposure, the more reliably a white shirt reads
as white and the more certainly it was thrown away.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from polyfut_video.pipeline.team_preview import _is_neutral_crop


def _solid(h, s, v, size=(24, 12)):
    return cv2.cvtColor(np.full((size[0], size[1], 3), (h, s, v), np.uint8),
                        cv2.COLOR_HSV2BGR)


# ------------------------------------------------- kits that must survive
@pytest.mark.parametrize("name,hsv", [
    ("white shirt", (0, 12, 235)),
    ("white in slight shade", (25, 20, 205)),
    ("off-white", (30, 25, 215)),
    ("pale blue", (110, 28, 225)),
    ("black kit", (0, 20, 30)),
    ("very dark navy", (115, 25, 45)),
])
def test_a_real_kit_is_never_dropped_for_lacking_colour(name, hsv):
    assert not _is_neutral_crop(_solid(*hsv)), name


@pytest.mark.parametrize("name,hsv", [
    ("navy", (112, 190, 70)),
    ("yellow", (25, 220, 210)),
    ("red", (0, 200, 170)),
    ("green", (60, 180, 120)),
])
def test_a_saturated_kit_is_kept(name, hsv):
    assert not _is_neutral_crop(_solid(*hsv)), name


# ------------------------------------------- what it is actually there for
@pytest.mark.parametrize("name,hsv", [
    ("worn concrete", (20, 18, 120)),
    ("deep shadow on tarmac", (0, 10, 70)),
    ("washed-out blur", (40, 15, 150)),
])
def test_a_colourless_mid_tone_is_still_dropped(name, hsv):
    assert _is_neutral_crop(_solid(*hsv)), name


def test_the_old_rule_would_have_failed_this():
    """Pins the regression directly: under `s < 30 or (v > 210 and s < 50)`
    both of these were dropped."""
    white = _solid(0, 12, 235)
    black = _solid(0, 20, 30)
    old = lambda c: (lambda hsv: hsv[1] < 30 or (hsv[2] > 210 and hsv[1] < 50))(
        [int(x) for x in np.median(
            cv2.cvtColor(c, cv2.COLOR_BGR2HSV).reshape(-1, 3), axis=0)])
    assert old(white) and old(black)          # the old rule dropped both
    assert not _is_neutral_crop(white)
    assert not _is_neutral_crop(black)
