"""Stage 0: multi-sample target seed.

The UI (out of scope here) presents 3-4 live-play frames spread across the match
and the user taps their player in each. This module turns those taps into the
two artefacts the rest of the pipeline needs:

  * a robust **median kit colour** (Stage 6 team filter), built from several
    samples so a single bad crop can't skew it, and
  * an **appearance gallery** of torso crops spread across the timeline, which
    Stage 7 (Step 4) will turn into embeddings for same-kit disambiguation.

Building it from multiple samples across the timeline captures appearance drift
(lighting, sleeves, mud, bibs) so a late-match contact still matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from polyfut_v2.pipeline.color import hsv_feature, median_hsv, torso_crop, torso_hsv


@dataclass
class TargetSeed:
    """The target player's colour + appearance model."""

    kit_hsv: np.ndarray | None                       # median torso HSV across samples
    gallery: list[np.ndarray] = field(default_factory=list)  # torso crops (for Stage 7)
    n_samples: int = 0

    def is_weak(self) -> bool:
        """True if built from a single sample — the doc warns to enlarge the
        review montage in this case (weak gallery)."""
        return self.n_samples <= 1

    def has_color(self) -> bool:
        return self.kit_hsv is not None


def build_seed_from_torso_crops(crops: list[np.ndarray]) -> TargetSeed:
    """Build a seed directly from pre-cropped torso images (test/CLI path)."""
    feats = [f for f in (_hsv_of(c) for c in crops) if f is not None]
    return TargetSeed(
        kit_hsv=median_hsv(feats),
        gallery=[c for c in crops if c is not None and c.size > 0],
        n_samples=len(crops),
    )


def build_seed_from_taps(
    samples: list[tuple[np.ndarray, tuple[float, float]]],
    player_detector,
    *,
    max_tap_dist_px: float = 80.0,
    min_torso_px: int = 0,
) -> TargetSeed:
    """Build a seed from (frame, tap_point) pairs.

    For each tap, the player detector finds candidate boxes and the one whose
    box the tap falls in (or nearest, within ``max_tap_dist_px``) is taken as
    the target; its torso crop feeds both the colour and the gallery.

    ``min_torso_px`` mirrors the Stage 6 guard so the seed kit colour is built
    with the same reliability floor as the contacts it is compared against.
    """
    from polyfut_v2.pipeline.player_contacts import nearest_player  # local import (avoid cycle)

    feats: list[np.ndarray] = []
    gallery: list[np.ndarray] = []
    n = 0
    for frame, tap in samples:
        n += 1
        players = player_detector.detect(frame, tap)
        pl, _dist = nearest_player(players, tap, max_tap_dist_px)
        if pl is None:
            continue
        crop = torso_crop(frame, pl.bbox)
        hsv = torso_hsv(frame, pl.bbox, min_area=min_torso_px)
        if crop is not None and crop.size > 0:
            gallery.append(crop)
        if hsv is not None:
            feats.append(hsv)
    return TargetSeed(kit_hsv=median_hsv(feats), gallery=gallery, n_samples=n)


def _hsv_of(crop: np.ndarray) -> np.ndarray | None:
    """Median HSV of an already-cropped torso image."""
    if crop is None or crop.size == 0:
        return None
    return hsv_feature(crop)
