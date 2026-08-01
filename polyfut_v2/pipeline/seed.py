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

from polyfut_v2.pipeline.color import (
    cluster_hsv, jersey_colors, jersey_colors_from_crop, median_hsv, torso_crop,
)


@dataclass
class TargetSeed:
    """The target player's colour + appearance model."""

    kit_hsv: np.ndarray | None                       # dominant torso HSV across samples
    gallery: list[np.ndarray] = field(default_factory=list)  # torso crops (for Stage 7)
    n_samples: int = 0
    # The OTHER team's kit colour (from the team picker), if known. Lets the
    # per-contact team gate use both centroids — a touch clearly closer to the
    # opponent's kit than to yours is dropped even in the ambiguous colour band.
    opponent_kit_hsv: np.ndarray | None = None
    # Confirmed on-pitch positions from the seed taps / seed-clip tracklets
    # (x, y in pipeline px, t_sec). Stage 7 bootstraps the orbital from these
    # so identity tracking starts from "who you clicked", not the first random
    # strong appearance match.
    sightings: list[tuple[float, float, float]] = field(default_factory=list)
    # Additional colours of the SAME kit beyond the dominant one, for kits that
    # aren't one flat colour (red/blue halves, hoops, contrast sleeves). A median
    # across such a kit lands on a colour it does not contain — the classic
    # red+blue → purple — so the distinct modes are kept and matching uses the
    # nearest of them (``my_kits`` / ``opponent_kits``). ``kit_hsv`` stays the
    # dominant colour so single-colour call sites and serialization are unchanged.
    kit_hsv_alts: list[np.ndarray] = field(default_factory=list)
    opponent_kit_hsv_alts: list[np.ndarray] = field(default_factory=list)

    def is_weak(self) -> bool:
        """True if built from a single sample — the doc warns to enlarge the
        review montage in this case (weak gallery)."""
        return self.n_samples <= 1

    def has_color(self) -> bool:
        return self.kit_hsv is not None

    def my_kits(self) -> list[np.ndarray]:
        """Every colour your kit contains (dominant first). Empty if unknown."""
        if self.kit_hsv is None:
            return list(self.kit_hsv_alts)
        return [self.kit_hsv, *self.kit_hsv_alts]

    def opponent_kits(self) -> list[np.ndarray]:
        """Every colour the opponent kit contains (dominant first)."""
        if self.opponent_kit_hsv is None:
            return list(self.opponent_kit_hsv_alts)
        return [self.opponent_kit_hsv, *self.opponent_kit_hsv_alts]


def _kit_colors(feats: list[np.ndarray]) -> tuple[np.ndarray | None, list[np.ndarray]]:
    """Split torso samples into (dominant colour, other colours of the same kit).

    Falls back to the plain median when clustering finds a single mode, so a flat
    single-colour kit behaves exactly as before.
    """
    if not feats:
        return None, []
    modes = cluster_hsv(feats)
    if len(modes) <= 1:
        return median_hsv(feats), []
    return modes[0], modes[1:]


def build_seed_from_torso_crops(crops: list[np.ndarray]) -> TargetSeed:
    """Build a seed directly from pre-cropped torso images (test/CLI path)."""
    feats: list[np.ndarray] = []
    for c in crops:
        feats.extend(_hsv_of(c))
    primary, alts = _kit_colors(feats)
    return TargetSeed(
        kit_hsv=primary,
        kit_hsv_alts=alts,
        gallery=[c for c in crops if c is not None and c.size > 0],
        n_samples=len(crops),
    )


def build_seed_from_taps(
    samples: list[tuple[np.ndarray, tuple[float, float]]],
    player_detector,
    *,
    max_tap_dist_px: float = 80.0,
    min_torso_px: int = 0,
    sample_times_sec: list[float] | None = None,
) -> TargetSeed:
    """Build a seed from (frame, tap_point) pairs.

    For each tap, the player detector finds candidate boxes and the one whose
    box the tap falls in (or nearest, within ``max_tap_dist_px``) is taken as
    the target; its torso crop feeds both the colour and the gallery.

    ``min_torso_px`` mirrors the Stage 6 guard so the seed kit colour is built
    with the same reliability floor as the contacts it is compared against.

    ``sample_times_sec`` (optional, parallel to ``samples``) records when each
    tap happened so Stage 7 can bootstrap the orbital from real sightings.
    """
    from polyfut_v2.pipeline.player_contacts import nearest_player  # local import (avoid cycle)
    from polyfut_v2.pipeline.player_detector import player_contact_class_ids

    feats: list[np.ndarray] = []
    gallery: list[np.ndarray] = []
    sightings: list[tuple[float, float, float]] = []
    n = 0
    contact_ids = None
    human_min = human_max = 0.0
    min_h = min_a = 0.0
    if player_detector is not None and hasattr(player_detector, "cfg"):
        contact_ids = player_contact_class_ids(player_detector.cfg)
        cfg = player_detector.cfg
        human_min = float(getattr(cfg, "player_human_min_aspect", 0.0) or 0.0)
        human_max = float(getattr(cfg, "player_human_max_aspect", 0.0) or 0.0)
        min_h = float(getattr(cfg, "player_min_height_px", 0.0) or 0.0)
        min_a = float(getattr(cfg, "player_min_aspect", 0.0) or 0.0)
    for idx, (frame, tap) in enumerate(samples):
        n += 1
        players = player_detector.detect(frame, tap)
        pl, _dist = nearest_player(
            players, tap, max_tap_dist_px, contact_class_ids=contact_ids,
            min_height_px=min_h, min_aspect=min_a,
            human_min_aspect=human_min, human_max_aspect=human_max,
        )
        if pl is None:
            continue
        crop = torso_crop(frame, pl.bbox)
        # Grass-masked kit colours (no area floor — the grass mask self-guards).
        # Each tap contributes every colour its torso carries, not one median of
        # them: a half-and-half shirt medians to a colour it doesn't contain, and
        # with many taps behind the read a genuine second colour is safe to
        # confirm. The per-contact read stays single-valued.
        for hsv in jersey_colors(frame, pl.bbox):
            feats.append(hsv)
        if crop is not None and crop.size > 0:
            gallery.append(crop)
        t_sec = 0.0
        if sample_times_sec is not None and idx < len(sample_times_sec):
            t_sec = float(sample_times_sec[idx])
        bx = 0.5 * (pl.bbox[0] + pl.bbox[2])
        by = 0.5 * (pl.bbox[1] + pl.bbox[3])
        sightings.append((float(bx), float(by), t_sec))
    primary, alts = _kit_colors(feats)
    return TargetSeed(
        kit_hsv=primary, kit_hsv_alts=alts, gallery=gallery, n_samples=n,
        sightings=sightings,
    )


def _hsv_of(crop: np.ndarray) -> list[np.ndarray]:
    """Grass-masked kit colours of an already-cropped torso image (one entry for
    a flat kit, more for a genuinely multi-coloured one)."""
    return jersey_colors_from_crop(crop)
