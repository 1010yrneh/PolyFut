"""Stage 7: target confidence = appearance x orbital, scored sequentially.

Each surviving your-team contact gets a *confidence* (not a hard yes/no) used to
rank the review montage and to auto-accept / auto-hide the extremes. Two
complementary signals are multiplied:

  confidence = appearance_match(gallery) x orbital_prior(dt, distance)

Orbital prior
-------------
A bounded region ("orbital") predicts where the target can be, anchored on the
last high-confidence sighting, its radius growing with the time gap. A contact
inside is not penalised; one that would need the player to teleport is
down-weighted — but only ever **boosted / tie-broken, never hard-rejected**
(the prior is floored well above zero), so a shaky prior can never turn a real
touch into a silent false negative.

Safety rails from the design doc (§7):
  * anchor only on *high-confidence appearance* matches (wrong anchors propagate);
  * let the radius grow while unanchored, and past ``orbital_max_gap_sec`` the
    orbital covers the pitch → it degrades gracefully to "no prior" (neutral);
  * positions are pixel-space here — a ``transform`` hook (identity by default)
    is where camera-motion compensation / pitch homography plug in later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.appearance import AppearanceModel, HistogramAppearance
from polyfut_v2.pipeline.player_contacts import PlayerContact
from polyfut_v2.pipeline.seed import TargetSeed

Transform = Callable[[tuple[float, float], float], tuple[float, float]]


def _identity(xy: tuple[float, float], t: float) -> tuple[float, float]:
    return xy


@dataclass
class ScoredContact:
    contact: PlayerContact
    appearance_score: float | None   # gallery similarity [0,1], or None (unmeasured)
    orbital_prior: float             # [orbital_floor, 1]
    confidence: float                # final [0,1]
    tracklet_id: int
    anchored: bool                   # this contact (re)anchored the orbital

    def to_dict(self) -> dict:
        return {
            **self.contact.to_dict(),
            "appearance_score": None if self.appearance_score is None
            else round(self.appearance_score, 4),
            "orbital_prior": round(self.orbital_prior, 4),
            "confidence": round(self.confidence, 4),
            "tracklet_id": self.tracklet_id,
            "anchored": self.anchored,
        }


def orbital_prior(dist_px: float, dt_sec: float, cfg: PipelineV2Config) -> float:
    """Motion-continuity prior in [orbital_floor, 1].

    Inside the (time-growing) orbital radius → 1.0. Outside → decays with how far
    past the radius the contact is, but never below ``orbital_floor``. Past
    ``orbital_max_gap_sec`` there is no usable signal → neutral 1.0.
    """
    if dt_sec > cfg.orbital_max_gap_sec:
        return 1.0
    radius = cfg.orbital_base_px + cfg.orbital_growth_px_s * max(0.0, dt_sec)
    if dist_px <= radius:
        return 1.0
    over = (dist_px - radius) / max(radius, 1e-6)
    return max(cfg.orbital_floor, 1.0 - cfg.orbital_falloff * over)


def score_contacts(
    contacts: list[PlayerContact],
    crops: list[np.ndarray | None],
    seed: TargetSeed,
    cfg: PipelineV2Config | None = None,
    *,
    appearance: AppearanceModel | None = None,
    transform: Transform | None = None,
) -> list[ScoredContact]:
    """Score contacts sequentially (time order), linking them into tracklets and
    propagating a motion anchor from strong appearance matches.

    ``crops[i]`` is the torso crop for ``contacts[i]`` (or None if unavailable).
    Returns scored contacts in ascending ``processed_sec``.
    """
    cfg = cfg or PipelineV2Config()
    appearance = appearance or HistogramAppearance()
    transform = transform or _identity

    gallery = appearance.gallery_descriptors(seed.gallery) if seed.gallery else []

    order = sorted(range(len(contacts)), key=lambda i: contacts[i].candidate.processed_sec)

    anchor: tuple[float, float, float] | None = None  # (x, y, t)
    tracklet_id = -1
    last_t: float | None = None
    scored: list[ScoredContact] = []

    for i in order:
        c = contacts[i]
        t = c.candidate.processed_sec
        xy = transform((c.candidate.x, c.candidate.y), t)

        # New tracklet when the temporal gap is too large to link.
        new_tracklet = last_t is None or (t - last_t) > cfg.tracklet_max_gap_sec
        if new_tracklet:
            tracklet_id += 1

        # Orbital prior against the current anchor (before this contact can
        # re-anchor), neutral if unanchored.
        if anchor is None:
            prior = 1.0
        else:
            dt = t - anchor[2]
            dist = math.hypot(xy[0] - anchor[0], xy[1] - anchor[1])
            prior = orbital_prior(dist, dt, cfg)

        app = appearance.gallery_score(crops[i], gallery) if i < len(crops) else None
        app_eff = app if app is not None else cfg.appearance_default
        confidence = max(0.0, min(1.0, app_eff * prior))

        # (Re)anchor only on a high-confidence, confirmed-team appearance match
        # that is ALSO motion-consistent with the current chain. With identical
        # kits, appearance alone can't stop a teammate touch from hijacking the
        # anchor, so a mid-tracklet contact must sit inside the orbital
        # (prior == 1.0) to (re)anchor. Bootstrapping a fresh tracklet (or the
        # very first anchor) is exempt — there is no chain to be consistent with.
        strong = (
            app is not None
            and app >= cfg.orbital_anchor_min
            and c.is_my_team is True
        )
        motion_ok = anchor is None or new_tracklet or prior >= 1.0
        anchored = strong and motion_ok
        if anchored:
            anchor = (xy[0], xy[1], t)

        last_t = t
        scored.append(ScoredContact(
            contact=c,
            appearance_score=app,
            orbital_prior=prior,
            confidence=confidence,
            tracklet_id=tracklet_id,
            anchored=anchored,
        ))

    return scored
