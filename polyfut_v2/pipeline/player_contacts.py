"""Stages 5 + 6: attribute each contact to a player and apply the team filter.

For every kinematic contact candidate (Step 2) this:

  * seeks a small window of frames around the contact,
  * runs sparse player detection in a crop around the ball (Stage 5),
  * picks the *contacting* player (nearest to the ball, within a distance gate),
  * samples that player's jersey colour across the window and compares it to the
    seed kit colour (Stage 6) to decide your-team vs opponent.

Opponent contacts are dropped; your-team (and colour-undecided) contacts survive
to the scoring / review stages. Identity — you vs a same-kit teammate — is *not*
resolved here; that is Stage 7 + the human montage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.color import hsv_distance, jersey_hsv, median_hsv, torso_crop
from polyfut_v2.pipeline.contacts import ContactCandidate
from polyfut_v2.pipeline.player_detector import PlayerDetection
from polyfut_v2.pipeline.seed import TargetSeed


def point_to_bbox_dist(point: tuple[float, float], bbox: list[float]) -> float:
    """Euclidean distance from a point to a bbox (0 if inside)."""
    px, py = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def nearest_player(
    players: list[PlayerDetection],
    point: tuple[float, float],
    max_dist: float,
) -> tuple[PlayerDetection | None, float]:
    """Player whose box is closest to ``point`` within ``max_dist`` (else None)."""
    best: PlayerDetection | None = None
    best_d = float("inf")
    for p in players:
        d = point_to_bbox_dist(point, p.bbox)
        if d < best_d:
            best_d, best = d, p
    if best is None or best_d > max_dist:
        return None, best_d
    return best, best_d


class FrameProvider(Protocol):
    def window(
        self, center_index: int, radius: int, step: int
    ) -> list[tuple[int, np.ndarray]]:
        """Frames near ``center_index`` (inclusive), ordered by index."""
        ...


@dataclass
class PlayerContact:
    """A contact candidate enriched with its player + team decision."""

    candidate: ContactCandidate
    player_bbox: list[float] | None
    player_dist_px: float | None
    jersey_hsv: list[float] | None
    color_dist: float | None
    is_my_team: bool | None      # True / False / None (undecided — no colour)
    n_color_samples: int
    # The contacting player's torso crop at the contact frame, captured during
    # enrichment for Stage 7 appearance scoring (avoids a second decode pass).
    # Not serialized.
    torso_crop: object = None

    def to_dict(self) -> dict:
        return {
            **self.candidate.to_dict(),
            "player_bbox": None if self.player_bbox is None
            else [round(float(v), 2) for v in self.player_bbox],
            "player_dist_px": None if self.player_dist_px is None
            else round(self.player_dist_px, 2),
            "jersey_hsv": None if self.jersey_hsv is None
            else [round(float(v), 1) for v in self.jersey_hsv],
            "color_dist": None if self.color_dist is None else round(self.color_dist, 2),
            "is_my_team": self.is_my_team,
            "n_color_samples": self.n_color_samples,
        }


def enrich_contact(
    cand: ContactCandidate,
    provider: FrameProvider,
    detector,
    seed: TargetSeed,
    cfg: PipelineV2Config,
) -> PlayerContact:
    ball_pt = (cand.x, cand.y)
    step = max(1, cfg.ball_sample_every_n * cfg.contact_color_step)
    # When the colour filter is off we only need the contacting player on the
    # frame nearest the contact — 1 detection instead of (2*window+1). Jersey
    # colour is sampled across the window only when the filter will actually use
    # it. This is the dominant Stage 5-6 cost.
    radius = cfg.contact_color_window * step if cfg.team_filter_enabled else 0
    frames = provider.window(cand.frame_index, radius, step)

    player_bbox: list[float] | None = None
    player_dist: float | None = None
    best_gap: int | None = None
    torso: np.ndarray | None = None
    feats: list[np.ndarray] = []
    for idx, frame in frames:
        players = detector.detect(frame, ball_pt)
        pl, d = nearest_player(players, ball_pt, cfg.contact_max_player_dist_px)
        if pl is None:
            continue
        # No min-area floor: grass-masking has its own reliability guard (it needs
        # enough non-grass pixels), and the old area floor (tuned for grass-
        # contaminated torso_hsv) would blind the team gate on exactly the small,
        # distant players where wrong-team touches happen.
        hsv = jersey_hsv(frame, pl.bbox)
        if hsv is not None:
            feats.append(hsv)
        gap = abs(idx - cand.frame_index)
        if best_gap is None or gap < best_gap:
            best_gap, player_bbox, player_dist = gap, pl.bbox, d
            # Capture the crop now (frame already decoded) for Stage 7 — no
            # separate torso-crop pass over the video.
            torso = torso_crop(frame, pl.bbox)

    kit = median_hsv(feats)
    n_samples = len(feats)
    color_dist = hsv_distance(kit, seed.kit_hsv)
    # Grass-masked colour now separates teams reliably, so classify three ways —
    # independent of the aggressive team_filter switch:
    #   clearly the other team (drop) / clearly your team / undecided (keep).
    # The conservative "other" cutoff sits well above the same-team band so only
    # obviously-different kits are ever labelled opponent.
    is_my_team: bool | None
    if color_dist is None:
        is_my_team = None                                  # colour unmeasurable → keep
    elif color_dist > cfg.contact_other_team_dist:
        is_my_team = False                                 # clearly other team → drop
    elif color_dist <= cfg.team_color_max_dist:
        is_my_team = True
    else:
        is_my_team = None                                  # in-between → keep (undecided)

    return PlayerContact(
        candidate=cand,
        player_bbox=player_bbox,
        player_dist_px=player_dist,
        jersey_hsv=None if kit is None else [float(v) for v in kit],
        color_dist=color_dist,
        is_my_team=is_my_team,
        n_color_samples=n_samples,
        torso_crop=torso,
    )


def enrich_contacts(
    candidates: Iterable[ContactCandidate],
    provider: FrameProvider,
    detector,
    seed: TargetSeed,
    cfg: PipelineV2Config | None = None,
) -> list[PlayerContact]:
    cfg = cfg or PipelineV2Config()
    return [enrich_contact(c, provider, detector, seed, cfg) for c in candidates]


def filter_my_team(
    contacts: list[PlayerContact], *, keep_undecided: bool = True, enabled: bool = True
) -> list[PlayerContact]:
    """Keep your-team contacts; drop opponents. Colour-undecided contacts are
    kept by default (recall over precision — the montage is the safety net).

    ``enabled=False`` disables the filter entirely (keep everything), for footage
    where torso colour can't separate teams and dropping would risk a silent
    false negative on the target itself.
    """
    if not enabled:
        return list(contacts)
    out = []
    for c in contacts:
        if c.is_my_team is True:
            out.append(c)
        elif c.is_my_team is None and keep_undecided:
            out.append(c)
    return out


def contact_crops(contacts: list[PlayerContact]) -> list[np.ndarray | None]:
    """Torso crops captured during enrichment, for Stage 7 appearance scoring."""
    return [c.torso_crop for c in contacts]
