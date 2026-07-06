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
from polyfut_v2.pipeline.color import hsv_distance, median_hsv, torso_crop, torso_hsv
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
    radius = cfg.contact_color_window * step
    frames = provider.window(cand.frame_index, radius, step)

    # ONE player-detection pass over the window, reused for BOTH the contacting
    # player (frame nearest the contact) and the jersey colour (median over
    # frames). Previously the centre frame was detected twice (once for the
    # player, again inside jersey sampling) — the dominant Stage 5-6 cost when
    # there are thousands of candidates.
    player_bbox: list[float] | None = None
    player_dist: float | None = None
    best_gap: int | None = None
    feats: list[np.ndarray] = []
    for idx, frame in frames:
        players = detector.detect(frame, ball_pt)
        pl, d = nearest_player(players, ball_pt, cfg.contact_max_player_dist_px)
        if pl is None:
            continue
        hsv = torso_hsv(frame, pl.bbox, min_area=cfg.color_min_torso_px)
        if hsv is not None:
            feats.append(hsv)
        gap = abs(idx - cand.frame_index)
        if best_gap is None or gap < best_gap:
            best_gap, player_bbox, player_dist = gap, pl.bbox, d

    jersey_hsv = median_hsv(feats)
    n_samples = len(feats)
    color_dist = hsv_distance(jersey_hsv, seed.kit_hsv)
    is_my_team: bool | None
    if not cfg.team_filter_enabled or color_dist is None:
        # Filter off (colour unreliable on this footage) or colour unmeasurable →
        # leave undecided. color_dist is still recorded for diagnostics/ranking,
        # but is never used to hard-label — so it can't drop or de-anchor a real
        # touch on footage where colour doesn't separate teams.
        is_my_team = None
    else:
        is_my_team = color_dist <= cfg.team_color_max_dist

    return PlayerContact(
        candidate=cand,
        player_bbox=player_bbox,
        player_dist_px=player_dist,
        jersey_hsv=None if jersey_hsv is None else [float(v) for v in jersey_hsv],
        color_dist=color_dist,
        is_my_team=is_my_team,
        n_color_samples=n_samples,
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


def contact_torso_crops(
    contacts: list[PlayerContact], provider: FrameProvider
) -> list[np.ndarray | None]:
    """Torso crop of each contact's player at its contact frame (for Stage 7
    appearance scoring). None where no player was found."""
    crops: list[np.ndarray | None] = []
    for c in contacts:
        if c.player_bbox is None:
            crops.append(None)
            continue
        win = provider.window(c.candidate.frame_index, 0, 1)
        crops.append(torso_crop(win[0][1], c.player_bbox) if win else None)
    return crops
