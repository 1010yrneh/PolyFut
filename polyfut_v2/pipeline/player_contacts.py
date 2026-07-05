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
from polyfut_v2.pipeline.color import hsv_distance, median_hsv, torso_hsv
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


def _sample_jersey(
    frames: list[tuple[int, np.ndarray]],
    ball_pt: tuple[float, float],
    detector,
    cfg: PipelineV2Config,
) -> tuple[np.ndarray | None, int]:
    """Median torso HSV of the ball-nearest player across the window frames."""
    feats: list[np.ndarray] = []
    for _idx, frame in frames:
        players = detector.detect(frame, ball_pt)
        pl, _d = nearest_player(players, ball_pt, cfg.contact_max_player_dist_px)
        if pl is None:
            continue
        hsv = torso_hsv(frame, pl.bbox, min_area=cfg.color_min_torso_px)
        if hsv is not None:
            feats.append(hsv)
    return median_hsv(feats), len(feats)


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

    # Contacting player from the frame nearest the contact.
    center_frame = _center_frame(frames, cand.frame_index)
    player_bbox: list[float] | None = None
    player_dist: float | None = None
    if center_frame is not None:
        players = detector.detect(center_frame, ball_pt)
        pl, d = nearest_player(players, ball_pt, cfg.contact_max_player_dist_px)
        if pl is not None:
            player_bbox, player_dist = pl.bbox, d

    jersey_hsv, n_samples = _sample_jersey(frames, ball_pt, detector, cfg)
    color_dist = hsv_distance(jersey_hsv, seed.kit_hsv)
    is_my_team: bool | None
    if color_dist is None:
        is_my_team = None  # couldn't measure colour — leave undecided (keep for recall)
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
    contacts: list[PlayerContact], *, keep_undecided: bool = True
) -> list[PlayerContact]:
    """Keep your-team contacts; drop opponents. Colour-undecided contacts are
    kept by default (recall over precision — the montage is the safety net)."""
    out = []
    for c in contacts:
        if c.is_my_team is True:
            out.append(c)
        elif c.is_my_team is None and keep_undecided:
            out.append(c)
    return out


def _center_frame(
    frames: list[tuple[int, np.ndarray]], target_index: int
) -> np.ndarray | None:
    if not frames:
        return None
    return min(frames, key=lambda fr: abs(fr[0] - target_index))[1]
