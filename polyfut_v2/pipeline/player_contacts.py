"""Stages 5 + 6: attribute each contact to a player and apply the team filter.

For every kinematic contact candidate (Step 2) this:

  * seeks a small window of frames around the contact,
  * runs sparse player detection in a crop around the ball (Stage 5),
  * picks the *contacting* player (nearest to the ball, within a distance gate;
    goalkeepers count, referees never do),
  * samples that player's jersey colour across the window and classifies it
    against the per-match team centroids (Stage 6).

Confirmed opponents, officials, and sideline staff are dropped; your-team and
colour-undecided (``unknown``) contacts survive to the scoring / review stages.
Identity — you vs a same-kit teammate — is *not* resolved here; that is Stage 7
+ the human montage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

import numpy as np

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.color import (
    hsv_distance,
    hsv_distance_multi,
    is_off_pitch,
    jersey_hsv,
    kits_separable,
    looks_like_official_kit,
    median_hsv,
    torso_crop,
)
from polyfut_v2.pipeline.contacts import ContactCandidate
from polyfut_v2.pipeline.player_detector import (
    PlayerDetection,
    player_contact_class_ids,
)
from polyfut_v2.pipeline.seed import TargetSeed

# Per-contact team labels (serialized). ``my_team`` / ``opponent`` are the two
# match centroids; ``official`` is model-class referee or soft official kit;
# ``sideline`` is a coach/bench body whose feet aren't on the pitch; ``unknown``
# is keep. Both ``official`` and ``sideline`` are dropped (is_my_team=False).
TEAM_MY = "my_team"
TEAM_OPPONENT = "opponent"
TEAM_OFFICIAL = "official"
TEAM_SIDELINE = "sideline"
TEAM_UNKNOWN = "unknown"


def point_to_bbox_dist(point: tuple[float, float], bbox: list[float]) -> float:
    """Euclidean distance from a point to a bbox (0 if inside)."""
    px, py = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def _looks_like_ball(
    bbox: list[float], *, min_height_px: float, min_aspect: float,
) -> bool:
    """True for a "player" box that's actually the ball misclassified.

    The model occasionally labels the ball itself as "player" — a small,
    roughly-square box sitting almost exactly on the ball (near-zero distance
    to the search point, since that point *is* the ball). A real person, even
    small and distant, is reliably either taller than this or clearly
    taller-than-wide, so a box only counts as ball-shaped when it fails BOTH
    checks — short AND not elongated — to avoid excluding genuine small,
    stocky-looking distant players caught by only one of the two.
    """
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = max(x2 - x1, 1e-6)
    return h < min_height_px and (h / w) < min_aspect


def _looks_like_ball_soft(
    bbox: list[float], *, min_height_px: float, min_aspect: float,
) -> bool:
    """Loose ball-shape: short OR not elongated. Used only with ball-overlap."""
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = max(x2 - x1, 1e-6)
    return h < min_height_px or (h / w) < min_aspect


def looks_like_person(
    bbox: list[float],
    *,
    min_height_px: float = 16.0,
    min_aspect: float = 1.30,
    max_aspect: float = 5.50,
) -> bool:
    """True when a detection's box has standing-person proportions.

    Requires a minimum height and a height/width ratio inside a human band.
    Near-square spare balls and cameras (aspect ≈ 1), tiny fragments, and
    ultra-thin poles fall outside and must not be tagged as players. Pass
    ``min_aspect <= 0`` to disable the aspect band (height floor still applies
    when ``min_height_px > 0``).
    """
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = max(x2 - x1, 1e-6)
    if min_height_px > 0 and h < min_height_px:
        return False
    aspect = h / w
    if min_aspect > 0 and aspect < min_aspect:
        return False
    if max_aspect > 0 and aspect > max_aspect:
        return False
    return True


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def ball_proxy_bbox(point: tuple[float, float], half_px: float) -> list[float]:
    """Square stand-in for the ball detection around a contact point."""
    x, y = point
    h = max(1.0, float(half_px))
    return [x - h, y - h, x + h, y + h]


def _feet_point(bbox: list[float]) -> tuple[float, float]:
    """Centre of the lower third of the box (feet / legs region)."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y1 + (5.0 / 6.0) * (y2 - y1))


def _should_reject_as_ball(
    bbox: list[float],
    *,
    min_height_px: float,
    min_aspect: float,
    ball_bbox: list[float] | None,
    ball_iou_soft: float,
) -> bool:
    """Strict AND without ball overlap; soft OR when heavily overlapping the ball."""
    if min_height_px <= 0:
        return False
    if ball_bbox is not None and ball_iou_soft > 0:
        if _bbox_iou(bbox, ball_bbox) >= ball_iou_soft:
            return _looks_like_ball_soft(
                bbox, min_height_px=min_height_px, min_aspect=min_aspect,
            )
    return _looks_like_ball(
        bbox, min_height_px=min_height_px, min_aspect=min_aspect,
    )


def _should_reject_detection(
    bbox: list[float],
    *,
    min_height_px: float,
    min_aspect: float,
    ball_bbox: list[float] | None = None,
    ball_iou_soft: float = 0.25,
    human_min_aspect: float = 0.0,
    human_max_aspect: float = 0.0,
) -> bool:
    """Reject boxes that aren't person-shaped, or that look like the ball.

    The human-proportion band is opt-in (``human_min_aspect`` / ``max`` > 0) so
    low-level callers that only want the legacy ball soft-reject keep working.
    """
    if human_min_aspect > 0 or human_max_aspect > 0:
        if not looks_like_person(
            bbox,
            min_height_px=min_height_px,
            min_aspect=human_min_aspect,
            max_aspect=human_max_aspect,
        ):
            return True
    return _should_reject_as_ball(
        bbox,
        min_height_px=min_height_px,
        min_aspect=min_aspect,
        ball_bbox=ball_bbox,
        ball_iou_soft=ball_iou_soft,
    )


def _contest_rank_key(
    p: PlayerDetection, point: tuple[float, float],
) -> tuple[float, float, float]:
    """Sort key for contesting players (lower is better).

    Primary: distance from the ball to the feet/legs point — prefers the player
    whose lower body is nearer the ball over a large sideline box that merely
    contains the ball point near its top. Secondary: smaller box (sideline blobs
    lose to compact players). Tertiary: higher confidence.
    """
    fx, fy = _feet_point(p.bbox)
    d_feet = math.hypot(point[0] - fx, point[1] - fy)
    x1, y1, x2, y2 = p.bbox
    area = max(1.0, (x2 - x1) * (y2 - y1))
    return (d_feet, area, -float(p.conf))


@dataclass
class MetricGate:
    """Judge ball-to-player distance in metres instead of pixels.

    Only usable when the user has calibrated the pitch. The pixel gate stays as
    an outer cap; this decides within it.

    Recall-safety, in order:

    * **Unmeasurable keeps.** If either point can't be placed on the pitch (no
      camera track, a shot boundary, above the horizon) the candidate passes and
      the pixel gate alone applies. Never drop a touch because geometry was
      unavailable.
    * **Image adjacency wins.** A ground-plane map cannot know the ball's height;
      an airborne ball projects further away than it is, which can only inflate
      a distance and cause a rejection. So a player within ``override_px`` of the
      ball in the image is kept regardless — that protects headers and close
      control from being rejected by a measurement that cannot see height.
    """

    to_pitch: Callable[[tuple[float, float]], tuple[float, float] | None]
    max_dist_m: float
    override_px: float = 25.0

    def ok(self, ball_pt: tuple[float, float], bbox: list[float],
           dist_px: float) -> bool:
        if dist_px <= self.override_px:
            return True
        d = self.distance_m(ball_pt, bbox)
        return True if d is None else d <= self.max_dist_m

    def distance_m(self, ball_pt: tuple[float, float],
                   bbox: list[float]) -> float | None:
        """Ground distance from the ball to the player's FEET, in metres.

        Feet (box bottom-centre), because the map is of the ground and only a
        ground contact point projects correctly.
        """
        a = self.to_pitch(ball_pt)
        if a is None:
            return None
        feet = ((bbox[0] + bbox[2]) / 2.0, bbox[3])
        b = self.to_pitch(feet)
        if b is None:
            return None
        return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def nearest_player(
    players: list[PlayerDetection],
    point: tuple[float, float],
    max_dist: float,
    *,
    min_height_px: float = 0.0,
    min_aspect: float = 0.0,
    contact_class_ids: set[int] | None = None,
    ball_bbox: list[float] | None = None,
    ball_iou_soft: float = 0.25,
    human_min_aspect: float = 0.0,
    human_max_aspect: float = 0.0,
    metric: "MetricGate | None" = None,
) -> tuple[PlayerDetection | None, float]:
    """Best *contesting* player near ``point`` within ``max_dist`` (else None).

    Skips non-person boxes (cameras, spare balls, poles) when the human-aspect
    band is enabled, ball-shaped / ball-overlapping boxes, and (when
    ``contact_class_ids`` is set) ineligible classes. Ranking uses feet
    proximity + size + confidence rather than raw nearest-box distance alone,
    so a large sideline box that contains the ball does not beat a compact
    player whose feet are nearer.
    """
    candidates = _contest_candidates(
        players, point, max_dist,
        min_height_px=min_height_px, min_aspect=min_aspect,
        contact_class_ids=contact_class_ids, ball_bbox=ball_bbox,
        ball_iou_soft=ball_iou_soft, human_min_aspect=human_min_aspect,
        human_max_aspect=human_max_aspect, metric=metric,
    )
    if not candidates:
        return None, float("inf")
    _key, best, best_d = candidates[0]
    return best, best_d


def _contest_candidates(
    players: list[PlayerDetection],
    point: tuple[float, float],
    max_dist: float,
    *,
    min_height_px: float = 0.0,
    min_aspect: float = 0.0,
    contact_class_ids: set[int] | None = None,
    ball_bbox: list[float] | None = None,
    ball_iou_soft: float = 0.25,
    human_min_aspect: float = 0.0,
    human_max_aspect: float = 0.0,
    metric: "MetricGate | None" = None,
) -> list[tuple[tuple[float, float, float], PlayerDetection, float]]:
    """Eligible contesting players near ``point``, best-ranked first.

    Split out of :func:`nearest_player` so the runner-up is available: whether
    attribution can be trusted depends on how far back the *second* candidate
    is, not only on who came first.
    """
    out: list[tuple[tuple[float, float, float], PlayerDetection, float]] = []
    for p in players:
        if contact_class_ids is not None and int(p.class_id) not in contact_class_ids:
            continue
        if _should_reject_detection(
            p.bbox,
            min_height_px=min_height_px,
            min_aspect=min_aspect,
            ball_bbox=ball_bbox,
            ball_iou_soft=ball_iou_soft,
            human_min_aspect=human_min_aspect,
            human_max_aspect=human_max_aspect,
        ):
            continue
        d = point_to_bbox_dist(point, p.bbox)
        if d > max_dist:
            continue
        if metric is not None and not metric.ok(point, p.bbox, d):
            continue
        out.append((_contest_rank_key(p, point), p, d))
    out.sort(key=lambda t: t[0])
    return out


def attribution_is_ambiguous(
    players: list[PlayerDetection],
    point: tuple[float, float],
    max_dist: float,
    *,
    gap_heights: float = 0.8,
    **kw,
) -> bool:
    """Is there a real contest for "who touched it", or a clear winner?

    A count of nearby bodies answers the wrong question. Eight players can be
    packed around the ball while one of them is unmistakably on it, and two
    players can be genuinely 50/50. What decides whether *attribution* can be
    trusted is how far back the runner-up is.

    The threshold is expressed in multiples of the winning box's own height, so
    it means the same thing near and far from the camera — a player box is a
    direct depth proxy (~1.75m of real person), which a fixed pixel gap is not.

    No runner-up at all ⇒ unambiguous. Unmeasurable ⇒ ambiguous (recall-safe:
    the caller uses this to *relax* a review flag, so absence of evidence must
    not relax it).
    """
    cands = _contest_candidates(players, point, max_dist, **kw)
    if not cands:
        return True
    if len(cands) == 1:
        return False
    best_h = max(1.0, float(cands[0][1].bbox[3] - cands[0][1].bbox[1]))
    gap = float(cands[1][2]) - float(cands[0][2])
    return gap < gap_heights * best_h


def count_contesting_players(
    players: list[PlayerDetection],
    point: tuple[float, float],
    radius: float,
    *,
    min_height_px: float = 0.0,
    min_aspect: float = 0.0,
    contact_class_ids: set[int] | None = None,
    ball_bbox: list[float] | None = None,
    ball_iou_soft: float = 0.25,
    human_min_aspect: float = 0.0,
    human_max_aspect: float = 0.0,
    metric: "MetricGate | None" = None,
) -> int:
    """How many eligible players sit within ``radius`` of the ball.

    Same eligibility filters as :func:`nearest_player` (class + person shape +
    ball-shape), so the count means "bodies that could plausibly be the toucher"
    rather than raw box count. Used to flag crowded contacts (corners,
    goalmouth scrambles).
    """
    n = 0
    for p in players:
        if contact_class_ids is not None and int(p.class_id) not in contact_class_ids:
            continue
        if _should_reject_detection(
            p.bbox,
            min_height_px=min_height_px,
            min_aspect=min_aspect,
            ball_bbox=ball_bbox,
            ball_iou_soft=ball_iou_soft,
            human_min_aspect=human_min_aspect,
            human_max_aspect=human_max_aspect,
        ):
            continue
        d = point_to_bbox_dist(point, p.bbox)
        if d > radius:
            continue
        # Without this, a far-side moment counts everyone within ~56m and looks
        # like a scramble, while a genuine one near the camera spans only ~9m
        # and never trips the flag.
        if metric is not None and not metric.ok(point, p.bbox, d):
            continue
        n += 1
    return n


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
    # Model class of the contacting box (player / keeper); None if no player.
    player_class_id: int | None = None
    # Explicit per-match label: my_team / opponent / official / unknown.
    team_label: str = TEAM_UNKNOWN
    # Stage 7 appearance descriptor cached for Stage 8 grouping reuse (Issue 7).
    # Not serialized.
    appearance_descriptor: object = None
    # Eligible players packed around the ball at the contact, and whether that
    # count crossed ``crowd_min_players`` (corner kick / scramble).
    n_nearby_players: int = 0
    crowded: bool = False

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
            "player_class_id": self.player_class_id,
            "team_label": self.team_label,
            "n_nearby_players": self.n_nearby_players,
            "crowded": self.crowded,
        }


def _team_kit_sets(seed: TargetSeed):
    """(my colours, opponent colours) — every colour each kit contains.

    Falls back to the single ``kit_hsv`` fields for seeds built before
    multi-colour kits existed (and for hand-built test seeds).
    """
    mine = seed.my_kits() if hasattr(seed, "my_kits") else (
        [] if seed.kit_hsv is None else [seed.kit_hsv])
    opp = seed.opponent_kits() if hasattr(seed, "opponent_kits") else (
        [] if getattr(seed, "opponent_kit_hsv", None) is None
        else [seed.opponent_kit_hsv])
    return mine, opp


def _closer_to_opponent(kit, my_dist, seed: TargetSeed, cfg: PipelineV2Config) -> bool:
    """True when a contact's kit (already in the ambiguous 60-115 band vs. your
    kit) is clearly closer to the OPPONENT's kit than to yours.

    Only fires when the opponent kit is known and the two kits are actually
    distinguishable by colour — with multi-coloured kits that means their
    *closest* pair of colours is more than ``team_color_max_dist`` apart, since a
    colour both teams wear cannot separate them. Requires a
    ``contact_opponent_margin`` lead so a near-tie stays undecided (kept for
    review) rather than being dropped.
    """
    mine, opp = _team_kit_sets(seed)
    if not opp or kit is None or my_dist is None:
        return False
    if not kits_separable(mine, opp, cfg.team_color_max_dist):
        return False                                       # kits too alike to tell apart
    opp_dist = hsv_distance_multi(kit, opp)
    return opp_dist is not None and opp_dist + cfg.contact_opponent_margin < my_dist


def _closer_to_my_team(kit, my_dist, seed: TargetSeed, cfg: PipelineV2Config) -> bool:
    """Mirror of ``_closer_to_opponent``: undecided-band kit clearly nearer yours."""
    mine, opp = _team_kit_sets(seed)
    if not opp or kit is None or my_dist is None:
        return False
    if not kits_separable(mine, opp, cfg.team_color_max_dist):
        return False
    opp_dist = hsv_distance_multi(kit, opp)
    return opp_dist is not None and my_dist + cfg.contact_opponent_margin < opp_dist


def classify_team(
    kit: np.ndarray | None,
    seed: TargetSeed,
    cfg: PipelineV2Config,
    *,
    player_class_id: int | None = None,
) -> tuple[bool | None, str, float | None]:
    """Assign ``(is_my_team, team_label, color_dist_to_my_kit)``.

    Uses both match centroids when the opponent kit is known and the two kits
    are colour-separable — shrinking the old 60–115 undecided band. Officials
    (referee class, or a soft black/neon official-kit read that matches neither
    team) are always ``official`` / not my team. Unreadable colour stays
    ``unknown`` (kept for recall).
    """
    ref_id = getattr(cfg, "referee_class_id", None)
    if (
        player_class_id is not None
        and ref_id is not None
        and int(player_class_id) == int(ref_id)
    ):
        return False, TEAM_OFFICIAL, None

    mine, opp = _team_kit_sets(seed)
    # Distance to the NEAREST colour your kit contains, so a red/blue kit is
    # matched by either half instead of by a purple average of the two.
    color_dist = hsv_distance_multi(kit, mine) if kit is not None else None
    if color_dist is None:
        return None, TEAM_UNKNOWN, None

    # Soft official-kit fallback: black / neon yellow that matches neither team
    # — catches refs the model mislabels as ``player``. Never overrides a clear
    # your-team (or opponent) match above.
    if getattr(cfg, "official_kit_reject_enabled", True):
        if looks_like_official_kit(
            kit, mine, opp, team_max_dist=cfg.team_color_max_dist,
        ):
            return False, TEAM_OFFICIAL, color_dist

    # Two-centroid path: positive assignment to the nearer team when separable.
    if _closer_to_opponent(kit, color_dist, seed, cfg):
        return False, TEAM_OPPONENT, color_dist
    if _closer_to_my_team(kit, color_dist, seed, cfg):
        return True, TEAM_MY, color_dist

    # One-sided bands (also the fallback when opponent kit is unknown / alike).
    if color_dist > cfg.contact_other_team_dist:
        return False, TEAM_OPPONENT, color_dist
    if color_dist <= cfg.team_color_max_dist:
        return True, TEAM_MY, color_dist
    return None, TEAM_UNKNOWN, color_dist


def _eligible_players(
    frame: np.ndarray,
    players: list[PlayerDetection],
    cfg: PipelineV2Config,
) -> tuple[list[PlayerDetection], list[PlayerDetection]]:
    """Split detections into on-pitch contestants vs confirmed sideline bodies.

    Sideline / coach boxes (feet not on grass) are withheld from attribution so
    a bench staffer near a touchline throw-in can't win ``nearest_player``.
    """
    if not getattr(cfg, "sideline_reject_enabled", True):
        return list(players), []
    min_g = float(getattr(cfg, "sideline_min_feet_grass", 0.18) or 0.18)
    on_pitch: list[PlayerDetection] = []
    sideline: list[PlayerDetection] = []
    for p in players:
        if is_off_pitch(frame, p.bbox, min_feet_grass=min_g):
            sideline.append(p)
        else:
            on_pitch.append(p)
    return on_pitch, sideline


def enrich_contact(
    cand: ContactCandidate,
    provider: FrameProvider,
    detector,
    seed: TargetSeed,
    cfg: PipelineV2Config,
    mapper=None,
) -> PlayerContact:
    ball_pt = (cand.x, cand.y)
    step = max(1, cfg.ball_sample_every_n * cfg.contact_color_step)
    # When the colour filter is off we only need the contacting player on the
    # frame nearest the contact — 1 detection instead of (2*window+1). Jersey
    # colour is sampled across the window only when the filter will actually use
    # it. This is the dominant Stage 5-6 cost.
    radius = cfg.contact_color_window * step if cfg.team_filter_enabled else 0
    frames = provider.window(cand.frame_index, radius, step)
    detections = [detector.detect(frame, ball_pt) for _idx, frame in frames]
    return _enrich_from_detections(cand, frames, detections, seed, cfg, mapper)


def _metric_gate(cand, cfg, mapper):
    """Metre-space attribution gate for this contact, or None to use pixels.

    Returns None whenever the pitch map is unavailable at this moment (no
    calibration, no camera track, or the contact sits past a shot boundary), so
    the caller silently keeps today's pixel behaviour.
    """
    if mapper is None:
        return None
    # Frame index, not processed_sec: the latter is a compacted timeline.
    fi = int(getattr(cand, "frame_index", -1))
    if fi < 0 or mapper.image_to_pitch(fi) is None:
        return None
    return MetricGate(
        to_pitch=lambda xy, _f=fi: mapper.to_pitch(_f, xy),
        max_dist_m=float(getattr(cfg, "contact_max_player_dist_m", 8.0)),
        override_px=float(getattr(cfg, "contact_metric_override_px", 25.0)),
    )


def _enrich_from_detections(
    cand: ContactCandidate,
    frames: list[tuple[int, np.ndarray]],
    detections: list[list[PlayerDetection]],
    seed: TargetSeed,
    cfg: PipelineV2Config,
    mapper=None,
) -> PlayerContact:
    """Shared enrichment body given already-run player detections per frame."""
    ball_pt = (cand.x, cand.y)
    gate = _metric_gate(cand, cfg, mapper)
    crowd_gate = None if gate is None else MetricGate(
        to_pitch=gate.to_pitch,
        max_dist_m=float(getattr(cfg, "crowd_radius_m", 9.0)),
        override_px=gate.override_px,
    )
    if mapper is not None:
        mapper.note(gate is not None)
    contact_ids = player_contact_class_ids(cfg)
    ball_box = ball_proxy_bbox(ball_pt, cfg.contact_ball_proxy_half_px)
    soft_iou = cfg.contact_ball_iou_soft

    player_bbox: list[float] | None = None
    player_dist: float | None = None
    player_class_id: int | None = None
    best_gap: int | None = None
    torso: np.ndarray | None = None
    feats: list[np.ndarray] = []
    ref_bbox: list[float] | None = None
    ref_dist: float | None = None
    ref_class_id: int | None = None
    ref_gap: int | None = None
    ref_id = getattr(cfg, "referee_class_id", None)
    ref_ids = {int(ref_id)} if ref_id is not None else None
    crowd_on = bool(getattr(cfg, "crowd_detect_enabled", False))
    n_nearby = 0
    # Sticky across the sampled window, like n_nearby: ambiguity on any frame
    # means this contact's attribution is contestable.
    ambiguous = False
    sideline_bbox: list[float] | None = None
    sideline_dist: float | None = None
    sideline_gap: int | None = None

    for (idx, frame), players in zip(frames, detections):
        on_pitch, sideline = _eligible_players(frame, players, cfg)
        if crowd_on:
            # Max over the window: a pack on any sampled frame means the
            # attribution at this contact can't be trusted. Sideline bodies
            # don't count as contesting.
            n_nearby = max(n_nearby, count_contesting_players(
                on_pitch, ball_pt, cfg.crowd_radius_px,
                min_height_px=cfg.player_min_height_px,
                min_aspect=cfg.player_min_aspect,
                contact_class_ids=contact_ids,
                ball_bbox=ball_box, ball_iou_soft=soft_iou,
                human_min_aspect=cfg.player_human_min_aspect,
                human_max_aspect=cfg.player_human_max_aspect,
                metric=crowd_gate,
            ))
            # ...but a pack is only an *attribution* problem when the pack is
            # actually contesting. If one body is clearly on the ball and the
            # next is well behind it, "who touched it" was never in doubt and
            # the surrounding players are bystanders.
            if getattr(cfg, "crowd_require_ambiguity", False) and not ambiguous:
                ambiguous = attribution_is_ambiguous(
                    on_pitch, ball_pt, cfg.crowd_radius_px,
                    gap_heights=float(getattr(cfg, "crowd_ambiguous_gap_heights",
                                              0.8)),
                    min_height_px=cfg.player_min_height_px,
                    min_aspect=cfg.player_min_aspect,
                    contact_class_ids=contact_ids,
                    ball_bbox=ball_box, ball_iou_soft=soft_iou,
                    human_min_aspect=cfg.player_human_min_aspect,
                    human_max_aspect=cfg.player_human_max_aspect,
                    metric=crowd_gate,
                )
        pl, d = nearest_player(
            on_pitch, ball_pt, cfg.contact_max_player_dist_px,
            min_height_px=cfg.player_min_height_px, min_aspect=cfg.player_min_aspect,
            contact_class_ids=contact_ids,
            ball_bbox=ball_box, ball_iou_soft=soft_iou,
            human_min_aspect=cfg.player_human_min_aspect,
            human_max_aspect=cfg.player_human_max_aspect,
            metric=gate,
        )
        if pl is None:
            if ref_ids is not None:
                ref, rd = nearest_player(
                    players, ball_pt, cfg.contact_max_player_dist_px,
                    min_height_px=cfg.player_min_height_px,
                    min_aspect=cfg.player_min_aspect,
                    contact_class_ids=ref_ids,
                    ball_bbox=ball_box, ball_iou_soft=soft_iou,
                    human_min_aspect=cfg.player_human_min_aspect,
                    human_max_aspect=cfg.player_human_max_aspect,
                    metric=gate,
                )
                if ref is not None:
                    gap = abs(idx - cand.frame_index)
                    if ref_gap is None or gap < ref_gap:
                        ref_gap, ref_bbox, ref_dist = gap, ref.bbox, rd
                        ref_class_id = int(ref.class_id)
            # Remember the nearest sideline body so a coach-only contact can be
            # labelled and dropped rather than surviving as "unknown".
            if sideline:
                sl, sd = nearest_player(
                    sideline, ball_pt, cfg.contact_max_player_dist_px,
                    min_height_px=cfg.player_min_height_px,
                    min_aspect=cfg.player_min_aspect,
                    contact_class_ids=contact_ids,
                    ball_bbox=ball_box, ball_iou_soft=soft_iou,
                    human_min_aspect=cfg.player_human_min_aspect,
                    human_max_aspect=cfg.player_human_max_aspect,
                    metric=gate,
                )
                if sl is not None:
                    gap = abs(idx - cand.frame_index)
                    if sideline_gap is None or gap < sideline_gap:
                        sideline_gap, sideline_bbox, sideline_dist = gap, sl.bbox, sd
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
            player_class_id = int(pl.class_id)
            # Capture the crop now (frame already decoded) for Stage 7 — no
            # separate torso-crop pass over the video.
            torso = torso_crop(frame, pl.bbox)

    crowded = crowd_on and n_nearby >= int(cfg.crowd_min_players)
    if crowded and getattr(cfg, "crowd_require_ambiguity", False) and not ambiguous:
        # Bodies nearby, but a clear winner — not an attribution problem.
        crowded = False

    if player_bbox is None and ref_bbox is not None:
        return PlayerContact(
            candidate=cand,
            player_bbox=ref_bbox,
            player_dist_px=ref_dist,
            jersey_hsv=None,
            color_dist=None,
            is_my_team=False,
            n_color_samples=0,
            torso_crop=None,
            player_class_id=ref_class_id,
            team_label=TEAM_OFFICIAL,
            n_nearby_players=n_nearby,
            crowded=crowded,
        )

    if player_bbox is None and sideline_bbox is not None:
        return PlayerContact(
            candidate=cand,
            player_bbox=sideline_bbox,
            player_dist_px=sideline_dist,
            jersey_hsv=None,
            color_dist=None,
            is_my_team=False,
            n_color_samples=0,
            torso_crop=None,
            player_class_id=None,
            team_label=TEAM_SIDELINE,
            n_nearby_players=n_nearby,
            crowded=crowded,
        )

    kit = median_hsv(feats)
    n_samples = len(feats)
    is_my_team, team_label, color_dist = classify_team(
        kit, seed, cfg, player_class_id=player_class_id,
    )

    return PlayerContact(
        candidate=cand,
        player_bbox=player_bbox,
        player_dist_px=player_dist,
        jersey_hsv=None if kit is None else [float(v) for v in kit],
        color_dist=color_dist,
        is_my_team=is_my_team,
        n_color_samples=n_samples,
        torso_crop=torso,
        player_class_id=player_class_id,
        team_label=team_label,
        n_nearby_players=n_nearby,
        crowded=crowded,
    )


def enrich_contacts(
    candidates: Iterable[ContactCandidate],
    provider: FrameProvider,
    detector,
    seed: TargetSeed,
    cfg: PipelineV2Config | None = None,
    mapper=None,
) -> list[PlayerContact]:
    """Enrich every candidate. On the default path (colour window off) batches
    nearby contact frames into a single model call when ``player_batch_size`` > 1.
    """
    cfg = cfg or PipelineV2Config()
    cands = list(candidates)
    if not cands:
        return []

    batch_size = max(1, int(getattr(cfg, "player_batch_size", 1) or 1))
    # Batching only applies to the single-frame default path — the 3-frame colour
    # window is the expensive opt-in path and stays sequential.
    can_batch = (
        not cfg.team_filter_enabled
        and batch_size > 1
        and hasattr(detector, "detect_many")
    )
    if not can_batch:
        return [enrich_contact(c, provider, detector, seed, cfg, mapper)
                for c in cands]

    # Decode every contact's centre frame first (provider already seeks cheaply),
    # then run detection in batches so Ultralytics/OpenVINO share one predict.
    frames_per: list[list[tuple[int, np.ndarray]]] = []
    detect_items: list[tuple[np.ndarray, tuple[float, float]]] = []
    for cand in cands:
        win = provider.window(cand.frame_index, 0, 1)
        frames_per.append(win)
        if win:
            detect_items.append((win[0][1], (cand.x, cand.y)))
        else:
            detect_items.append((np.zeros((8, 8, 3), dtype=np.uint8), (cand.x, cand.y)))

    all_dets: list[list[PlayerDetection]] = []
    for i in range(0, len(detect_items), batch_size):
        chunk = detect_items[i:i + batch_size]
        # Skip empty-window placeholders with a real call only when needed —
        # detect_many handles them; empty frames just yield no players.
        all_dets.extend(detector.detect_many(chunk))

    out: list[PlayerContact] = []
    for cand, frames, dets in zip(cands, frames_per, all_dets):
        if not frames:
            out.append(PlayerContact(
                candidate=cand, player_bbox=None, player_dist_px=None,
                jersey_hsv=None, color_dist=None, is_my_team=None,
                n_color_samples=0, team_label=TEAM_UNKNOWN,
            ))
            continue
        out.append(_enrich_from_detections(cand, frames, [dets], seed, cfg, mapper))
    return out


def filter_my_team(
    contacts: list[PlayerContact],
    *,
    keep_undecided: bool = True,
    keep_crowded: bool = True,
    enabled: bool = True,
) -> list[PlayerContact]:
    """Keep your-team contacts; drop opponents. Colour-undecided contacts are
    kept by default (recall over precision — the montage is the safety net), as
    are crowded ones (the colour belongs to whichever body won the nearest-player
    race, which in a pack is often not the toucher).

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
        elif keep_crowded and c.crowded and c.team_label not in (
            TEAM_OFFICIAL, TEAM_SIDELINE,
        ):
            out.append(c)
    return out


def contact_crops(contacts: list[PlayerContact]) -> list[np.ndarray | None]:
    """Torso crops captured during enrichment, for Stage 7 appearance scoring."""
    return [c.torso_crop for c in contacts]
