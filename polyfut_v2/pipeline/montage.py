"""Stage 8: the human review montage — where high-recall CV becomes precision.

Surviving contacts become short clips (cropped/zoomed to the contact), ranked by
Stage 7 confidence. The clear extremes are auto-accepted / auto-hidden; the rest
the user resolves with a me / not-me tap. This module is the backend for that UI:
it builds the ranked manifest and applies the returned decisions. The actual clip
rendering and tapping live in the app.

Identity (you vs a same-kit teammate) is decided here, by the human, from motion
and context in a 1-2s clip — the one thing CV cannot reliably resolve on amateur
footage. That is the "never wasted" property: an over-generating run is corrected
by review rather than silently wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.scoring import ScoredContact

# Decision / status vocab
AUTO_ACCEPT = "auto_accept"
AUTO_HIDE = "auto_hide"
REVIEW = "review"
ME = "me"
NOT_ME = "not_me"


@dataclass
class MontageItem:
    scored: ScoredContact
    rank: int                 # 0 = highest confidence
    clip_start_sec: float     # source-video time
    clip_end_sec: float
    crop: list[float]         # [x1,y1,x2,y2] zoom box around the contact
    confidence: float
    status: str               # AUTO_ACCEPT | REVIEW | AUTO_HIDE
    decision: str | None      # ME | NOT_ME | None (awaiting the user)

    def to_dict(self) -> dict:
        cand = self.scored.contact.candidate
        return {
            "rank": self.rank,
            "t_sec": round(cand.t_sec, 3),
            # Source-frame index the contact (and player_bbox) was read from —
            # the review tracker must anchor on this, not ``round(t_sec * fps)``,
            # or the box can land on a neighbouring body a few frames away.
            "frame_index": int(cand.frame_index),
            "clip_start_sec": round(self.clip_start_sec, 3),
            "clip_end_sec": round(self.clip_end_sec, 3),
            "crop": [round(float(v), 1) for v in self.crop],
            # The player's actual detection box at the touch instant (distinct
            # from `crop`, which is a fixed-size zoom window centered on the
            # ball) — lets the review UI outline the player's body, not the
            # ball's neighbourhood. None when colour/detection couldn't place
            # them at this contact.
            "player_bbox": None if self.scored.contact.player_bbox is None
            else [round(float(v), 1) for v in self.scored.contact.player_bbox],
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "decision": self.decision,
            "kinds": list(cand.kinds),
            "tracklet_id": self.scored.tracklet_id,
            # Corner kicks / goalmouth scrambles: several bodies on the ball, so
            # the attribution behind this clip is unreliable and the human is
            # asked to confirm it regardless of confidence.
            "crowded": bool(getattr(self.scored.contact, "crowded", False)),
            "n_nearby_players": int(
                getattr(self.scored.contact, "n_nearby_players", 0) or 0
            ),
            # Continuity link to the seeded player — review UI only boxes when
            # True; auto-accept into hotspots also requires it.
            "identity_linked": bool(getattr(self.scored, "identity_linked", False)),
            # Why a contact was hidden, when it was — so a missing touch is
            # traceable to a reason rather than to a threshold.
            "negative_evidence": list(
                getattr(self.scored, "negative_evidence", ()) or ()),
        }


def _status(scored: ScoredContact | float, cfg: PipelineV2Config) -> str:
    """Auto verdict from confidence, gated by identity continuity.

    High appearance on an unlinked contact (far from the seeded orbital, or
    after a long neutral gap) stays in REVIEW so it cannot become a hotspot
    without an explicit "That's my touch".
    """
    negatives: tuple[str, ...] = ()
    if isinstance(scored, (int, float)):
        conf = float(scored)
        linked = True  # legacy call sites (crowded over-budget fallback)
    else:
        conf = scored.confidence
        linked = bool(getattr(scored, "identity_linked", False))
        negatives = tuple(getattr(scored, "negative_evidence", ()) or ())
        if not getattr(cfg, "autoaccept_require_identity", True):
            linked = True
    if conf >= cfg.autoaccept_conf and linked:
        return AUTO_ACCEPT
    if conf <= cfg.autohide_conf:
        return AUTO_HIDE
    # Evidence-based hide. The numeric threshold above is unreachable in
    # practice — appearance_default x orbital_floor floors confidence at 0.25
    # against an autohide of 0.15 — so without this the only way out of the
    # ambiguous band is a human. This fires only on a *measured* reason to
    # believe it is not you; an unmeasurable contact carries no reasons and
    # still goes to review.
    if negatives and getattr(cfg, "autohide_on_evidence", False):
        return AUTO_HIDE
    return REVIEW


def _default_decision(status: str) -> str | None:
    # Extremes are pre-decided; review items wait for the human.
    return {AUTO_ACCEPT: ME, AUTO_HIDE: NOT_ME}.get(status)


def is_crowded(item: MontageItem) -> bool:
    """Was this contact made in a pack of players (corner kick / scramble)?"""
    return bool(getattr(item.scored.contact, "crowded", False))


def build_montage(
    scored: list[ScoredContact],
    cfg: PipelineV2Config | None = None,
    *,
    duration_sec: float | None = None,
) -> list[MontageItem]:
    """Ranked montage (highest confidence first) with clip windows + auto status."""
    cfg = cfg or PipelineV2Config()
    ordered = sorted(scored, key=lambda s: s.confidence, reverse=True)

    items: list[MontageItem] = []
    for rank, s in enumerate(ordered):
        cand = s.contact.candidate
        t = cand.t_sec
        start = max(0.0, t - cfg.montage_clip_pad_sec)
        end = t + cfg.montage_clip_pad_sec
        if duration_sec is not None:
            end = min(end, duration_sec)
        half = cfg.montage_crop_half_px
        crop = [cand.x - half, cand.y - half, cand.x + half, cand.y + half]
        status = _status(s, cfg)
        # Crowded contacts always reach the human: in a corner-kick pack the
        # nearest-player attribution (and the kit colour taken from it) can
        # easily belong to the wrong body, so neither auto verdict is
        # trustworthy. The automatic verdict is kept as the pre-filled decision
        # so an unreviewed crowded touch still resolves the way confidence says
        # — the user gets the chance to overrule it, not the burden of having to.
        crowded = bool(getattr(s.contact, "crowded", False))
        decision = _default_decision(status)
        if crowded and getattr(cfg, "crowd_force_review", False):
            status = REVIEW
        items.append(MontageItem(
            scored=s, rank=rank,
            clip_start_sec=start, clip_end_sec=end,
            crop=crop, confidence=s.confidence,
            status=status, decision=decision,
        ))

    # Cap the review queue to the top-N by confidence (items are already
    # confidence-descending). Excess review items are auto-hidden so the human
    # never faces hundreds of clips when appearance can't discriminate. Crowded
    # touches draw on their own budget so a long ordinary queue can't bury the
    # very clips whose attribution we least trust.
    reviewed = 0
    crowded_reviewed = 0
    for it in items:
        if it.status != REVIEW:
            continue
        if is_crowded(it):
            crowded_reviewed += 1
            cap = int(getattr(cfg, "crowd_max_review", 0) or 0)
            if cap > 0 and crowded_reviewed > cap:
                # Past the budget, fall back to what confidence alone said
                # rather than blanket-hiding — an over-cap crowded touch that
                # would have auto-accepted keeps that verdict.
                it.status = _status(it.scored, cfg)
                if it.status == REVIEW:
                    it.status = AUTO_HIDE
                    it.decision = NOT_ME
                else:
                    it.decision = _default_decision(it.status)
            continue
        reviewed += 1
        if cfg.max_review and cfg.max_review > 0 and reviewed > cfg.max_review:
            it.status = AUTO_HIDE
            it.decision = NOT_ME
    return items


def review_queue(items: list[MontageItem]) -> list[MontageItem]:
    """Items the user still needs to judge (in rank order)."""
    return [it for it in items if it.status == REVIEW]


def apply_decisions(items: list[MontageItem], decisions: dict[int, str]) -> None:
    """Apply user me/not-me decisions in place, keyed by montage ``rank``."""
    for it in items:
        if it.rank in decisions:
            d = decisions[it.rank]
            if d not in (ME, NOT_ME):
                raise ValueError(f"decision must be '{ME}' or '{NOT_ME}', got {d!r}")
            it.decision = d


def confirmed_me_times(items: list[MontageItem]) -> list[float]:
    """Source-video times of contacts confirmed as the target ('me'), sorted."""
    return sorted(it.scored.contact.candidate.t_sec
                  for it in items if it.decision == ME)
