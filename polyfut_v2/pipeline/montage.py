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
        return {
            "rank": self.rank,
            "t_sec": round(self.scored.contact.candidate.t_sec, 3),
            "clip_start_sec": round(self.clip_start_sec, 3),
            "clip_end_sec": round(self.clip_end_sec, 3),
            "crop": [round(float(v), 1) for v in self.crop],
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "decision": self.decision,
            "kinds": list(self.scored.contact.candidate.kinds),
            "tracklet_id": self.scored.tracklet_id,
        }


def _status(conf: float, cfg: PipelineV2Config) -> str:
    if conf >= cfg.autoaccept_conf:
        return AUTO_ACCEPT
    if conf <= cfg.autohide_conf:
        return AUTO_HIDE
    return REVIEW


def _default_decision(status: str) -> str | None:
    # Extremes are pre-decided; review items wait for the human.
    return {AUTO_ACCEPT: ME, AUTO_HIDE: NOT_ME}.get(status)


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
        status = _status(s.confidence, cfg)
        items.append(MontageItem(
            scored=s, rank=rank,
            clip_start_sec=start, clip_end_sec=end,
            crop=crop, confidence=s.confidence,
            status=status, decision=_default_decision(status),
        ))

    # Cap the review queue to the top-N by confidence (items are already
    # confidence-descending). Excess review items are auto-hidden so the human
    # never faces hundreds of clips when appearance can't discriminate.
    if cfg.max_review and cfg.max_review > 0:
        reviewed = 0
        for it in items:
            if it.status != REVIEW:
                continue
            reviewed += 1
            if reviewed > cfg.max_review:
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
