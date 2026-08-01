"""Stage 7: target confidence from appearance × orbital, scored sequentially.

Each surviving your-team contact gets a *confidence* (not a hard yes/no) used to
rank the review montage and to auto-accept / auto-hide the extremes.

Two complementary signals:

  * **appearance** — gallery match (colour histogram by default; swappable).
  * **orbital / motion continuity** — a growing search radius from the last
    high-confidence sighting, colour-locked like seed-clip tracking so identity
    can be *carried* between contacts instead of re-decided every touch.

On a well-anchored chain (contact firmly inside the orbital), motion **leads**:
weak / ambiguous appearance is floored up by ``motion_carry_floor`` rather than
dragging the product down. Outside the orbital, the classic
``appearance × prior`` product still applies. The prior is floored well above
zero so a shaky motion estimate can never hard-reject a real touch.

Safety rails:
  * anchor only on high-confidence appearance matches that are motion-consistent;
  * past ``orbital_max_gap_sec`` the prior goes neutral and the chain hard-breaks;
  * ``transform`` hook (identity by default) is where camera-motion compensation
    plugs in later (Issue 5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.appearance import AppearanceModel, HistogramAppearance
from polyfut_v2.pipeline.color import hsv_distance_multi
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
    motion_carried: bool = False     # confidence floored by motion continuity
    # True when this contact is linked to the seeded identity via orbital /
    # motion continuity (or a seed-sighting bootstrap). Required for auto-
    # accept into hotspots — high appearance alone is not enough.
    identity_linked: bool = False
    # Positive reasons to believe this is NOT the target, as opposed to merely
    # not knowing. Empty is the normal case and must stay indistinguishable from
    # "unmeasured" in every consumer that can hide a touch.
    negative_evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            **self.contact.to_dict(),
            "appearance_score": None if self.appearance_score is None
            else round(self.appearance_score, 4),
            "orbital_prior": round(self.orbital_prior, 4),
            "confidence": round(self.confidence, 4),
            "tracklet_id": self.tracklet_id,
            "anchored": self.anchored,
            "motion_carried": self.motion_carried,
            "identity_linked": self.identity_linked,
            "negative_evidence": list(self.negative_evidence),
        }


def negative_evidence_for(
    app: float | None,
    prior: float,
    *,
    had_anchor: bool,
    dt_anchor: float,
    is_my_team,
    cfg: PipelineV2Config,
) -> tuple[str, ...]:
    """Positive evidence that this contact is *not* the target.

    The pipeline has always been able to say "definitely you" and "I don't
    know", but never "definitely not you": ``appearance_default`` and
    ``orbital_floor`` are both 0.5, so the lowest reachable confidence is 0.25
    while ``autohide_conf`` is 0.15. Measured on a real run, 0 of 88 contacts
    were ever auto-hidden and the minimum confidence was exactly 0.250 — the
    product of the two floors. Every uncertain touch therefore drained to the
    human.

    The fix is not a lower threshold — that would hide *unmeasurable* touches,
    which is the one thing recall-safety forbids. It is to require a positive
    signal. Each reason below is something we measured, never something we
    failed to measure:

    * ``appearance_mismatch`` — appearance WAS read and was clearly wrong.
      ``app is None`` (unreadable crop, no gallery) never qualifies.
    * ``outside_orbital`` — a live anchor said you were demonstrably elsewhere
      at this moment, within the window where the orbital still means something.
    * ``other_team`` — the kit gate positively identified the opposing kit.
    """
    reasons: list[str] = []
    reject_max = float(getattr(cfg, "appearance_reject_max", 0.0))
    if reject_max > 0 and app is not None and app < reject_max:
        reasons.append("appearance_mismatch")
    if (had_anchor and prior <= cfg.orbital_floor + 1e-9
            and dt_anchor <= cfg.orbital_max_gap_sec):
        reasons.append("outside_orbital")
    if is_my_team is False:
        reasons.append("other_team")
    return tuple(reasons)


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


def kit_compatible(
    kit_a: list[float] | np.ndarray | None,
    kit_b: list[float] | np.ndarray | list[np.ndarray] | None,
    max_dist: float,
) -> bool:
    """Colour-lock for chain linking (seed-clip style).

    True when either kit is unreadable (don't break on a bad frame) or the
    hue-weighted distance is within ``max_dist``. False only when both are
    measurable and clearly different kits.

    ``kit_b`` may be a *set* of colours (a multi-coloured kit, or the seed's
    colour set): matching any one of them keeps the chain, because a red/blue
    player photographs as red on one touch and blue on the next.
    """
    if kit_a is None or kit_b is None:
        return True
    a = np.asarray(kit_a, dtype=np.float32)
    refs = kit_b if isinstance(kit_b, list) and kit_b and not np.isscalar(kit_b[0]) \
        else [kit_b]
    refs = [np.asarray(r, dtype=np.float32) for r in refs if r is not None]
    if not refs:
        return True
    d = hsv_distance_multi(a, refs)
    if d is None:
        return True
    return d <= max_dist


def combine_confidence(
    app: float | None,
    prior: float,
    cfg: PipelineV2Config,
    *,
    in_chain: bool,
) -> tuple[float, bool]:
    """Combine appearance and orbital into a confidence.

    When ``in_chain`` and the contact is firmly inside the orbital (prior==1),
    a weak/ambiguous appearance is motion-carried up to ``motion_carry_floor``.
    Otherwise appearance × prior (with appearance default). Never hard-rejects.
    """
    app_eff = app if app is not None else cfg.appearance_default
    product = max(0.0, min(1.0, app_eff * prior))
    carried = False
    # Motion leads only on an active chain, firmly inside the orbital, when
    # appearance is weak/unmeasured (cannot separate same-kit alone).
    weak_app = app is None or app < cfg.orbital_anchor_min
    if in_chain and prior >= 1.0 and weak_app:
        floor = max(0.0, min(1.0, cfg.motion_carry_floor * prior))
        if floor > product:
            product = floor
            carried = True
    return product, carried


def _bootstrap_anchor_from_seed(
    seed: TargetSeed,
    t: float,
    continuity_cap: float,
) -> tuple[float, float, float] | None:
    """Latest seed sighting at or before ``t`` within the continuity horizon.

    Returns (x, y, t) or None. Used so the orbital starts from who the user
    clicked, not from the first strong appearance match on a teammate.
    """
    best: tuple[float, float, float] | None = None
    for sx, sy, st in getattr(seed, "sightings", None) or []:
        if st > t:
            continue
        if (t - st) > continuity_cap:
            continue
        if best is None or st > best[2]:
            best = (float(sx), float(sy), float(st))
    return best


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

    Continuity (Issue 4): gaps larger than ``tracklet_max_gap_sec`` still keep
    the same tracklet when the contact stays inside the orbital, kit colour is
    compatible with the anchor (colour-locked rejoin), and the gap is within
    ``continuity_max_gap_sec``. Past that — or on a colour/position break — the
    chain hard-resets.

    ``crops[i]`` is the torso crop for ``contacts[i]`` (or None if unavailable).
    Returns scored contacts in ascending ``processed_sec``.
    """
    cfg = cfg or PipelineV2Config()
    appearance = appearance or HistogramAppearance()
    transform = transform or _identity

    gallery = appearance.gallery_descriptors(seed.gallery) if seed.gallery else []

    order = sorted(range(len(contacts)), key=lambda i: contacts[i].candidate.processed_sec)

    anchor: tuple[float, float, float] | None = None  # (x, y, t)
    # One colour (list[float], from a contact's jersey read) or a colour set
    # (list[np.ndarray], when bootstrapped from a multi-coloured seed kit).
    anchor_kit: list | None = None
    tracklet_id = -1
    last_t: float | None = None
    chain_active = False
    scored: list[ScoredContact] = []

    continuity_cap = getattr(cfg, "continuity_max_gap_sec", cfg.orbital_max_gap_sec)
    color_lock = getattr(cfg, "continuity_color_max_dist", cfg.team_color_max_dist)

    for i in order:
        c = contacts[i]
        t = c.candidate.processed_sec
        xy = transform((c.candidate.x, c.candidate.y), t)
        gap = None if last_t is None else (t - last_t)

        # Bootstrap the orbital from a seed sighting when we have no live anchor
        # yet (cold start, or after a continuity-horizon wipe).
        seed_bootstrapped = False
        if anchor is None:
            boot = _bootstrap_anchor_from_seed(seed, t, continuity_cap)
            if boot is not None:
                ax, ay, at = boot
                ax, ay = transform((ax, ay), at)
                anchor = (ax, ay, at)
                # Every colour your kit contains, so a chain bootstrapped from
                # the seed doesn't break when a multi-coloured kit reads as its
                # other colour on the next touch.
                seed_kits = seed.my_kits() if hasattr(seed, "my_kits") else (
                    [] if seed.kit_hsv is None else [seed.kit_hsv])
                if seed_kits:
                    anchor_kit = [np.asarray(k, dtype=np.float32) for k in seed_kits]
                seed_bootstrapped = True
                chain_active = True

        # Orbital prior against the current anchor (before this contact can
        # re-anchor), neutral if unanchored.
        had_prior_anchor = anchor is not None
        if anchor is None:
            prior = 1.0
            dt_anchor = 0.0
            dist = 0.0
        else:
            dt_anchor = t - anchor[2]
            dist = math.hypot(xy[0] - anchor[0], xy[1] - anchor[1])
            prior = orbital_prior(dist, dt_anchor, cfg)

        inside_orbital = anchor is not None and prior >= 1.0 and dt_anchor <= cfg.orbital_max_gap_sec
        colour_ok = kit_compatible(c.jersey_hsv, anchor_kit, color_lock)

        # Hard break: no previous contact, past continuity cap, or past orbital
        # with no usable motion signal. Soft continue: gap > tracklet_max but
        # still inside orbital + colour-locked (seed-clip rejoin pattern).
        hard_break = (
            last_t is None
            or gap is None
            or gap > continuity_cap
            or (gap > cfg.tracklet_max_gap_sec and not (inside_orbital and colour_ok))
            or (anchor is not None and dt_anchor > cfg.orbital_max_gap_sec)
        )
        # Seed bootstrap is not a "previous contact" — don't treat the first
        # contact after a seed sighting as a hard break just because last_t
        # is None.
        if seed_bootstrapped and last_t is None:
            hard_break = (
                (anchor is not None and dt_anchor > cfg.orbital_max_gap_sec)
                or not (inside_orbital and colour_ok)
            )
        if hard_break:
            tracklet_id += 1
            chain_active = False
            # Drop a stale anchor only past the continuity horizon — shorter
            # hard-breaks (e.g. teleport outside orbital) keep the prior so a
            # later near-anchor contact can still match the real player.
            if anchor is not None and gap is not None and gap > continuity_cap:
                anchor = None
                anchor_kit = None
                prior = 1.0
                inside_orbital = False
                had_prior_anchor = False
            elif seed_bootstrapped and not inside_orbital:
                # Seed said you were elsewhere — this contact is not linked.
                # Keep the seed anchor for a later near-you contact.
                chain_active = False
        # else: keep tracklet_id — continuity carry across the soft gap

        in_chain = (
            (chain_active or seed_bootstrapped)
            and not hard_break
            and inside_orbital
            and colour_ok
        )

        # One descriptor compute: used for gallery match AND cached on the
        # contact for Stage 8 grouping (Issue 7 — no second histogram pass).
        desc = None
        if i < len(crops) and crops[i] is not None:
            desc = appearance.descriptor(crops[i])
        c.appearance_descriptor = desc
        if desc is not None and gallery:
            app = max(appearance.similarity(desc, g) for g in gallery)
        else:
            app = None
        confidence, motion_carried = combine_confidence(
            app, prior, cfg, in_chain=in_chain,
        )

        # (Re)anchor only on a high-confidence, confirmed-team appearance match
        # that is ALSO motion-consistent with the current chain. With identical
        # kits, appearance alone can't stop a teammate touch from hijacking the
        # anchor, so a mid-chain contact must sit inside the orbital
        # (prior == 1.0) to (re)anchor. Bootstrapping a fresh chain (or the
        # very first anchor) is exempt — there is no chain to be consistent with.
        strong = (
            app is not None
            and app >= cfg.orbital_anchor_min
            and c.is_my_team is not False
        )
        # Re-anchor only inside the live orbital (or on a true cold start with
        # no prior). After a contact-chain hard break, a strong appearance may
        # start a new chain (not identity-linked until continuity confirms).
        # A seed bootstrap that says "you were over there" must not let a far
        # same-kit body steal the chain just because hard_break fired.
        motion_ok = (not had_prior_anchor) or prior >= 1.0
        if hard_break and last_t is not None and not seed_bootstrapped:
            motion_ok = True
        if seed_bootstrapped and not inside_orbital:
            motion_ok = False
        colour_anchor_ok = (not had_prior_anchor) or colour_ok or (
            hard_break and last_t is not None and not seed_bootstrapped
        )
        if seed_bootstrapped and not colour_ok:
            colour_anchor_ok = False
        anchored = strong and motion_ok and colour_anchor_ok
        if anchored:
            anchor = (xy[0], xy[1], t)
            anchor_kit = list(c.jersey_hsv) if c.jersey_hsv is not None else anchor_kit
            chain_active = True
        elif in_chain:
            # Stay on the chain without moving the orbital centre — a weak
            # same-kit teammate near the ball must not hijack the anchor.
            chain_active = True

        # Identity link: motion-carried, or (re)anchored while already inside a
        # live orbital / seed bootstrap. A cold-start strong appearance with no
        # seed sighting nearby is NOT linked — it goes to human review.
        identity_linked = bool(
            motion_carried
            or (anchored and had_prior_anchor and prior >= 1.0
                and dt_anchor <= cfg.orbital_max_gap_sec)
            or (in_chain and prior >= 1.0
                and app is not None and app >= cfg.orbital_anchor_min)
        )

        negatives = negative_evidence_for(
            app, prior,
            had_anchor=had_prior_anchor, dt_anchor=dt_anchor,
            is_my_team=c.is_my_team, cfg=cfg,
        )
        # An identity-linked contact is one motion continuity vouched for; a
        # single weak appearance read must not overturn that, or the same-kit
        # case (which is exactly what continuity exists to solve) starts losing
        # real touches.
        if identity_linked:
            negatives = tuple(r for r in negatives if r == "other_team")

        last_t = t
        scored.append(ScoredContact(
            contact=c,
            appearance_score=app,
            orbital_prior=prior,
            confidence=confidence,
            tracklet_id=tracklet_id,
            anchored=anchored,
            motion_carried=motion_carried,
            identity_linked=identity_linked,
            negative_evidence=negatives,
        ))

    return scored
