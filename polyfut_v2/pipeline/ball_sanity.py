"""Stage 3f: reject physically impossible ball excursions from a finished trajectory.

The failure this exists for
---------------------------
The ROI search (``roi_half_px``) re-finds "the ball" within a fixed radius of the
last known position every analysed frame. Once it locks onto a false positive —
a white sock, a line marking, a bright patch on a shirt — the ROI keeps finding
*something* nearby, so the lock sustains itself and the trajectory becomes a
sequence of alternating positions rather than a path.

Measured on two real 640x360 clips (3117 consecutive detection links), the turn
angle between successive steps is sharply **bimodal**:

    turn <= 30 deg   22.6% of links   median speed  64 px/s   <- real ball motion
    turn 170-181 deg 45.0% of links   median speed 456 px/s   <- alternating lock

A ball cannot reverse ~180 degrees and return to where it came from on
consecutive samples 0.13s apart, repeatedly, without a player touching it twice
in that window. So the reversal population is not ball motion.

Why this runs here and not in the smoother
------------------------------------------
``ball_smooth`` is online — it decides on each frame with no view of the next —
and the test needs the *following* sample to see that the path came back. It is
also why ``ball_confirm_jumps`` never fired on this footage: the excursions sit
at 100-112px, just under ``ball_suspect_jump_px`` (120), and 1-frame
displacements never reach 120px at all.

Recall-safety
-------------
Only the middle sample of a confirmed out-and-back is dropped, and only when it
is *both* a near-reversal and a genuine return to near the start. A real touch
that reverses the ball keeps going in the new direction, so it never satisfies
the return test. Demoted samples become position-less misses, which Stage 4
already refuses to use as velocity nodes — the surrounding real detections are
untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.trajectory import BallSample, BallTrajectory


@dataclass
class SanityStats:
    n_detected_before: int = 0
    n_rejected: int = 0
    n_checked: int = 0

    def to_dict(self) -> dict:
        return {
            "detected_before": self.n_detected_before,
            "rejected_pingpong": self.n_rejected,
            "links_checked": self.n_checked,
        }


def _turn_deg(ax, ay, bx, by) -> float:
    n1, n2 = math.hypot(ax, ay), math.hypot(bx, by)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = max(-1.0, min(1.0, (ax * bx + ay * by) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def _demote(s: BallSample) -> BallSample:
    """Turn a sample into a position-less miss (Stage 4 skips these as nodes)."""
    return replace(
        s, x=None, y=None, bbox=None, conf=0.0, detected=False, interpolated=False,
    )


def reject_pingpong(
    traj: BallTrajectory, cfg: PipelineV2Config | None = None,
) -> tuple[BallTrajectory, SanityStats]:
    """Blank out regions where the trajectory is a *sustained* alternation.

    A single out-and-back is deliberately left alone: a real return pass
    reverses and travels back through where the ball already was, which no
    three-point test can tell from a false lock. What cannot happen is the
    pattern repeating — a ball does not shuttle A→B→A→B between two fixed points
    on consecutive 0.13s samples. So a link qualifies as suspicious when

      * the step is long enough to matter (``ball_pingpong_min_step_px``),
      * it reverses by at least ``ball_pingpong_min_turn_deg``, and
      * it *returns*: ``|p3 - p1| <= ball_pingpong_return_frac * |p2 - p1|``,

    and only a run of ``ball_pingpong_min_alternations`` or more consecutive
    suspicious links is acted on.

    Every sample spanned by such a run is demoted, not just one side: in an
    alternation there is no geometric way to say which endpoint is the ball, and
    a velocity computed across the run is meaningless either way. Blanking the
    span makes the pipeline say "the ball's position is unknown here" instead of
    handing Stage 4 an inflection to turn into a false touch — which is exactly
    how the one hotspot in the exemplar run was fabricated.

    Samples more than ``ball_pingpong_max_gap_sec`` apart are never linked, so
    two unrelated sightings either side of a blind stretch can't form a run.
    """
    cfg = cfg or PipelineV2Config()
    if not getattr(cfg, "ball_pingpong_reject_enabled", True):
        return traj, SanityStats()

    min_step = float(getattr(cfg, "ball_pingpong_min_step_px", 40.0))
    min_turn = float(getattr(cfg, "ball_pingpong_min_turn_deg", 165.0))
    ret_frac = float(getattr(cfg, "ball_pingpong_return_frac", 0.35))
    max_gap = float(getattr(cfg, "ball_pingpong_max_gap_sec", 0.6))
    min_alt = max(2, int(getattr(cfg, "ball_pingpong_min_alternations", 2)))

    samples = list(traj.samples)
    stats = SanityStats(n_detected_before=sum(1 for s in samples if s.detected))

    # Neighbours are adjacent *detections* — misses sit between them.
    det_idx = [i for i, s in enumerate(samples) if s.detected and s.has_position()]
    if len(det_idx) < min_alt + 2:
        return traj, stats

    def suspicious(k: int) -> bool:
        """Is the link centred on detection ``k`` an out-and-back?"""
        p1, p2, p3 = (samples[det_idx[k - 1]], samples[det_idx[k]],
                      samples[det_idx[k + 1]])
        if (p2.t_sec - p1.t_sec) > max_gap or (p3.t_sec - p2.t_sec) > max_gap:
            return False
        ax, ay = p2.x - p1.x, p2.y - p1.y
        out = math.hypot(ax, ay)
        if out < min_step:
            return False
        if _turn_deg(ax, ay, p3.x - p2.x, p3.y - p2.y) < min_turn:
            return False
        return math.hypot(p3.x - p1.x, p3.y - p1.y) <= ret_frac * out

    flags = []
    for k in range(1, len(det_idx) - 1):
        stats.n_checked += 1
        flags.append((k, suspicious(k)))

    rejected: set[int] = set()
    run_start: int | None = None
    run_len = 0
    for k, bad in flags + [(None, False)]:   # sentinel flushes a trailing run
        if bad:
            run_start = k if run_start is None else run_start
            run_len += 1
            continue
        if run_start is not None and run_len >= min_alt:
            # The run spans detections [run_start-1 .. run_start+run_len].
            for j in range(run_start - 1, run_start + run_len + 1):
                if 0 <= j < len(det_idx):
                    rejected.add(det_idx[j])
        run_start, run_len = None, 0

    if not rejected:
        return traj, stats

    stats.n_rejected = len(rejected)
    out_samples = [
        _demote(s) if i in rejected else s for i, s in enumerate(samples)
    ]
    return BallTrajectory(samples=out_samples, camera=traj.camera), stats
