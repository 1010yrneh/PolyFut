"""Stage 4: contact-candidate detection from ball kinematics.

Pure signal processing on a :class:`BallTrajectory` — no player information and
no model. A "contact" (someone touched the ball) shows up in the ball's motion
as one of three kinematic signatures:

  * **direction_change** — the heading swings sharply (a deflection / pass angle),
  * **stop** — the ball decelerates hard toward rest (a trap / control touch),
  * **kick** — the ball accelerates from slow/stopped to fast (a strike / pass).

The stage is tuned for high recall: it over-generates candidates cheaply and
lets later stages (and ultimately the human montage) supply precision.

Occlusion handling
------------------
The design doc's key risk (§8) is that occlusion gaps fake or hide inflections.
Two guards address it:

1. Velocity is computed only between **detected** samples, using their true time
   deltas. Interpolated (held) frames — which freeze the ball's position and
   would otherwise manufacture a zero-speed dip followed by a fake spike — are
   skipped as velocity nodes, so a displacement spanning a hold is averaged over
   the real elapsed time instead of appearing as an instantaneous jump.
2. Candidates whose supporting interval spans a large detection gap are flagged
   ``from_gap`` and down-weighted, so a shaky inflection can never masquerade as
   a strong contact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.trajectory import BallSample, BallTrajectory


@dataclass
class ContactCandidate:
    """A frame where the ball's motion suggests it was touched."""

    frame_index: int
    t_sec: float
    processed_sec: float
    x: float
    y: float
    kinds: list[str] = field(default_factory=list)  # {"direction_change","stop","kick"}
    strength: float = 0.0            # heuristic salience, ~[0, 1]
    speed_before: float = 0.0        # px/s into the contact
    speed_after: float = 0.0         # px/s out of the contact
    angle_change_deg: float = 0.0    # heading change at the contact
    from_gap: bool = False           # supported across a large detection gap (unreliable)

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "t_sec": round(self.t_sec, 4),
            "processed_sec": round(self.processed_sec, 4),
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "kinds": list(self.kinds),
            "strength": round(self.strength, 4),
            "speed_before": round(self.speed_before, 2),
            "speed_after": round(self.speed_after, 2),
            "angle_change_deg": round(self.angle_change_deg, 2),
            "from_gap": bool(self.from_gap),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ContactCandidate":
        return cls(
            frame_index=int(d["frame_index"]),
            t_sec=float(d["t_sec"]),
            processed_sec=float(d["processed_sec"]),
            x=float(d["x"]),
            y=float(d["y"]),
            kinds=list(d.get("kinds", [])),
            strength=float(d.get("strength", 0.0)),
            speed_before=float(d.get("speed_before", 0.0)),
            speed_after=float(d.get("speed_after", 0.0)),
            angle_change_deg=float(d.get("angle_change_deg", 0.0)),
            from_gap=bool(d.get("from_gap", False)),
        )


# --------------------------------------------------------------------------- #
# Pure kinematic helpers
# --------------------------------------------------------------------------- #

def _angle_between(vx1: float, vy1: float, vx2: float, vy2: float) -> float:
    """Angle (degrees, 0..180) between two 2-D vectors; 0 if either is null."""
    n1 = math.hypot(vx1, vy1)
    n2 = math.hypot(vx2, vy2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    cos = (vx1 * vx2 + vy1 * vy2) / (n1 * n2)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def split_tracks(samples: list[BallSample]) -> list[list[BallSample]]:
    """Split the trajectory into runs of positioned samples.

    A missing-position sample (ball lost entirely) breaks continuity; velocity
    is never computed across such a break.
    """
    tracks: list[list[BallSample]] = []
    cur: list[BallSample] = []
    for s in samples:
        if s.has_position():
            cur.append(s)
        elif cur:
            tracks.append(cur)
            cur = []
    if cur:
        tracks.append(cur)
    return tracks


def _smooth_positions(
    nodes: list[BallSample], window: int
) -> list[tuple[float, float]]:
    """Centered moving-average of node positions (window in samples)."""
    pts = [(float(s.x), float(s.y)) for s in nodes]
    if window <= 1 or len(pts) < 3:
        return pts
    half = window // 2
    out: list[tuple[float, float]] = []
    for i in range(len(pts)):
        lo = max(0, i - half)
        hi = min(len(pts), i + half + 1)
        seg = pts[lo:hi]
        out.append((sum(p[0] for p in seg) / len(seg),
                    sum(p[1] for p in seg) / len(seg)))
    return out


# --------------------------------------------------------------------------- #
# Stage 4 entry point
# --------------------------------------------------------------------------- #

def _classify_node(
    speed_before: float,
    speed_after: float,
    angle_deg: float,
    cfg: PipelineV2Config,
) -> tuple[list[str], float]:
    """Return (kinds, strength) for one interior node's kinematics."""
    kinds: list[str] = []
    subscores: list[float] = []
    fast_enough = max(speed_before, speed_after) >= cfg.contact_min_speed_px_s

    # Direction change: sharp heading swing while genuinely moving on both sides.
    if (
        angle_deg >= cfg.contact_dir_change_deg
        and min(speed_before, speed_after) >= cfg.contact_min_speed_px_s
    ):
        kinds.append("direction_change")
        subscores.append(min(1.0, angle_deg / 180.0))

    # Stop / trap: was moving, then decelerates hard toward rest.
    if (
        speed_before >= cfg.contact_min_speed_px_s
        and speed_after <= speed_before * cfg.contact_stop_ratio
    ):
        kinds.append("stop")
        subscores.append(min(1.0, 1.0 - speed_after / max(speed_before, 1e-6)))

    # Kick / strike: slow (or stopped), then accelerates hard.
    if (
        fast_enough
        and speed_after >= cfg.contact_min_speed_px_s
        and speed_after >= speed_before * cfg.contact_speed_spike_ratio
    ):
        kinds.append("kick")
        subscores.append(min(1.0, 1.0 - speed_before / max(speed_after, 1e-6)))

    strength = max(subscores) if subscores else 0.0
    return kinds, strength


def _raw_candidates(track: list[BallSample], cfg: PipelineV2Config) -> list[ContactCandidate]:
    nodes = [s for s in track if s.detected]
    if len(nodes) < 3:
        return []
    pts = _smooth_positions(nodes, cfg.contact_smooth_window)

    out: list[ContactCandidate] = []
    for i in range(1, len(nodes) - 1):
        dt_in = nodes[i].t_sec - nodes[i - 1].t_sec
        dt_out = nodes[i + 1].t_sec - nodes[i].t_sec
        if dt_in <= 0 or dt_out <= 0:
            continue
        vx_in = (pts[i][0] - pts[i - 1][0]) / dt_in
        vy_in = (pts[i][1] - pts[i - 1][1]) / dt_in
        vx_out = (pts[i + 1][0] - pts[i][0]) / dt_out
        vy_out = (pts[i + 1][1] - pts[i][1]) / dt_out
        speed_before = math.hypot(vx_in, vy_in)
        speed_after = math.hypot(vx_out, vy_out)
        angle = _angle_between(vx_in, vy_in, vx_out, vy_out)

        kinds, strength = _classify_node(speed_before, speed_after, angle, cfg)
        if not kinds:
            continue

        from_gap = max(dt_in, dt_out) > cfg.contact_gap_sec
        if from_gap:
            strength *= 0.5

        out.append(ContactCandidate(
            frame_index=nodes[i].frame_index,
            t_sec=nodes[i].t_sec,
            processed_sec=nodes[i].processed_sec,
            x=float(nodes[i].x), y=float(nodes[i].y),
            kinds=kinds, strength=strength,
            speed_before=speed_before, speed_after=speed_after,
            angle_change_deg=angle, from_gap=from_gap,
        ))
    return out


def _merge_candidates(
    cands: list[ContactCandidate], merge_sec: float
) -> list[ContactCandidate]:
    """Collapse candidates within ``merge_sec`` (one touch often trips several
    signatures on adjacent frames) into a single, strongest representative."""
    if not cands:
        return []
    cands = sorted(cands, key=lambda c: c.processed_sec)
    merged: list[ContactCandidate] = []
    cluster: list[ContactCandidate] = [cands[0]]

    def _flush(group: list[ContactCandidate]) -> ContactCandidate:
        rep = max(group, key=lambda c: c.strength)
        kinds: list[str] = []
        for c in group:
            for k in c.kinds:
                if k not in kinds:
                    kinds.append(k)
        rep.kinds = kinds
        rep.from_gap = any(c.from_gap for c in group)
        return rep

    for c in cands[1:]:
        if c.processed_sec - cluster[-1].processed_sec <= merge_sec:
            cluster.append(c)
        else:
            merged.append(_flush(cluster))
            cluster = [c]
    merged.append(_flush(cluster))
    return merged


def cap_candidates(
    cands: list[ContactCandidate],
    cfg: PipelineV2Config,
    duration_sec: float | None = None,
) -> list[ContactCandidate]:
    """Bound the candidate count (each survivor costs a Stage 5-6 player pass).

    The budget is the smaller of ``cfg.max_candidates`` and a duration-scaled
    cap (``max_candidates_per_min`` × minutes, floored at 60 so short clips are
    never starved). Selection is *time-fair*: candidates are bucketed into
    ``candidate_fair_window_sec`` windows and the strongest are taken
    round-robin across windows, so one noisy phase of the trajectory can't eat
    the whole budget and silence real touches elsewhere in the clip.
    """
    caps = []
    if cfg.max_candidates:
        caps.append(int(cfg.max_candidates))
    per_min = int(getattr(cfg, "max_candidates_per_min", 0) or 0)
    if per_min and duration_sec:
        caps.append(max(60, int(per_min * float(duration_sec) / 60.0)))
    if not caps:
        return cands
    k = min(caps)
    if len(cands) <= k:
        return cands

    window = float(getattr(cfg, "candidate_fair_window_sec", 30.0) or 30.0)
    buckets: dict[int, list[ContactCandidate]] = {}
    for c in cands:
        buckets.setdefault(int(c.processed_sec // window), []).append(c)
    for b in buckets.values():
        b.sort(key=lambda c: c.strength, reverse=True)

    picked: list[ContactCandidate] = []
    keys = sorted(buckets)
    rank = 0
    while len(picked) < k:
        took_any = False
        for key in keys:
            b = buckets[key]
            if rank < len(b):
                picked.append(b[rank])
                took_any = True
                if len(picked) >= k:
                    break
        if not took_any:
            break
        rank += 1
    picked.sort(key=lambda c: c.processed_sec)
    return picked


def detect_contacts(
    traj: BallTrajectory, cfg: PipelineV2Config | None = None
) -> list[ContactCandidate]:
    """Flag contact-candidate frames from ball kinematics alone.

    Returns time-ordered :class:`ContactCandidate` objects (merged).
    """
    cfg = cfg or PipelineV2Config()
    raw: list[ContactCandidate] = []
    for track in split_tracks(list(traj.samples)):
        raw.extend(_raw_candidates(track, cfg))
    return _merge_candidates(raw, cfg.contact_merge_sec)


def contacts_doc(candidates: list[ContactCandidate], source: dict | None = None) -> dict:
    """Serializable Stage 4 output with a small kind breakdown."""
    kinds_count: dict[str, int] = {}
    for c in candidates:
        for k in c.kinds:
            kinds_count[k] = kinds_count.get(k, 0) + 1
    return {
        "version": 1,
        "step": 2,
        "stage": 4,
        "source_video": (source or {}).get("source_video"),
        "count": len(candidates),
        "from_gap": sum(1 for c in candidates if c.from_gap),
        "kinds": kinds_count,
        "candidates": [c.to_dict() for c in candidates],
    }


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    p = argparse.ArgumentParser(
        description="PolyFut v2 — Step 2 (Stage 4): contact candidates from a trajectory"
    )
    p.add_argument("--trajectory", required=True,
                   help="ball_trajectory.json produced by Step 1")
    p.add_argument("--out", default=None, help="Output contacts.json (default: alongside input)")
    args = p.parse_args()

    doc = json.loads(Path(args.trajectory).read_text(encoding="utf-8"))
    traj = BallTrajectory.from_dict(doc.get("trajectory", doc))
    candidates = detect_contacts(traj, PipelineV2Config())

    out = Path(args.out) if args.out else Path(args.trajectory).with_name("contacts.json")
    result = contacts_doc(candidates, source=doc)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({
        "trajectory": args.trajectory,
        "contacts_path": str(out),
        "n_candidates": result["count"],
        "kinds": result["kinds"],
        "from_gap": result["from_gap"],
        "n_trajectory_samples": len(traj),
    }, indent=2))
    if len(traj) and traj.detected_ratio() == 0.0:
        print("\n[!] trajectory has zero detections — no contacts are derivable. "
              "Plug in a soccer ball model (see Step 1).")


if __name__ == "__main__":
    main()
