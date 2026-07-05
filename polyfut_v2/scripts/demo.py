"""Try-it-out demo for the v2 pipeline.

Runs the full Stage 0-9 chain on a real video and prints a readable summary.

Because the default COCO ball detector cannot see a small soccer ball on amateur
footage (Stage 3 → empty trajectory), this demo injects a *synthetic* ball by
default so you can watch every downstream stage actually do something and produce
a populated montage. Pass ``--real-ball`` to use the real detector instead (and
see the honest empty-trajectory warning until a soccer ball model is plugged in).

Examples
--------
    python -m polyfut_v2.scripts.demo --video uploads/<clip>.mp4
    python -m polyfut_v2.scripts.demo --video <clip>.mp4 --confirm-top 8
    python -m polyfut_v2.scripts.demo --video <clip>.mp4 --real-ball
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.orchestrator import run_v2
from polyfut_v2.pipeline.ball_detector import BallDetection
from polyfut_video.pipeline.decode import probe_video  # v2 reuses v1 decode
from polyfut_v2.pipeline.frame_provider import VideoFrameProvider
from polyfut_v2.pipeline.hotspots import assemble_hotspots
from polyfut_v2.pipeline.player_detector import YoloPlayerDetector
from polyfut_v2.pipeline.seed import TargetSeed, build_seed_from_taps


class SyntheticBall:
    """A stand-in ball that patrols with periodic sharp turns → real contacts.
    Deterministic, so repeated runs are identical."""

    def __init__(self) -> None:
        self.i = 0
        self.x, self.y, self.heading = 320.0, 180.0, 0.3

    def detect(self, frame, last_center=None) -> BallDetection:
        if self.i % 7 == 0 and self.i > 0:      # sharp turn ~every 1.75 s
            self.heading += 1.9
        sp = 30.0                                # ~120 px/s
        self.x += sp * math.cos(self.heading)
        self.y += sp * math.sin(self.heading)
        if not (30 < self.x < 610):
            self.heading = math.pi - self.heading
            self.x = min(610.0, max(30.0, self.x))
        if not (30 < self.y < 330):
            self.heading = -self.heading
            self.y = min(330.0, max(30.0, self.y))
        self.i += 1
        return BallDetection(bbox=[self.x - 4, self.y - 4, self.x + 4, self.y + 4], conf=0.9)


def auto_seed(video: str, cfg: PipelineV2Config, pdet: YoloPlayerDetector) -> TargetSeed | None:
    """Pick the largest visible player in a mid-match frame as the 'target' and
    build a seed from it (stands in for the user's taps)."""
    info = probe_video(video)
    n = int(info.get("frame_count") or 0)
    with VideoFrameProvider(video, target_width=cfg.target_width) as prov:
        for idx in (n // 2, n // 3, 2 * n // 3, n // 4, n // 5):
            win = prov.window(idx, 0, 1)
            if not win:
                continue
            players = pdet.detect(win[0][1], None)
            if not players:
                continue
            big = max(players, key=lambda p: p.bbox[3] - p.bbox[1])
            center = ((big.bbox[0] + big.bbox[2]) / 2, (big.bbox[1] + big.bbox[3]) / 2)
            taps = [(w[0][1], center) for o in (-30, 0, 30)
                    if (w := prov.window(idx + o, 0, 1))]
            print(f"  seed: target ≈ player of height {big.bbox[3] - big.bbox[1]:.0f}px "
                  f"at frame {idx}")
            return build_seed_from_taps(
                taps, pdet,
                max_tap_dist_px=cfg.contact_max_player_dist_px,
                min_torso_px=cfg.color_min_torso_px,
            )
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="PolyFut v2 try-it-out demo")
    ap.add_argument("--video", required=True, help="Match video (e.g. uploads/<clip>.mp4)")
    ap.add_argument("--out", default="output_v2_demo")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--real-ball", action="store_true",
                    help="Use the real COCO ball detector (expect an empty trajectory)")
    ap.add_argument("--confirm-top", type=int, default=8,
                    help="Simulate the user confirming the top-N montage clips as 'me', "
                         "to demonstrate hotspot assembly")
    args = ap.parse_args()

    # Colour filter off by default: on wide footage torso colour can't separate
    # teams (see the debug notes), so we keep every contact and let ranking + the
    # (simulated) review carry it.
    cfg = PipelineV2Config(device=args.device, team_filter_enabled=False)
    pdet = YoloPlayerDetector(cfg)

    print(f"\nBuilding target seed from {args.video} …")
    seed = auto_seed(args.video, cfg, pdet)
    if seed is None:
        print("  (no player found for a seed — appearance scoring will be neutral)")

    ball = None if args.real_ball else SyntheticBall()
    print(f"\nRunning pipeline ({'REAL' if args.real_ball else 'SYNTHETIC'} ball)…")

    def prog(frac: float, msg: str) -> None:
        print(f"  [{frac * 100:5.1f}%] {msg}")

    meta = run_v2(args.video, args.out, cfg=cfg, seed=seed,
                  ball_detector=ball, player_detector=pdet, progress=prog)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  ball trajectory samples : {meta['n_samples']}")
    print(f"  ball detected ratio     : {meta['detected_ratio']}")
    print(f"  contact candidates      : {meta['n_candidates']}")
    print(f"  your-team contacts      : {meta['n_your_team']}")
    print(f"  awaiting review         : {meta['n_review']}")
    print(f"  hotspots (auto-accepted): {meta['n_hotspots']}")
    for w in meta.get("warnings", []):
        print(f"  [!] {w}")

    out = Path(args.out)
    montage = json.loads((out / "montage.json").read_text(encoding="utf-8"))
    items = montage["items"]
    if items:
        print("\n  Top montage clips (what the review UI would show):")
        print(f"    {'rank':>4} {'t_sec':>8} {'conf':>6}  {'status':<12} kinds")
        for it in items[:min(8, len(items))]:
            print(f"    {it['rank']:>4} {it['t_sec']:>8.2f} {it['confidence']:>6.2f}  "
                  f"{it['status']:<12} {','.join(it['kinds'])}")

    # Demonstrate Stage 9: pretend the user tapped 'me' on the top-N clips.
    if items and args.confirm_top > 0:
        chosen = items[: args.confirm_top]
        me_times = sorted(it["t_sec"] for it in chosen)
        hs = assemble_hotspots(me_times, cfg, duration_sec=meta_duration(out))
        print(f"\n  If you confirm the top {len(chosen)} clips as 'me' → {len(hs)} hotspot(s):")
        for h in hs:
            print(f"    {h.start_sec:7.2f}s – {h.end_sec:7.2f}s "
                  f"({h.duration_sec:.1f}s, {len(h.contact_times)} touch)")

    print(f"\n  Outputs written to: {out.resolve()}")
    print("    montage.json   – ranked review clips (Stage 8)")
    print("    hotspots.json  – confirmed hotspots (Stage 9)")
    print("    contacts.json  – raw kinematic candidates (Stage 4)\n")


def meta_duration(out_dir: Path) -> float | None:
    doc = json.loads((out_dir / "montage.json").read_text(encoding="utf-8"))
    return doc.get("duration_sec")


if __name__ == "__main__":
    main()
