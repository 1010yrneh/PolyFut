"""End-to-end test of the Stage 4-9 orchestration (assemble_touches) with fakes.

No real video or model: a synthetic trajectory with spaced turns drives Stage 4,
a fake provider serves solid-colour frames and a fake detector puts a player on
the ball, so the whole contacts → player+colour → scoring → montage → hotspots
chain runs deterministically.
"""

import numpy as np

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.orchestrator import assemble_touches
from polyfut_v2.pipeline.player_detector import PlayerDetection
from polyfut_v2.pipeline.trajectory import BallSample, BallTrajectory
from polyfut_v2.pipeline.seed import build_seed_from_torso_crops

RED = (0, 0, 200)


def _traj(n_turns=5, dt=0.25):
    """Straight runs (100 px/s) with a 90° turn every 8 samples → contacts."""
    sm = []
    x, y, vx, vy, i = 300.0, 180.0, 100 * dt, 0.0, 0
    for _seg in range(n_turns + 1):
        for _k in range(8):
            if not (30 < x < 610):
                vx = -vx
            if not (30 < y < 330):
                vy = -vy
            x += vx
            y += vy
            sm.append(BallSample(i, i * dt, i * dt, x, y, [x - 4, y - 4, x + 4, y + 4],
                                 0.9, True, False))
            i += 1
        vx, vy = -vy, vx  # turn
    return BallTrajectory(sm)


class FakeProvider:
    def _frame(self):
        return np.full((360, 640, 3), RED, dtype=np.uint8)

    def window(self, center_index, radius, step):
        step = max(1, step)
        return [(i, self._frame())
                for i in range(center_index - radius, center_index + radius + 1, step)]


class FakePlayerDetector:
    """Puts a player box on the ball, so every contact has a contacting player."""
    def detect(self, frame, near=None):
        if near is None:
            return []
        x, y = near
        return [PlayerDetection(bbox=[x - 20, y - 30, x + 20, y + 30], conf=0.9)]


def _seed():
    crop = np.full((40, 30, 3), RED, dtype=np.uint8)
    return build_seed_from_torso_crops([crop, crop, crop])


def test_pipeline_produces_candidates_and_montage():
    cfg = PipelineV2Config()
    res = assemble_touches(_traj(), 60.0, cfg, _seed(),
                           FakePlayerDetector(), FakeProvider())
    assert len(res["candidates"]) > 0
    # Red player matches the red seed → all kept as your-team.
    assert len(res["kept"]) == len(res["candidates"])
    assert all(k.is_my_team is True for k in res["kept"])
    # Montage is 1:1 with kept and ranked by descending confidence.
    m = res["montage"]
    assert len(m) == len(res["kept"])
    confs = [it.confidence for it in m]
    assert confs == sorted(confs, reverse=True)
    assert [it.rank for it in m] == list(range(len(m)))


def test_decisions_flow_through_to_hotspots():
    cfg = PipelineV2Config()
    prov, det, seed = FakeProvider(), FakePlayerDetector(), _seed()
    res = assemble_touches(_traj(), 60.0, cfg, seed, det, prov)
    # Simulate the user marking every montage item as "me".
    decisions = {it.rank: "me" for it in res["montage"]}
    res2 = assemble_touches(_traj(), 60.0, cfg, seed, det, prov, decisions=decisions)
    hs = res2["hotspots"]
    assert len(hs) > 0
    total = sum(len(h.contact_times) for h in hs)
    assert total == len(res2["montage"])  # every confirmed touch lands in a hotspot


def test_empty_trajectory_yields_empty_outputs():
    cfg = PipelineV2Config()
    res = assemble_touches(BallTrajectory(), 60.0, cfg, _seed(),
                           FakePlayerDetector(), FakeProvider())
    assert res["candidates"] == []
    assert res["montage"] == []
    assert res["hotspots"] == []
