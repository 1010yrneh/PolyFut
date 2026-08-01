"""Issue 7 — stage timings, descriptor reuse, and batched player inference."""

from __future__ import annotations

import numpy as np

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.orchestrator import assemble_touches
from polyfut_v2.pipeline.appearance import HistogramAppearance
from polyfut_v2.pipeline.contacts import ContactCandidate
from polyfut_v2.pipeline.grouping import assign_player_groups
from polyfut_v2.pipeline.player_contacts import PlayerContact, enrich_contacts
from polyfut_v2.pipeline.player_detector import PlayerDetection
from polyfut_v2.pipeline.scoring import score_contacts
from polyfut_v2.pipeline.seed import TargetSeed, build_seed_from_torso_crops
from polyfut_v2.pipeline.trajectory import BallSample, BallTrajectory
from polyfut_v2.tests.test_orchestrator import FakePlayerDetector, FakeProvider, _seed, _traj

RED = (0, 0, 200)


def test_assemble_touches_reports_stage_timings():
    res = assemble_touches(_traj(), 60.0, PipelineV2Config(), _seed(),
                           FakePlayerDetector(), FakeProvider())
    timings = res["timings"]
    for key in ("contacts", "player_enrich", "scoring", "montage"):
        assert key in timings
        assert timings[key] >= 0.0


def test_appearance_descriptor_cached_for_grouping_reuse():
    """Stage 7 writes appearance_descriptor; Stage 8 must use it (no recompute)."""
    crop = np.full((40, 30, 3), RED, dtype=np.uint8)
    seed = build_seed_from_torso_crops([crop, crop.copy()])
    cand = ContactCandidate(0, 0.0, 0.0, 100, 100, ["kick"], 0.8)
    contact = PlayerContact(
        candidate=cand, player_bbox=[90, 70, 110, 130], player_dist_px=0.0,
        jersey_hsv=[0.0, 200.0, 200.0], color_dist=5.0, is_my_team=True,
        n_color_samples=1, torso_crop=crop,
    )
    scored = score_contacts([contact], [crop], seed, PipelineV2Config())
    assert scored[0].contact.appearance_descriptor is not None

    class CountingApp(HistogramAppearance):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def descriptor(self, crop):
            self.calls += 1
            return super().descriptor(crop)

    # Monkey-patch: grouping builds its own HistogramAppearance. Verify the
    # cached path is taken by clearing torso_crop so a miss would yield no
    # descriptor — cache alone must still populate the appearance group.
    from types import SimpleNamespace
    cached = scored[0].contact.appearance_descriptor
    scored[0].contact.torso_crop = None
    item = SimpleNamespace(rank=0, scored=scored[0])
    g = assign_player_groups([item], seed)
    assert g[0]["appearance_group"] >= 0  # used the cache, not the missing crop
    assert scored[0].contact.appearance_descriptor is cached


class BatchCountingDetector:
    """Counts detect_many batches and total items; detect() is a fallback."""

    def __init__(self):
        self.batch_calls = 0
        self.items_seen = 0

    def detect(self, frame, near=None):
        if near is None:
            return []
        x, y = near
        return [PlayerDetection(bbox=[x - 20, y - 30, x + 20, y + 30], conf=0.9)]

    def detect_many(self, items):
        self.batch_calls += 1
        self.items_seen += len(items)
        out = []
        for frame, near in items:
            out.append(self.detect(frame, near))
        return out


def test_enrich_contacts_batches_on_default_path():
    """Default (filter off) packs nearby contacts into detect_many chunks."""
    cfg = PipelineV2Config(team_filter_enabled=False, player_batch_size=4)
    # Build a few kinematic-looking candidates directly.
    cands = [
        ContactCandidate(i * 10, float(i), float(i), 100.0 + i, 100.0, ["kick"], 0.8)
        for i in range(10)
    ]
    det = BatchCountingDetector()
    provider = FakeProvider()
    seed = _seed()
    contacts = enrich_contacts(cands, provider, det, seed, cfg)
    assert len(contacts) == 10
    # 10 items / batch 4 → 3 detect_many calls, not 10.
    assert det.batch_calls == 3
    assert det.items_seen == 10
    assert all(c.player_bbox is not None for c in contacts)


def test_enrich_contacts_stays_sequential_when_colour_window_on():
    """The 3-frame colour-window path must not silently switch to batching."""
    cfg = PipelineV2Config(team_filter_enabled=True, player_batch_size=8)

    class CountingDetect:
        def __init__(self):
            self.calls = 0
            self.batch_calls = 0

        def detect(self, frame, near=None):
            self.calls += 1
            x, y = near if near else (50, 50)
            return [PlayerDetection(bbox=[x - 20, y - 30, x + 20, y + 30], conf=0.9)]

        def detect_many(self, items):
            self.batch_calls += 1
            return [self.detect(f, n) for f, n in items]

    det = CountingDetect()
    cand = ContactCandidate(30, 3.0, 3.0, 50, 80, ["kick"], 0.8)
    enrich_contacts([cand], FakeProvider(), det, _seed(), cfg)
    assert det.batch_calls == 0
    assert det.calls == 3  # centre ± 1


def test_team_filter_still_off_by_default():
    assert PipelineV2Config().team_filter_enabled is False


def test_team_preview_has_no_debug_logger():
    import polyfut_video.pipeline.team_preview as tp
    assert not hasattr(tp, "_dbg_log")
    assert not hasattr(tp, "_DEBUG_LOG")
