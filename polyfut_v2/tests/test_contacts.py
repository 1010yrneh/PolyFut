"""Tests for Stage 4 kinematic contact-candidate detection."""

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.contacts import (
    ContactCandidate,
    contacts_doc,
    detect_contacts,
    split_tracks,
)
from polyfut_v2.pipeline.trajectory import BallSample, BallTrajectory


def _traj(rows):
    """rows: list of (t, x, y, detected). x=None → missing (ball lost);
    detected=False with a position → interpolated (held)."""
    samples = []
    for i, (t, x, y, detected) in enumerate(rows):
        if x is None:
            samples.append(BallSample(i, t, t, None, None, None, 0.0, False, False))
        else:
            samples.append(BallSample(
                i, t, t, float(x), float(y), [x - 1, y - 1, x + 1, y + 1],
                0.9 if detected else 0.4, detected, not detected,
            ))
    return BallTrajectory(samples)


# Geometric tests isolate the kinematic classifier on exact synthetic turns, so
# position smoothing is pinned off (the production default is 3, for noisy real
# detector output — see config).
CFG = PipelineV2Config(contact_smooth_window=1)


def test_straight_line_has_no_contacts():
    traj = _traj([(i * 0.1, i * 20, 0, True) for i in range(6)])
    assert detect_contacts(traj, CFG) == []


def test_direction_change_detected():
    traj = _traj([
        (0.0, 0, 0, True), (0.1, 20, 0, True), (0.2, 40, 0, True),
        (0.3, 40, 20, True), (0.4, 40, 40, True),
    ])
    cands = detect_contacts(traj, CFG)
    assert len(cands) == 1
    assert "direction_change" in cands[0].kinds
    assert cands[0].angle_change_deg > 80  # ~90° turn
    assert not cands[0].from_gap


def test_stop_detected():
    traj = _traj([
        (0.0, 0, 0, True), (0.1, 30, 0, True), (0.2, 60, 0, True),
        (0.3, 61, 0, True), (0.4, 61, 0, True),
    ])
    cands = detect_contacts(traj, CFG)
    assert len(cands) == 1
    assert "stop" in cands[0].kinds
    assert cands[0].speed_before > cands[0].speed_after


def test_kick_detected():
    traj = _traj([
        (0.0, 0, 0, True), (0.1, 0, 0, True), (0.2, 2, 0, True),
        (0.3, 40, 0, True), (0.4, 80, 0, True),
    ])
    cands = detect_contacts(traj, CFG)
    assert any("kick" in c.kinds for c in cands)
    kick = next(c for c in cands if "kick" in c.kinds)
    assert kick.speed_after > kick.speed_before


def test_missing_gap_splits_tracks_and_blocks_cross_gap_contact():
    # "right" segment, ball lost, then "up" segment. Without splitting this would
    # look like a sharp direction change at the seam — it must not.
    traj = _traj([
        (0.0, 0, 0, True), (0.1, 20, 0, True), (0.2, 40, 0, True),
        (0.3, None, None, False),                       # ball lost
        (0.4, 40, 20, True), (0.5, 40, 40, True), (0.6, 40, 60, True),
    ])
    tracks = split_tracks(traj.samples)
    assert len(tracks) == 2
    assert [len(t) for t in tracks] == [3, 3]
    # Each segment is straight → no contacts, and none manufactured at the seam.
    assert detect_contacts(traj, CFG) == []


def test_interpolated_frames_skipped_and_flagged_from_gap():
    # A long hold spans the touch; velocity is averaged over real elapsed time
    # (no fake spike) and the candidate is flagged unreliable.
    rows = [(0.0, 0, 0, True), (0.1, 20, 0, True)]
    rows += [(0.2 + 0.1 * k, 20, 0, False) for k in range(5)]  # held frames
    rows += [(0.7, 20, 60, True)]
    traj = _traj(rows)
    cands = detect_contacts(traj, CFG)
    assert len(cands) == 1
    assert cands[0].from_gap is True
    # from_gap halves strength: a ~90° turn would be 0.5, halved to ~0.25.
    assert cands[0].strength <= 0.25 + 1e-6


def test_adjacent_events_merge_with_union_of_kinds():
    traj = _traj([
        (0.0, 0, 0, True), (0.1, 20, 0, True), (0.2, 40, 0, True),   # turn at node2
        (0.3, 40, 20, True), (0.4, 40, 21, True), (0.5, 40, 21, True),  # stop at node3
    ])
    cands = detect_contacts(traj, CFG)
    assert len(cands) == 1  # two signatures 0.1s apart collapse into one
    assert set(cands[0].kinds) == {"direction_change", "stop"}


def test_candidate_roundtrip_and_doc_summary():
    c = ContactCandidate(
        frame_index=5, t_sec=0.5, processed_sec=0.5, x=40, y=20,
        kinds=["stop", "kick"], strength=0.8, speed_before=300, speed_after=10,
        angle_change_deg=12.0, from_gap=True,
    )
    c2 = ContactCandidate.from_dict(c.to_dict())
    assert c2.kinds == ["stop", "kick"] and c2.from_gap is True

    doc = contacts_doc([c], source={"source_video": "x.mp4"})
    assert doc["count"] == 1
    assert doc["from_gap"] == 1
    assert doc["kinds"] == {"stop": 1, "kick": 1}
    assert doc["source_video"] == "x.mp4"
