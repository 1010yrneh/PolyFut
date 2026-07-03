"""Tests for shot-level team crop accumulator."""

import numpy as np

from polyfut_video.pipeline.team_classify import TeamCropAccumulator


def _frame(color):
    img = np.zeros((120, 80, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def test_accumulator_clusters_across_chunks():
    acc = TeamCropAccumulator(max_crops_per_track=3)
    red = _frame((0, 0, 200))
    blue = _frame((200, 0, 0))

    acc.observe(red, [{"class": "player", "track_id": 1, "bbox": [10, 10, 30, 90]}])
    acc.observe(blue, [{"class": "player", "track_id": 2, "bbox": [40, 10, 60, 90]}])
    # second chunk — same track IDs, more crops
    acc.observe(red, [{"class": "player", "track_id": 1, "bbox": [12, 10, 32, 90]}])
    acc.observe(blue, [{"class": "player", "track_id": 2, "bbox": [42, 10, 62, 90]}])

    labels = acc.team_labels(min_samples=1, min_cluster_size=1, eps=25.0)
    assert labels[1] in (0, 1)
    assert labels[2] in (0, 1)
    assert labels[1] != labels[2]


def test_accumulator_resets():
    acc = TeamCropAccumulator(max_crops_per_track=2)
    acc.observe(_frame((0, 0, 200)), [
        {"class": "player", "track_id": 5, "bbox": [10, 10, 30, 90]},
    ])
    acc.reset()
    assert acc.team_labels() == {}
