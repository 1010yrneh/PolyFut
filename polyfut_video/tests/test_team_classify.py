"""Tests for Stage 7 team classification."""

import numpy as np

from polyfut_video.pipeline.team_classify import classify_teams


def test_classify_two_color_clusters():
    red = np.full((20, 20, 3), (0, 0, 200), dtype=np.uint8)
    white = np.full((20, 20, 3), (230, 230, 230), dtype=np.uint8)
    crops = [red] * 5 + [white] * 5
    labels = classify_teams(crops, eps=25.0, min_samples=2, min_cluster_size=3)
    assert len(labels) == 10
    red_labels = set(labels[:5])
    white_labels = set(labels[5:])
    assert len(red_labels) >= 1
    assert len(white_labels) >= 1
