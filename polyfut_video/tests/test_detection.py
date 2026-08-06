"""Tests for Stage 5 detection."""

import numpy as np
import pytest

from polyfut_video.pipeline.detection import detect, DetectConfig, Detector


@pytest.mark.slow
def test_detect_on_synthetic_frame():
    """Requires YOLO weights download on first run."""
    # The one test in the suite that loads a real model. Everything else injects
    # a fake detector, which is why CI installs neither ultralytics nor torch
    # (~2GB). Skipping rather than failing keeps that choice cheap: the test
    # still runs wherever ultralytics is present, and CI stays green without it.
    pytest.importorskip("ultralytics")
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, :] = (35, 140, 35)
    det = Detector(DetectConfig(conf_threshold=0.5))
    dets = det.detect_frame(frame)
    assert isinstance(dets, list)
