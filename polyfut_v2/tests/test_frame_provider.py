"""Tests for VideoFrameProvider's forward-cursor window access."""

import cv2
import numpy as np

from polyfut_v2.pipeline.frame_provider import VideoFrameProvider


def _clip(path, n=60, fps=12, w=64, h=48):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        vw.write(np.full((h, w, 3), i % 256, dtype=np.uint8))
    vw.release()


def _idx(win):
    return [i for i, _ in win]


def test_forward_and_backward_windows_return_requested_indices(tmp_path):
    clip = tmp_path / "c.mp4"
    _clip(clip)
    with VideoFrameProvider(str(clip), target_width=64) as p:
        assert _idx(p.window(10, 3, 1)) == [7, 8, 9, 10, 11, 12, 13]
        assert _idx(p.window(30, 2, 2)) == [28, 30, 32]     # forward, stepped
        assert _idx(p.window(5, 1, 1)) == [4, 5, 6]          # backward jump (rewind)
        assert _idx(p.window(40, 0, 1)) == [40]              # forward again
        assert p.window(40, 0, 1)[0][1].shape == (48, 64, 3)


def test_window_clamps_to_bounds(tmp_path):
    clip = tmp_path / "c.mp4"
    _clip(clip, n=20)
    with VideoFrameProvider(str(clip), target_width=64) as p:
        assert _idx(p.window(0, 3, 1)) == [0, 1, 2, 3]       # no negative indices
        assert max(_idx(p.window(19, 3, 1))) <= 19           # no past-end indices
