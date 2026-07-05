"""Random-access frame windows for the sparse stages.

Stage 5/6 need a few frames around each contact — a tiny fraction of the video —
so seeking on demand is far cheaper than holding decoded frames in memory.
"""

from __future__ import annotations

import cv2
import numpy as np

from polyfut_video.pipeline.decode import _resize_frame


class VideoFrameProvider:
    """Seek-based frame window over a video file.

    Frame indices match the *original* video indexing used by ``iter_frames``
    (and therefore by ``BallSample.frame_index``). Frames are resized to
    ``target_width`` for coordinate consistency with the trajectory.
    """

    def __init__(self, video_path: str, target_width: int = 640):
        self.video_path = str(video_path)
        self.target_width = target_width
        self._cap: cv2.VideoCapture | None = None

    def _capture(self) -> cv2.VideoCapture:
        if self._cap is None:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.video_path}")
            self._cap = cap
        return self._cap

    def window(
        self, center_index: int, radius: int, step: int
    ) -> list[tuple[int, np.ndarray]]:
        cap = self._capture()
        step = max(1, step)
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out: list[tuple[int, np.ndarray]] = []
        for idx in range(center_index - radius, center_index + radius + 1, step):
            if idx < 0 or (n_total and idx >= n_total):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            out.append((idx, _resize_frame(frame, self.target_width)))
        return out

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoFrameProvider":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
