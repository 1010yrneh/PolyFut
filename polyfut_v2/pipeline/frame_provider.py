"""Random-access frame windows for the sparse stages.

Stage 5/6 need a few frames around each contact — a tiny fraction of the video —
so seeking on demand is far cheaper than holding decoded frames in memory.
"""

from __future__ import annotations

import cv2
import numpy as np

from polyfut_video.pipeline.decode import _resize_frame

# Forward gap beyond which a seek beats grabbing frame-by-frame (~one GOP).
_SEEK_AHEAD_FRAMES = 250


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
        self._pos = 0  # index of the next frame the capture will read

    def _capture(self) -> cv2.VideoCapture:
        if self._cap is None:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.video_path}")
            self._cap = cap
            self._pos = 0
        return self._cap

    def window(
        self, center_index: int, radius: int, step: int
    ) -> list[tuple[int, np.ndarray]]:
        cap = self._capture()
        step = max(1, step)
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        wanted = [i for i in range(center_index - radius, center_index + radius + 1, step)
                  if i >= 0 and (not n_total or i < n_total)]
        if not wanted:
            return []

        # Adaptive cursor: grab() forward (cheap, no full decode) for small gaps,
        # but SEEK for a backward jump or a large forward jump. Phone/broadcast
        # video has sparse keyframes, so a seek re-decodes a whole GOP (~one
        # GOP's worth of frames); grabbing is cheaper only up to about that many
        # frames. The sparse contact stage steps forward in small hops (grab
        # wins); the seed's few taps are spread across the whole match (seek
        # wins) — this keeps both fast.
        first, last = wanted[0], wanted[-1]
        if first < self._pos or first - self._pos > _SEEK_AHEAD_FRAMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            self._pos = first
        while self._pos < first:
            if not cap.grab():
                return []
            self._pos += 1

        want = set(wanted)
        out: list[tuple[int, np.ndarray]] = []
        while self._pos <= last:
            if self._pos in want:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame is not None:
                    out.append((self._pos, _resize_frame(frame, self.target_width)))
            elif not cap.grab():
                break
            self._pos += 1
        return out

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoFrameProvider":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
