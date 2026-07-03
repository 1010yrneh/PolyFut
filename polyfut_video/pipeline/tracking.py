"""Stage 6: ByteTrack per shot — reset tracker at every shot boundary."""

from __future__ import annotations

import numpy as np

_TRACKER_CACHE: dict[str, object] = {}


def _make_bytetracker(track_thresh: float = 0.25, match_thresh: float = 0.8):
    try:
        from boxmot import BYTETracker
        return BYTETracker(track_thresh=track_thresh, match_thresh=match_thresh)
    except ImportError:
        pass
    try:
        from boxmot.trackers.bytetrack.bytetrack import BYTETracker
        return BYTETracker(track_thresh=track_thresh, match_thresh=match_thresh)
    except ImportError:
        pass
    # Minimal IoU fallback if boxmot unavailable (tests / lightweight env)
    return _IoUTracker()


class _IoUTracker:
    """Simple fallback tracker when boxmot is not installed."""

    def __init__(self, **kwargs):
        self._next_id = 1
        self._tracks: dict[int, np.ndarray] = {}

    def update(self, dets: np.ndarray, img: np.ndarray | None = None) -> np.ndarray:
        if dets is None or len(dets) == 0:
            return np.empty((0, 7))

        out_rows = []
        for row in dets:
            x1, y1, x2, y2, conf, cls = row[:6]
            best_id, best_iou = None, 0.1
            box = np.array([x1, y1, x2, y2], dtype=np.float32)
            for tid, prev in self._tracks.items():
                ii = _iou(box, prev)
                if ii > best_iou:
                    best_iou = ii
                    best_id = tid
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            self._tracks[best_id] = box
            out_rows.append([x1, y1, x2, y2, best_id, conf, cls])
        return np.array(out_rows, dtype=np.float32) if out_rows else np.empty((0, 7))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / (aa + bb - inter))


def _dets_to_array(detections: list[dict]) -> np.ndarray:
    rows = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cls_id = 0 if d["class"] == "player" else 32
        rows.append([x1, y1, x2, y2, d.get("conf", 0.5), cls_id])
    if not rows:
        return np.empty((0, 6))
    return np.array(rows, dtype=np.float32)


def track_shot(
    detections_per_frame: list[list[dict]],
    *,
    track_thresh: float = 0.25,
    match_thresh: float = 0.8,
    tracker: object | None = None,
) -> tuple[list[list[dict]], object]:
    """
    Input: per-frame detection lists for ONE shot only.
    Output: (tracked frames, tracker) — pass tracker back in for the next chunk.
    """
    if tracker is None:
        tracker = _make_bytetracker(track_thresh=track_thresh, match_thresh=match_thresh)
    tracked_frames: list[list[dict]] = []

    for frame_dets in detections_per_frame:
        arr = _dets_to_array(frame_dets)
        if len(arr) == 0:
            tracked_frames.append([])
            continue

        tracks = tracker.update(arr, img=None)
        out: list[dict] = []
        if tracks is None or len(tracks) == 0:
            tracked_frames.append([])
            continue

        # boxmot returns [x1,y1,x2,y2,track_id,conf,cls] or similar
        for i, det in enumerate(frame_dets):
            if i < len(tracks):
                tr = tracks[i]
                tid = int(tr[4]) if len(tr) > 4 else int(tr[4])
                det = {**det, "track_id": tid}
            out.append(det)
        # If counts mismatch, match by bbox proximity
        if len(out) != len(frame_dets):
            out = []
            used = set()
            for det in frame_dets:
                bx = np.array(det["bbox"], dtype=np.float32)
                best_j, best_iou = -1, 0.1
                for j, tr in enumerate(tracks):
                    if j in used:
                        continue
                    tb = np.array(tr[:4], dtype=np.float32)
                    ii = _iou(bx, tb)
                    if ii > best_iou:
                        best_iou = ii
                        best_j = j
                if best_j >= 0:
                    used.add(best_j)
                    det = {**det, "track_id": int(tracks[best_j][4])}
                else:
                    det = {**det, "track_id": -1}
                out.append(det)
        tracked_frames.append(out)

    return tracked_frames, tracker
