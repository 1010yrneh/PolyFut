"""Stage 7: DBSCAN jersey colour clustering per shot."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

import cv2


def _torso_crop(frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    h = max(y2 - y1, 1)
    w = max(x2 - x1, 1)
    ty1 = int(y1 + 0.15 * h)
    ty2 = int(y1 + 0.55 * h)
    tx1 = int(x1 + 0.2 * w)
    tx2 = int(x2 - 0.2 * w)
    if ty2 <= ty1 or tx2 <= tx1:
        return None
    crop = frame[ty1:ty2, tx1:tx2]
    return crop if crop.size > 0 else None


def _hsv_feature(crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return np.median(hsv.reshape(-1, 3), axis=0).astype(np.float32)


def classify_teams(
    player_crops: list[np.ndarray],
    *,
    eps: float = 18.0,
    min_samples: int = 3,
    min_cluster_size: int = 4,
) -> list[int]:
    """
    Returns cluster label per player crop.
    Two largest clusters map to 0 (team_a) and 1 (team_b).
    Outliers (-1) and small clusters are -1 (unassigned).
    """
    if not player_crops:
        return []

    feats = np.stack([_hsv_feature(c) for c in player_crops], axis=0)
    # Weight hue more for clustering
    scaled = feats.copy()
    scaled[:, 0] *= 2.0

    if len(player_crops) < min_samples:
        return [-1] * len(player_crops)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)

    # Count cluster sizes (exclude noise -1)
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique) == 0:
        return [-1] * len(player_crops)

    order = np.argsort(-counts)
    top_clusters = unique[order[:2]]
    cluster_to_team = {int(top_clusters[0]): 0}
    if len(top_clusters) > 1:
        cluster_to_team[int(top_clusters[1])] = 1

    out: list[int] = []
    for lab in labels:
        if lab < 0:
            out.append(-1)
            continue
        cnt = int(counts[list(unique).index(lab)]) if lab in unique else 0
        if cnt < min_cluster_size:
            out.append(-1)
        elif lab in cluster_to_team:
            out.append(cluster_to_team[int(lab)])
        else:
            out.append(-1)
    return out


def assign_teams_to_tracked_frame(
    frame: np.ndarray,
    tracked_dets: list[dict],
    team_labels: dict[int, int],
) -> list[dict]:
    """Attach team_id (0=team_a, 1=team_b, -1=unassigned) to player detections."""
    out = []
    for det in tracked_dets:
        if det.get("class") != "player":
            out.append(det)
            continue
        tid = det.get("track_id", -1)
        team_id = team_labels.get(tid, -1)
        out.append({**det, "team_id": team_id})
    return out


def build_team_labels_for_shot(
    frames: list[np.ndarray],
    tracked_per_frame: list[list[dict]],
    *,
    eps: float = 18.0,
    min_samples: int = 3,
    min_cluster_size: int = 4,
) -> dict[int, int]:
    """
    Cluster all player torso crops in a shot; return track_id -> team_id (0/1/-1).
    """
    crops: list[np.ndarray] = []
    track_ids: list[int] = []
    seen: set[int] = set()

    for frame, dets in zip(frames, tracked_per_frame):
        for det in dets:
            if det.get("class") != "player":
                continue
            tid = int(det.get("track_id", -1))
            if tid < 0 or tid in seen:
                continue
            crop = _torso_crop(frame, det["bbox"])
            if crop is None:
                continue
            crops.append(crop)
            track_ids.append(tid)
            seen.add(tid)

    if not crops:
        return {}

    cluster_labels = classify_teams(
        crops, eps=eps, min_samples=min_samples, min_cluster_size=min_cluster_size,
    )
    return {tid: lab for tid, lab in zip(track_ids, cluster_labels)}


class TeamCropAccumulator:
    """Collect player torso crops across chunks within one broadcast shot."""

    def __init__(self, max_crops_per_track: int = 3):
        self._max = max(1, max_crops_per_track)
        self._crops: dict[int, list[np.ndarray]] = {}

    def reset(self) -> None:
        self._crops.clear()

    def observe(self, frame: np.ndarray, tracked_dets: list[dict]) -> None:
        for det in tracked_dets:
            if det.get("class") != "player":
                continue
            tid = int(det.get("track_id", -1))
            if tid < 0:
                continue
            bucket = self._crops.setdefault(tid, [])
            if len(bucket) >= self._max:
                continue
            crop = _torso_crop(frame, det["bbox"])
            if crop is not None:
                bucket.append(crop)

    def team_labels(
        self,
        *,
        eps: float = 18.0,
        min_samples: int = 3,
        min_cluster_size: int = 4,
    ) -> dict[int, int]:
        """Cluster using the latest crop per track (stable IDs across chunks)."""
        if not self._crops:
            return {}
        crops: list[np.ndarray] = []
        track_ids: list[int] = []
        for tid, clist in self._crops.items():
            if not clist:
                continue
            crops.append(clist[-1])
            track_ids.append(tid)
        cluster_labels = classify_teams(
            crops,
            eps=eps,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
        )
        return {tid: lab for tid, lab in zip(track_ids, cluster_labels)}
