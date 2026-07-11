"""Adaptive review grouping (Stage 8 UX).

Tags each montage touch so the review UI can act on more than one clip per click:

  * ``kit_group``       — colour cluster of the contacting player's shirt.
  * ``is_other_team``   — shirt clearly differs from your seed kit ⇒ the other
                          team (safe to hard-remove when you say "Not me").
  * ``appearance_group``— within *your* kit, a soft appearance cluster, so a
                          same-kit decision can propagate (reversibly) to
                          look-alikes.

Cheap post-process: reuses the torso crops already captured at contact time —
no extra decode, no model inference. All O(n^2) work is over ≤ a few hundred
tiny histograms, i.e. milliseconds.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline.appearance import HistogramAppearance
from polyfut_v2.pipeline.color import hsv_distance, hsv_feature


def _kit_hsv(item) -> np.ndarray | None:
    """Kit colour for a montage item: the jersey HSV sampled at contact if the
    colour filter recorded one, else derived from the torso crop."""
    c = item.scored.contact
    if c.jersey_hsv is not None:
        return np.asarray(c.jersey_hsv, dtype=np.float32)
    crop = c.torso_crop
    if crop is not None and getattr(crop, "size", 0) > 0:
        return hsv_feature(crop)
    return None


def _greedy_cluster(
    feats: list[tuple[int, np.ndarray]],
    dist: Callable[[np.ndarray, np.ndarray], float | None],
    thresh: float,
) -> dict[int, int]:
    """Order-independent-ish greedy clustering: assign each feature to the nearest
    existing centroid within ``thresh``, else start a new cluster. Centroids are
    running means. Returns ``{key: group_id}``."""
    centroids: list[np.ndarray] = []
    members: list[list[np.ndarray]] = []
    assign: dict[int, int] = {}
    for key, f in feats:
        best, best_d = -1, float("inf")
        for gi, cent in enumerate(centroids):
            d = dist(f, cent)
            if d is not None and d < best_d:
                best_d, best = d, gi
        if best >= 0 and best_d <= thresh:
            assign[key] = best
            members[best].append(f)
            centroids[best] = np.mean(np.stack(members[best]), axis=0)
        else:
            assign[key] = len(centroids)
            centroids.append(np.asarray(f, dtype=np.float32))
            members.append([np.asarray(f, dtype=np.float32)])
    return assign


def assign_player_groups(items, seed, cfg: PipelineV2Config | None = None) -> dict[int, dict]:
    """Return ``{rank: {kit_group, is_other_team, appearance_group}}`` for the
    montage ``items`` (MontageItem objects). Safe defaults when data is missing:
    a touch with no measurable colour stays same-team (``is_other_team`` False),
    and the colour layer disables entirely without a seed kit."""
    cfg = cfg or PipelineV2Config()
    app = HistogramAppearance()

    out: dict[int, dict] = {}
    kit_by_rank: dict[int, np.ndarray] = {}
    desc_by_rank: dict[int, np.ndarray] = {}
    for it in items:
        out[it.rank] = {"kit_group": -1, "is_other_team": False, "appearance_group": -1}
        kf = _kit_hsv(it)
        if kf is not None:
            kit_by_rank[it.rank] = kf
        crop = it.scored.contact.torso_crop
        d = app.descriptor(crop) if crop is not None else None
        if d is not None:
            desc_by_rank[it.rank] = d

    # Colour clusters over all measurable kits.
    kit_assign = _greedy_cluster(
        list(kit_by_rank.items()), hsv_distance, cfg.grouping_kit_cluster_dist)
    for rank, g in kit_assign.items():
        out[rank]["kit_group"] = int(g)

    # "Other team" only when we know your kit and this shirt is clearly far from
    # it — never from clustering alone (which can over-split your own team).
    have_seed_kit = seed is not None and getattr(seed, "kit_hsv", None) is not None
    if have_seed_kit:
        for rank, kf in kit_by_rank.items():
            dd = hsv_distance(kf, seed.kit_hsv)
            out[rank]["is_other_team"] = dd is not None and dd > cfg.grouping_other_team_dist

    # Appearance clusters within your team (soft same-kit propagation).
    same_kit = [(r, desc_by_rank[r]) for r in out
                if not out[r]["is_other_team"] and r in desc_by_rank]
    app_assign = _greedy_cluster(
        same_kit,
        lambda a, b: 1.0 - app.similarity(a, b),
        1.0 - cfg.grouping_appearance_min_sim,
    )
    for rank, g in app_assign.items():
        out[rank]["appearance_group"] = int(g)

    return out
