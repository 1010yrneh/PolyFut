"""Stage 7 (appearance half): match a contact's player to the seed gallery.

Pluggable, like the detectors. The default is a cheap HSV colour histogram —
richer than the Stage 6 median hue (it captures multi-colour kits / patterns)
but still offline and model-free. A stronger embedding (ReID / SigLIP) or
jersey-number OCR drops in behind the same ``AppearanceModel`` protocol.

Gallery matching takes the *best* similarity over several reference crops
(spread across the match timeline), which the doc notes beats single-crop
matching on same-kit teammates and appearance drift.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np


class AppearanceModel(Protocol):
    def descriptor(self, crop: np.ndarray) -> np.ndarray | None: ...
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float: ...


class HistogramAppearance:
    """Default appearance model: normalized H-S histogram + correlation."""

    def __init__(self, h_bins: int = 12, s_bins: int = 4):
        self.h_bins = h_bins
        self.s_bins = s_bins

    def descriptor(self, crop: np.ndarray | None) -> np.ndarray | None:
        if crop is None or crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [self.h_bins, self.s_bins],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten().astype(np.float32)

    def similarity(self, a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return 0.0
        # Correlation ∈ [-1, 1]; clamp to [0, 1] so it reads as a similarity.
        corr = float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))
        return max(0.0, min(1.0, corr))

    def gallery_descriptors(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        out = []
        for c in crops:
            d = self.descriptor(c)
            if d is not None:
                out.append(d)
        return out

    def gallery_score(
        self, crop: np.ndarray | None, gallery: list[np.ndarray]
    ) -> float | None:
        """Best similarity of ``crop`` to any gallery descriptor, or None if the
        crop or gallery is empty (appearance unmeasurable)."""
        d = self.descriptor(crop)
        if d is None or not gallery:
            return None
        return max(self.similarity(d, g) for g in gallery)
