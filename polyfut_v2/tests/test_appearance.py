"""Tests for the Stage 7 appearance (gallery histogram) model."""

import numpy as np

from polyfut_v2.pipeline.appearance import HistogramAppearance

RED = (0, 0, 200)
BLUE = (200, 0, 0)


def _crop(color, h=40, w=30):
    return np.full((h, w, 3), color, dtype=np.uint8)


def test_descriptor_none_for_empty():
    ap = HistogramAppearance()
    assert ap.descriptor(None) is None
    assert ap.descriptor(np.zeros((0, 0, 3), np.uint8)) is None


def test_identical_crops_max_similarity():
    ap = HistogramAppearance()
    d = ap.descriptor(_crop(RED))
    assert ap.similarity(d, d) == 1.0


def test_different_kits_low_similarity():
    ap = HistogramAppearance()
    dr = ap.descriptor(_crop(RED))
    db = ap.descriptor(_crop(BLUE))
    assert ap.similarity(dr, db) < 0.3


def test_gallery_score_takes_best_match():
    ap = HistogramAppearance()
    gallery = ap.gallery_descriptors([_crop(BLUE), _crop(RED)])
    # A red probe matches the red gallery entry strongly.
    assert ap.gallery_score(_crop(RED), gallery) > 0.9
    # Empty gallery / missing crop → unmeasurable.
    assert ap.gallery_score(_crop(RED), []) is None
    assert ap.gallery_score(None, gallery) is None
