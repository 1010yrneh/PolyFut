"""Only one thread may be inside a model at a time.

An OpenVINO CompiledModel has ONE default infer request, and Ultralytics'
predict() reuses it. Models are cached globally and shared, while the server
runs inference on several threads at once: the analysis job, the seed prefetch,
the kit preview and the model warm-up. Two of them inside one model raises

    RuntimeError: Infer Request is busy

which killed a real 20-minute run at the 9-minute mark, after 6,200 frames.

The nasty part is that it is a race, so it hides. The same code ran to
completion many times when the threads happened not to overlap - including a
94-minute run - which is exactly why this needs a test that forces the overlap
rather than hoping to observe it.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from polyfut_v2.config import PipelineV2Config
from polyfut_v2.pipeline import fast_infer
from polyfut_v2.pipeline.ball_detector import YoloBallDetector


class _ReentrancyDetector:
    """Fails loudly if two threads are inside predict() at once.

    This is what OpenVINO does, minus the crash: it is the same contract, made
    observable.
    """

    def __init__(self, dwell=0.02):
        self.inside = 0
        self.collisions = 0
        self.calls = 0
        self._dwell = dwell
        self._guard = threading.Lock()

    def predict(self, image, **kw):
        with self._guard:
            self.inside += 1
            self.calls += 1
            if self.inside > 1:
                self.collisions += 1
        # hold the "request" long enough that an unlocked caller must overlap
        threading.Event().wait(self._dwell)
        with self._guard:
            self.inside -= 1
        return []


def _cfg():
    cfg = PipelineV2Config()
    cfg.fast_infer_enabled = False        # exercise the predict() fallback
    cfg.harvest_players_from_ball_pass = False
    return cfg


def _hammer(target, n_threads=4, n_calls=5):
    errors = []

    def work():
        try:
            for _ in range(n_calls):
                target()
        except Exception as exc:            # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return errors


def test_concurrent_detect_never_overlaps_in_the_model():
    """Four threads, one shared model: the lock must serialise them."""
    model = _ReentrancyDetector()
    cfg = _cfg()
    det = YoloBallDetector(cfg, model=model)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    errors = _hammer(lambda: det.detect(frame, last_center=None))

    assert not errors, f"inference raised under concurrency: {errors[:2]}"
    # Not every detect() reaches the model: the fake finds nothing, so misses
    # accumulate and the miss-storm backoff starts skipping full scans. That is
    # the detector working as designed - what matters is that the calls which
    # DO reach it never overlap.
    assert model.calls > 0, "no inference happened at all - test is not exercising it"
    assert model.collisions == 0, (
        f"{model.collisions} overlapping entries into one model - this is the "
        f"'Infer Request is busy' crash")


def test_roi_and_full_models_are_locked_independently():
    """Two different models must not serialise against each other.

    The ROI pass runs a second, smaller model. Locking them together would be
    correct but would throw away the parallelism between unrelated models, so
    the lock is per model object.
    """
    a, b = _ReentrancyDetector(), _ReentrancyDetector()
    assert fast_infer.model_lock(a) is not fast_infer.model_lock(b)
    assert fast_infer.model_lock(a) is fast_infer.model_lock(a)


def test_lock_is_reentrant():
    """A caller holding the lock can take the fast path without deadlocking."""
    m = _ReentrancyDetector()
    lk = fast_infer.model_lock(m)
    with lk:
        acquired = lk.acquire(timeout=2)
        assert acquired, "model lock is not reentrant - a nested call deadlocks"
        lk.release()


def test_player_detector_is_locked_too():
    """The player pass shares the same model object as the ball pass."""
    from polyfut_v2.pipeline.player_detector import YoloPlayerDetector

    model = _ReentrancyDetector()
    cfg = _cfg()
    det = YoloPlayerDetector(cfg, model=model)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    errors = _hammer(lambda: det.detect(frame), n_threads=4, n_calls=4)

    assert not errors, f"player inference raised under concurrency: {errors[:2]}"
    assert model.collisions == 0, (
        f"{model.collisions} overlapping entries into the player model")
