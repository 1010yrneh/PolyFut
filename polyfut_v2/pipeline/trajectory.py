"""Ball-trajectory data structures — the backbone every later stage hangs off.

A trajectory is a dense, time-ordered sequence of ball positions produced by the
continuous Stage 3 pass. Positions may be real detections or short interpolated
holds across YOLO misses; the ``interpolated`` / ``detected`` flags let later
stages (Stage 4 kinematics especially) reason about gap reliability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class BallSample:
    """One ball observation on one analysed frame.

    Coordinates are in the resized (``target_width``) frame space, consistent
    across the whole trajectory.
    """

    frame_index: int
    t_sec: float           # raw source timestamp
    processed_sec: float   # play-time with removed (dead/discard) segments collapsed
    x: float | None        # ball center x (None when no position is known)
    y: float | None        # ball center y
    bbox: list[float] | None  # [x1, y1, x2, y2]
    conf: float            # detection confidence (held detections are discounted)
    detected: bool         # True = real detection this frame
    interpolated: bool     # True = position carried/held from a previous frame

    def has_position(self) -> bool:
        return self.x is not None and self.y is not None

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "t_sec": round(self.t_sec, 4),
            "processed_sec": round(self.processed_sec, 4),
            "x": None if self.x is None else round(self.x, 2),
            "y": None if self.y is None else round(self.y, 2),
            "bbox": None if self.bbox is None else [round(float(v), 2) for v in self.bbox],
            "conf": round(float(self.conf), 4),
            "detected": bool(self.detected),
            "interpolated": bool(self.interpolated),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BallSample":
        return cls(
            frame_index=int(d["frame_index"]),
            t_sec=float(d["t_sec"]),
            processed_sec=float(d["processed_sec"]),
            x=None if d.get("x") is None else float(d["x"]),
            y=None if d.get("y") is None else float(d["y"]),
            bbox=None if d.get("bbox") is None else [float(v) for v in d["bbox"]],
            conf=float(d.get("conf", 0.0)),
            detected=bool(d.get("detected", False)),
            interpolated=bool(d.get("interpolated", False)),
        )


@dataclass
class BallTrajectory:
    """Ordered collection of :class:`BallSample` plus small summary helpers."""

    samples: list[BallSample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def add(self, sample: BallSample) -> None:
        self.samples.append(sample)

    def extend(self, samples: Iterable[BallSample]) -> None:
        self.samples.extend(samples)

    def positioned(self) -> list[BallSample]:
        """Samples that carry a known position (detected or held)."""
        return [s for s in self.samples if s.has_position()]

    def detected_ratio(self) -> float:
        """Fraction of samples backed by a real detection (recall proxy)."""
        if not self.samples:
            return 0.0
        return sum(1 for s in self.samples if s.detected) / len(self.samples)

    def to_dict(self) -> dict:
        n = len(self.samples)
        n_det = sum(1 for s in self.samples if s.detected)
        n_interp = sum(1 for s in self.samples if s.interpolated)
        n_missing = sum(1 for s in self.samples if not s.has_position())
        return {
            "count": n,
            "detected": n_det,
            "interpolated": n_interp,
            "missing": n_missing,
            "detected_ratio": round(self.detected_ratio(), 4),
            "samples": [s.to_dict() for s in self.samples],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BallTrajectory":
        return cls(samples=[BallSample.from_dict(s) for s in d.get("samples", [])])
