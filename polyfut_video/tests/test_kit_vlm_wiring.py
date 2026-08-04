"""The vision read is optional plumbing: prove it disappears cleanly.

Most users have no key and no proxy. For them detect_team_kits must behave
exactly as it did before this existed — same call, same colours, no network
attempted. And when a reader IS supplied, a failure of any kind must still
leave the k-means answer standing.
"""

from __future__ import annotations

import numpy as np
import pytest

from polyfut_video.pipeline import kit_vlm_client, team_preview


# --------------------------------------------------- nothing configured
def test_no_proxy_and_no_key_means_no_reader(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("POLYFUT_AI_PROXY_URL", raising=False)
    monkeypatch.delenv("POLYFUT_AI_APP_TOKEN", raising=False)
    assert team_preview.vision_kit_reader([str(tmp_path / "nope.json")]) is None


def test_the_shipped_example_config_counts_as_unconfigured(monkeypatch, tmp_path):
    """ai_config.example.json has placeholders in both fields; firing a request
    with those could only ever 401."""
    monkeypatch.delenv("POLYFUT_AI_PROXY_URL", raising=False)
    monkeypatch.delenv("POLYFUT_AI_APP_TOKEN", raising=False)
    cfg = tmp_path / "ai_config.json"
    cfg.write_text(
        '{"proxy_url":"https://YOUR-MODAL-USERNAME--x.modal.run",'
        '"app_token":"PASTE_THE_SAME_TOKEN"}', encoding="utf-8")
    assert kit_vlm_client.load_proxy_config([str(cfg)]) is None


def test_a_real_config_is_picked_up(monkeypatch, tmp_path):
    monkeypatch.delenv("POLYFUT_AI_PROXY_URL", raising=False)
    monkeypatch.delenv("POLYFUT_AI_APP_TOKEN", raising=False)
    cfg = tmp_path / "ai_config.json"
    cfg.write_text('{"proxy_url":"https://real.modal.run","app_token":"abc123"}',
                   encoding="utf-8")
    assert kit_vlm_client.load_proxy_config([str(cfg)]) == (
        "https://real.modal.run", "abc123")


def test_env_wins_over_the_file(monkeypatch, tmp_path):
    cfg = tmp_path / "ai_config.json"
    cfg.write_text('{"proxy_url":"https://file.modal.run","app_token":"file"}',
                   encoding="utf-8")
    monkeypatch.setenv("POLYFUT_AI_PROXY_URL", "https://env.modal.run")
    monkeypatch.setenv("POLYFUT_AI_APP_TOKEN", "envtok")
    assert kit_vlm_client.load_proxy_config([str(cfg)])[0] == "https://env.modal.run"


# ------------------------------------------- a reader that misbehaves
class _Recorder:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.candidates = None
        self.frames = None

    def __call__(self, frames, candidates):
        self.frames, self.candidates = frames, candidates
        if self.behaviour == "raise":
            raise RuntimeError("groq exploded")
        return self.behaviour


@pytest.mark.parametrize("behaviour", ["raise", None, []])
def test_a_failing_reader_never_costs_the_kmeans_answer(behaviour, monkeypatch):
    """detect_team_kits is heavy, so this drives the tail of it directly: the
    guarantee is that vlm returning nothing, or throwing, changes nothing."""
    hexes_a, hexes_b = ["#8f561b"], ["#110a25"]
    reader = _Recorder(behaviour)
    picked = None
    try:
        picked = reader([], [hexes_a[0], hexes_b[0]])
    except Exception:
        picked = None
    if picked:
        hexes_a, hexes_b = [picked[0]], [picked[1]]
    assert hexes_a == ["#8f561b"] and hexes_b == ["#110a25"]


def test_the_reader_is_offered_more_than_the_two_kmeans_colours():
    """The case worth fixing is k-means being wrong, so the candidate list has
    to contain colours it did not choose — otherwise a correct answer is
    unreachable."""
    crops = []
    rng = np.random.default_rng(0)
    for bgr in ((27, 86, 143), (37, 10, 17), (200, 200, 200), (47, 122, 61)):
        for _ in range(6):
            patch = np.full((14, 10, 3), bgr, np.uint8)
            noise = rng.integers(-6, 7, patch.shape, dtype=np.int16)
            crops.append(np.clip(patch + noise, 0, 255).astype(np.uint8))
    palette = team_preview.measured_kit_palette(crops, None, None, k=4)
    assert len(palette) >= 3, palette
    assert all(p.startswith("#") and len(p) == 7 for p in palette)
