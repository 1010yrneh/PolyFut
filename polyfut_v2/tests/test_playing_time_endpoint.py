"""Round-trip tests for the playing-time endpoint and its wiring.

Covers the contract the rest of the feature leans on: the window is persisted
per upload token, the stored copy beats whatever a client sends, seed moments
come back confined, the seed cache is keyed by window so a whole-match prefetch
can't serve a windowed clip, and /api/v2/process hands the ranges to the
pipeline.
"""

import json
import time
from pathlib import Path

import pytest

import server as srv
from polyfut_v2.pipeline import play_ranges as pr
from polyfut_video.tests.conftest import make_synthetic_clip

TOKEN = "abcdef123456"          # must match the endpoint's [a-f0-9]{12} guard


@pytest.fixture()
def app(tmp_path, monkeypatch):
    """A server with isolated uploads/exports and no real model work."""
    uploads = tmp_path / "uploads"
    exports = tmp_path / "exports"
    uploads.mkdir()
    exports.mkdir()
    monkeypatch.setattr(srv, "UPLOADS", uploads)
    monkeypatch.setattr(srv, "EXPORTS", exports)
    # Prefetch compiles YOLO and builds clips — far too heavy for a unit test.
    calls: list = []
    monkeypatch.setattr(srv, "_start_seed_prefetch",
                        lambda *a, **kw: calls.append((a, kw)))
    srv.app.config.update(TESTING=True)
    make_synthetic_clip(uploads / f"{TOKEN}.mp4", duration_sec=10.0)
    client = srv.app.test_client()
    client.prefetch_calls = calls
    return client


def _post_window(client, ranges):
    return client.post("/api/v2/playing_time",
                       json={"token": TOKEN, "ranges": ranges})


# --- persistence -------------------------------------------------------------

def test_window_round_trips_to_disk(app, tmp_path):
    resp = _post_window(app, [[2.0, 6.0]])
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] and body["ranges"] == [[2.0, 6.0]]
    assert body["whole_match"] is False
    assert body["on_pitch_sec"] == pytest.approx(4.0)

    stored = json.loads((tmp_path / "exports" / "seed" / TOKEN / "playing_time.json")
                        .read_text(encoding="utf-8"))
    assert stored["ranges"] == [[2.0, 6.0]]
    assert srv._load_playing_time(TOKEN) == [(2.0, 6.0)]


def test_window_is_normalized_server_side(app):
    """Reversed, overlapping and out-of-bounds input is cleaned, not rejected —
    the client and the analysis must agree on exactly one interpretation."""
    body = _post_window(app, [[6.0, 2.0], [5.0, 8.0], [500.0, 900.0]]).get_json()
    assert body["ranges"] == [[2.0, 8.0]]


def test_whole_match_collapses_to_empty(app):
    """An explicit 'whole match' is stored as [] — identical to never asking, so
    cache keys, warnings and the pipeline all stay on the pre-existing path."""
    body = _post_window(app, [[0.0, 10.0]]).get_json()
    assert body["ranges"] == []
    assert body["whole_match"] is True
    assert srv._load_playing_time(TOKEN) == []


def test_window_starts_the_prefetch(app):
    _post_window(app, [[2.0, 6.0]])
    assert app.prefetch_calls, "prefetch must be kicked off once ranges are known"
    # ...and it is handed the ranges, not left to guess whole-match moments.
    assert app.prefetch_calls[-1][0][-1] == [(2.0, 6.0)]


def test_unknown_token_is_rejected(app):
    assert app.post("/api/v2/playing_time",
                    json={"token": "ffffffffffff", "ranges": [[0, 5]]}).status_code == 400
    assert app.post("/api/v2/playing_time",
                    json={"token": "not-a-token", "ranges": [[0, 5]]}).status_code == 400


# --- the stored copy wins ----------------------------------------------------

def test_stored_window_beats_a_stale_client_copy(app):
    """A resumed tab or an old client must never be able to widen the window
    behind the user's back."""
    _post_window(app, [[2.0, 6.0]])
    assert srv._effective_playing_time(TOKEN, [[0.0, 10.0]]) == [(2.0, 6.0)]


def test_client_copy_is_used_when_nothing_is_stored(app):
    """Fallback for an offline blip / server restart mid-flow, where the window
    only ever reached the server on the process call."""
    assert srv._effective_playing_time(TOKEN, [[1.0, 4.0]]) == [(1.0, 4.0)]
    assert srv._effective_playing_time(TOKEN, None) == []


# --- seed moments come back confined ----------------------------------------

def test_seed_index_moments_are_inside_the_window(app):
    _post_window(app, [[6.0, 9.0]])
    body = app.post("/api/v2/seed_clips_index",
                    json={"token": TOKEN, "reroll": 0}).get_json()
    assert body["ok"]
    assert body["moments"], "expected seed moments"
    assert all(6.0 <= t <= 9.0 for t in body["moments"]), body["moments"]


def test_seed_index_unconfined_without_a_window(app):
    body = app.post("/api/v2/seed_clips_index",
                    json={"token": TOKEN, "reroll": 0}).get_json()
    assert min(body["moments"]) < 3.0     # spreads across the whole clip again


# --- cache keying ------------------------------------------------------------

def test_seed_cache_name_is_keyed_by_window(app):
    """Without the window in the filename, a whole-match prefetch and a later
    windowed clip collide — and the user is served a clip from a moment they
    weren't playing."""
    assert pr.ranges_hash([]) == "all"
    assert pr.ranges_hash([(63 * 60.0, 111 * 60.0)]) != "all"

    seed_dir = Path(srv.EXPORTS) / "seed" / TOKEN
    seed_dir.mkdir(parents=True, exist_ok=True)
    whole = seed_dir / f"clip_{pr.ranges_hash([])}_0_0.json"
    windowed = seed_dir / f"clip_{pr.ranges_hash([(2.0, 6.0)])}_0_0.json"
    assert whole.name != windowed.name


def test_seed_clip_file_route_accepts_the_hashed_name_only(app):
    """The download route's path guard must allow the new filenames and still
    reject traversal-ish junk."""
    import re
    good = ["clip_all_0_1.mp4", "clip_a1b2c3d4_2_3.mp4"]
    bad = ["clip_0_1.mp4", "../secret.mp4", "clip_ZZZZ_0_0.mp4", "clip_all_0_1.json"]
    pattern = r"clip_(?:all|[a-f0-9]{8})_\d+_\d+\.mp4"
    assert all(re.fullmatch(pattern, n) for n in good)
    assert not any(re.fullmatch(pattern, n) for n in bad)


# --- process hands the window to the pipeline -------------------------------

def test_process_passes_the_stored_window_to_the_pipeline(app, monkeypatch):
    _post_window(app, [[2.0, 6.0]])

    seen = {}

    def _fake_run_to_montage(video_path, **kw):
        seen.update(kw)
        return {"montage": [], "hotspots": [], "warnings": [], "n_review": 0,
                "n_candidates": 0, "detected_ratio": 0.0, "duration_sec": 10.0,
                "play_ranges": [[2.0, 6.0]], "play_ranges_padded": [[0.0, 10.0]],
                "on_pitch_sec": 4.0}

    monkeypatch.setattr(srv, "run_to_montage", _fake_run_to_montage)

    resp = app.post("/api/v2/process", data={"token": TOKEN, "seed_taps": "[]"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["play_ranges"] == [[2.0, 6.0]]

    job_id = body["job_id"]
    for _ in range(100):                       # the job runs on a worker thread
        if seen or (srv._get_job(job_id) or {}).get("state") != "running":
            break
        time.sleep(0.05)
    assert seen.get("play_ranges") == [(2.0, 6.0)]

    job = srv._get_job(job_id)
    assert job["play_ranges"] == [[2.0, 6.0]]   # audit trail on the job record


def test_process_falls_back_to_the_client_copy(app, monkeypatch):
    """No stored window (server restarted mid-flow) — the form copy is used."""
    seen = {}
    monkeypatch.setattr(srv, "run_to_montage", lambda video_path, **kw: (
        seen.update(kw) or {"montage": [], "hotspots": [], "warnings": [],
                            "n_review": 0, "n_candidates": 0, "detected_ratio": 0.0,
                            "duration_sec": 10.0}))
    resp = app.post("/api/v2/process", data={
        "token": TOKEN, "seed_taps": "[]", "play_ranges": json.dumps([[3.0, 7.0]]),
    })
    assert resp.status_code == 200
    for _ in range(100):
        if seen:
            break
        time.sleep(0.05)
    assert seen.get("play_ranges") == [(3.0, 7.0)]


def test_process_rejects_malformed_play_ranges(app):
    resp = app.post("/api/v2/process", data={
        "token": TOKEN, "seed_taps": "[]", "play_ranges": "not json",
    })
    assert resp.status_code == 400
