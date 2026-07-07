"""Tests for the v2 server glue: decision→hotspot logic and the endpoints."""

import pytest

from polyfut_v2.app_service import hotspots_from_decisions
from polyfut_v2.config import PipelineV2Config


def _item(rank, t, decision=None, status="review"):
    return {"rank": rank, "t_sec": t, "decision": decision, "status": status}


def test_hotspots_from_decisions_applies_and_merges():
    items = [_item(0, 10.0), _item(1, 12.0, "me", "auto_accept"), _item(2, 50.0)]
    hs, updated = hotspots_from_decisions(
        items, {"0": "me"}, PipelineV2Config(), duration_sec=100.0)
    # rank 0 marked me; rank1 already me; both within gap → one merged hotspot.
    assert len(hs) == 1
    assert hs[0]["contact_times"] == [10.0, 12.0]
    assert updated[0]["decision"] == "me"


def test_hotspots_from_decisions_not_me_excluded():
    items = [_item(0, 10.0), _item(1, 40.0)]
    hs, _ = hotspots_from_decisions(
        items, {"0": "me", "1": "not_me"}, PipelineV2Config(), duration_sec=100.0)
    assert len(hs) == 1
    assert hs[0]["contact_times"] == [10.0]


def test_hotspots_from_decisions_ignores_bad_keys():
    items = [_item(0, 10.0)]
    hs, _ = hotspots_from_decisions(items, {"notanint": "me"}, PipelineV2Config())
    assert hs == []  # nothing confirmed


# --- endpoint tests via Flask test client ---

@pytest.fixture
def client():
    import server
    return server, server.app.test_client()


def test_validate_video_rejects_empty_accepts_real(client, tmp_path):
    server, _c = client
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    assert server._validate_video(empty)                 # 0 bytes → error string
    truncated = tmp_path / "part.mp4"
    truncated.write_bytes(b"x" * 500)
    assert server._validate_video(truncated)             # too small → error
    # A real, readable clip validates clean (noise so it exceeds the size floor).
    import cv2
    import numpy as np
    ok = tmp_path / "ok.mp4"
    vw = cv2.VideoWriter(str(ok), cv2.VideoWriter_fourcc(*"mp4v"), 12, (160, 120))
    rng = np.random.default_rng(0)
    for _ in range(40):
        vw.write(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    vw.release()
    assert ok.stat().st_size > 10_000
    assert server._validate_video(ok) is None


def test_teams_rejects_empty_upload(client):
    server, c = client
    import io
    r = c.post("/api/teams", data={"video": (io.BytesIO(b""), "x.mp4")},
               content_type="multipart/form-data")
    assert r.status_code == 400
    assert r.get_json().get("invalid_video") is True


def test_v2_process_requires_token(client):
    server, c = client
    if not server.PIPELINE_V2_OK:
        pytest.skip("v2 pipeline not importable")
    r = c.post("/api/v2/process", data={})
    assert r.status_code == 400


def test_v2_decisions_endpoint_returns_hotspots(client):
    server, c = client
    if not server.PIPELINE_V2_OK:
        pytest.skip("v2 pipeline not importable")
    jid = "testjobv2abc"
    server._set_job(jid, state="review", pipeline_version="v2", duration_sec=100.0,
                    montage=[_item(0, 10.0), _item(1, 12.0)])
    r = c.post(f"/api/v2/decisions/{jid}", json={"decisions": {"0": "me", "1": "me"}})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["n_hotspots"] == 1  # 10 & 12 merge
    # Status endpoint now surfaces the v2 fields.
    s = c.get(f"/api/process/status/{jid}").get_json()
    assert s["pipeline_version"] == "v2"
    assert s["hotspots"] and s["hotspots"][0]["contact_times"] == [10.0, 12.0]
