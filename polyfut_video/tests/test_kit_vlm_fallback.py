"""When the shared quota runs out, fall back to the local pipeline — cheaply.

Catching the error per call is not enough. Groq's free tier is shared at the
ORGANISATION level, so once exhausted it stays exhausted, and without a circuit
breaker every later analysis would stall the team-picker screen for the full
45s timeout before giving up — and spend another slot of the proxy's own hourly
per-IP allowance doing it.

These pin that a refusal stops the asking, that the pause expires, and that the
kit read still returns k-means colours throughout.
"""

from __future__ import annotations

import urllib.error

import pytest

from polyfut_video.pipeline import kit_vlm_client


@pytest.fixture(autouse=True)
def _clean():
    kit_vlm_client.reset_block()
    yield
    kit_vlm_client.reset_block()


class _Boom(urllib.error.HTTPError):
    def __init__(self, code):
        super().__init__("http://x", code, "err", {}, None)

    def read(self):
        return b'{"error":"nope"}'


def _fail_with(monkeypatch, code):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise _Boom(code)

    monkeypatch.setattr(kit_vlm_client.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("POLYFUT_AI_PROXY_URL", "https://x.modal.run")
    monkeypatch.setenv("POLYFUT_AI_APP_TOKEN", "tok")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    return calls


# ------------------------------------------------------- quota exhausted
@pytest.mark.parametrize("code", [429, 413, 402])
def test_quota_refusal_stops_further_requests(monkeypatch, code):
    calls = _fail_with(monkeypatch, code)
    assert kit_vlm_client.ask([{"role": "user", "content": "x"}], "m") is None
    assert calls["n"] == 1
    # Every later run goes straight to k-means without touching the network.
    for _ in range(5):
        assert kit_vlm_client.ask([{"role": "user", "content": "x"}], "m") is None
    assert calls["n"] == 1, "kept calling a quota that had already refused"
    assert 0 < kit_vlm_client.blocked_for() <= kit_vlm_client.QUOTA_COOLDOWN_SEC


@pytest.mark.parametrize("code", [401, 403])
def test_rejected_credentials_pause_for_much_longer(monkeypatch, code):
    """A bad key does not come right on its own; a busy quota might."""
    _fail_with(monkeypatch, code)
    kit_vlm_client.ask([{"role": "user", "content": "x"}], "m")
    assert kit_vlm_client.blocked_for() > kit_vlm_client.QUOTA_COOLDOWN_SEC


def test_being_offline_also_pauses(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(kit_vlm_client.urllib.request, "urlopen", boom)
    monkeypatch.setenv("POLYFUT_AI_PROXY_URL", "https://x.modal.run")
    monkeypatch.setenv("POLYFUT_AI_APP_TOKEN", "tok")
    assert kit_vlm_client.ask([{"role": "user", "content": "x"}], "m") is None
    assert kit_vlm_client.blocked_for() > 0


# ------------------------------------------------------------- recovery
def test_the_pause_expires(monkeypatch):
    _fail_with(monkeypatch, 429)
    kit_vlm_client.ask([{"role": "user", "content": "x"}], "m")
    assert kit_vlm_client.blocked_for() > 0

    now = kit_vlm_client.time.time()
    monkeypatch.setattr(kit_vlm_client.time, "time",
                        lambda: now + kit_vlm_client.QUOTA_COOLDOWN_SEC + 1)
    assert kit_vlm_client.blocked_for() == 0


def test_a_pause_is_never_shortened_by_a_later_smaller_one(monkeypatch):
    """A 401's long pause must not be reset to 15 minutes by a later 429."""
    _fail_with(monkeypatch, 401)
    kit_vlm_client.ask([{"role": "user", "content": "x"}], "m")
    long_wait = kit_vlm_client.blocked_for()
    kit_vlm_client._block(kit_vlm_client.QUOTA_COOLDOWN_SEC, "later 429")
    assert kit_vlm_client.blocked_for() >= long_wait - 1


# ------------------------------------------- the kit read still works
def test_the_kit_read_returns_kmeans_colours_while_paused():
    """The contract that matters: a paused vision read is invisible."""
    from polyfut_video.pipeline import kit_vlm

    kit_vlm_client._block(kit_vlm_client.QUOTA_COOLDOWN_SEC, "test")
    reply = kit_vlm_client.ask([{"role": "user", "content": "x"}], "m")
    assert reply is None
    # ...and that is exactly what detect_team_kits treats as "keep k-means".
    assert kit_vlm.parse_choice(reply or "", ["#8f561b", "#110a25"]) is None
