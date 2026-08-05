"""The update check must be right, quiet, and impossible to break the app with.

Nothing used to tell a user a newer PolyFut existed, so an install stayed on
whatever version it started life as and every later fix was invisible to the
people who already had the app.

The comparison is the part that rots silently: a string compare puts "1.10.0"
BELOW "1.9.0", so the prompt would stop appearing exactly at the tenth release
— when it starts mattering — and nobody would get an error to investigate.
"""

from __future__ import annotations

import json
import sys

import pytest

import server


# ------------------------------------------------------- version ordering
@pytest.mark.parametrize("latest,current", [
    ("1.0.1", "1.0.0"),
    ("1.1.0", "1.0.9"),
    ("2.0.0", "1.99.99"),
    ("1.10.0", "1.9.0"),      # the string-compare trap
    ("1.0.10", "1.0.9"),      # and again one level down
    ("10.0.0", "9.0.0"),
    ("1.2", "1.1.9"),
])
def test_newer_versions_are_detected(latest, current):
    assert server._is_newer(latest, current), f"{latest} should beat {current}"


@pytest.mark.parametrize("latest,current", [
    ("1.0.0", "1.0.0"),       # same
    ("1.0.0", "1.0.1"),       # older
    ("1.9.0", "1.10.0"),      # the trap, reversed
    ("0.9.9", "1.0.0"),
])
def test_same_or_older_is_not_an_update(latest, current):
    assert not server._is_newer(latest, current)


@pytest.mark.parametrize("latest,current", [
    ("", "1.0.0"),
    ("garbage", "1.0.0"),
    ("1.0.0", "dev"),         # running from source
    ("dev", "1.0.0"),
    (None, "1.0.0"),
])
def test_unparseable_versions_never_claim_an_update(latest, current):
    """A wrong "update available" is worse than none: it sends people to
    re-download something they already have."""
    assert not server._is_newer(latest, current)


def test_a_v_prefix_is_tolerated():
    assert server._is_newer("v1.0.1", "1.0.0")
    assert not server._is_newer("v1.0.0", "v1.0.0")


def test_prerelease_sorts_below_the_plain_release():
    assert server._is_newer("1.1.0", "1.1.0-beta.2") is False
    assert server._version_tuple("1.2.0-beta") == (1, 2, 0)


# ------------------------------------------------------------- the endpoint
@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "APP_VERSION", "1.0.0")
    server._update_cache["at"] = 0.0
    server._update_cache["payload"] = None
    return server.app.test_client()


def _remote(monkeypatch, doc=None, exc=None, body=None):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            if body is not None:
                return body
            return json.dumps(doc).encode()

    import urllib.request

    def fake(req, timeout=None):
        if exc:
            raise exc
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake)


def test_reports_an_available_update(client, monkeypatch):
    _remote(monkeypatch, {"version": "1.2.0", "release_url": "https://x.test"})
    d = client.get("/api/update_check").get_json()
    assert d["update_available"] is True
    assert d["latest"] == "1.2.0" and d["current"] == "1.0.0"
    assert d["url"] == "https://x.test"


def test_says_nothing_when_current(client, monkeypatch):
    _remote(monkeypatch, {"version": "1.0.0"})
    assert client.get("/api/update_check").get_json()["update_available"] is False


@pytest.mark.parametrize("exc", [
    OSError("no route to host"),
    TimeoutError("slow"),
    ValueError("bad json"),
    Exception("something else entirely"),
])
def test_a_failed_check_is_never_an_error(client, monkeypatch, exc):
    """Offline must look like "no update", not a broken app — this runs on the
    setup screen while the user is trying to start work."""
    _remote(monkeypatch, exc=exc)
    r = client.get("/api/update_check")
    assert r.status_code == 200
    d = r.get_json()
    assert d["update_available"] is False and d["checked"] is False


def test_malformed_remote_json_is_survived(client, monkeypatch):
    _remote(monkeypatch, body=b"<html>404 not found</html>")
    d = client.get("/api/update_check").get_json()
    assert d["update_available"] is False


def test_missing_version_field_is_survived(client, monkeypatch):
    _remote(monkeypatch, {"note": "no version here"})
    d = client.get("/api/update_check").get_json()
    assert d["update_available"] is False


# ------------------------------------------------------------ opt out / cache
def test_can_be_switched_off(client, monkeypatch):
    monkeypatch.setenv("POLYFUT_UPDATE_CHECK", "0")
    called = {"n": 0}

    import urllib.request

    def boom(req, timeout=None):
        called["n"] += 1
        raise AssertionError("should not have reached the network")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    d = client.get("/api/update_check").get_json()
    assert d["update_available"] is False and called["n"] == 0


def test_a_source_checkout_never_prompts(client, monkeypatch):
    """"dev" has no meaningful ordering against a release number."""
    monkeypatch.setattr(server, "APP_VERSION", "dev")
    d = client.get("/api/update_check").get_json()
    assert d["update_available"] is False and d["checked"] is False


def test_the_result_is_cached(client, monkeypatch):
    calls = {"n": 0}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"version": "1.5.0"}).encode()

    import urllib.request

    def counting(req, timeout=None):
        calls["n"] += 1
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", counting)
    for _ in range(4):
        assert client.get("/api/update_check").get_json()["latest"] == "1.5.0"
    assert calls["n"] == 1, "hit the network once per page load"


def test_the_frozen_build_knows_its_own_version():
    """packaging/VERSION is bundled by the spec; without it every install would
    report "dev" and could never be told about an update."""
    assert server._read_app_version()
    assert server.APP_VERSION == server._read_app_version()
