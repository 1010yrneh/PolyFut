"""version.json comes off the network, so treat it as untrusted input.

The update check GETs https://polyfut.com/version.json and the client renders
what comes back. That is a trust boundary, and it was not being treated as one.

The hole: _version_tuple stops at the first non-numeric part, so

    "999.0<img src=x onerror=...>"  ->  (999,)

which beats 1.0.1, so update_available was true - and the FULL string, tag and
all, was handed to the client, which built the notice with innerHTML. That is
script execution inside the app page, and the app page is same origin with the
local API: it could read /api/ai_config for the AI proxy token, or
/api/catalogue for the upload tokens that unlock the match videos.

release_url had no scheme check either, so "javascript:..." went straight into
an href.

Anyone able to serve that JSON - a compromised site or GitHub Pages account -
owns every installed copy on next launch. These tests pin the server half; the
client half was also rewritten to build DOM nodes instead of concatenating
markup, so neither layer alone is load-bearing.
"""

from __future__ import annotations

import json

import pytest

import server


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "APP_VERSION", "1.0.1", raising=False)
    monkeypatch.setattr(server, "_update_enabled", lambda: True)
    server._update_cache["at"] = 0.0          # never answer from a stale cache
    server._update_cache["payload"] = None
    return server.app.test_client()


def _serve(monkeypatch, doc):
    """Make the update check see exactly this JSON document."""
    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(doc).encode("utf-8")

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())


# ------------------------------------------------------------- the injection
@pytest.mark.parametrize("evil", [
    "999.0<img src=x onerror=alert(1)>",
    '999.0"><script>fetch("/api/ai_config")</script>',
    "999.0<svg/onload=alert(1)>",
    "<img src=x onerror=alert(1)>",
    "999.0' onmouseover='alert(1)",
])
def test_markup_in_the_version_is_refused(client, monkeypatch, evil):
    """It must not reach the client at all - not escaped, refused."""
    _serve(monkeypatch, {"version": evil, "release_url": "https://polyfut.com"})
    body = client.get("/api/update_check").get_json()
    assert body["update_available"] is False, f"{evil!r} was offered as an update"
    assert body["latest"] is None
    assert "<" not in json.dumps(body), "markup reached the client"


def test_the_exact_payload_that_beat_the_version_compare(client, monkeypatch):
    """Regression pin: this parsed as (999,) and so counted as newer."""
    evil = "999.0<img src=x onerror=alert(1)>"
    assert server._version_tuple(evil) == (999,)      # still parses that way
    assert server._is_newer(evil, "1.0.1") is True    # ...and still "newer"
    _serve(monkeypatch, {"version": evil})
    body = client.get("/api/update_check").get_json()
    assert body["update_available"] is False          # but is refused upstream


# --------------------------------------------------------------- the href
@pytest.mark.parametrize("bad_url", [
    "javascript:alert(document.domain)",
    "data:text/html,<script>alert(1)</script>",
    "vbscript:msgbox(1)",
    "file:///C:/Windows/System32/",
])
def test_dangerous_url_schemes_are_replaced(client, monkeypatch, bad_url):
    _serve(monkeypatch, {"version": "9.9.9", "release_url": bad_url})
    body = client.get("/api/update_check").get_json()
    assert body["url"] == "https://polyfut.com", f"{bad_url!r} survived"


def test_a_normal_https_url_is_kept(client, monkeypatch):
    _serve(monkeypatch, {"version": "9.9.9",
                         "release_url": "https://polyfut.com/releases"})
    body = client.get("/api/update_check").get_json()
    assert body["url"] == "https://polyfut.com/releases"


# ------------------------------------------------- the feature still works
@pytest.mark.parametrize("good", ["1.0.2", "v1.1.0", "2.0.0", "1.0.2-beta",
                                  "1.2.3.4", "1.0.2+build7"])
def test_real_versions_still_offer_an_update(client, monkeypatch, good):
    _serve(monkeypatch, {"version": good, "release_url": "https://polyfut.com"})
    body = client.get("/api/update_check").get_json()
    assert body["update_available"] is True, f"{good!r} should be offered"
    assert body["latest"] == good


def test_an_older_version_is_not_offered(client, monkeypatch):
    _serve(monkeypatch, {"version": "1.0.0", "release_url": "https://polyfut.com"})
    body = client.get("/api/update_check").get_json()
    assert body["update_available"] is False


def test_a_broken_check_is_silent_not_an_error(client, monkeypatch):
    """A failed update check must never look like a broken app."""
    import urllib.request

    def _boom(*a, **k):
        raise OSError("no network")
    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    r = client.get("/api/update_check")
    assert r.status_code == 200
    assert r.get_json()["update_available"] is False
