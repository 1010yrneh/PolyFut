"""Only this machine's own pages may talk to the API.

The server used to send ``Access-Control-Allow-Origin: *`` on every response.
Binding to 127.0.0.1 does not make that safe: a browser tab is a local program
running a remote site's code, so any page the user had open could reach the
API, and the wildcard made the replies readable.

The chain that made it more than theoretical:

    GET /api/catalogue      -> every analysis, including its upload "token"
    GET /api/video/<token>  -> the full match video for that token

which is a hostile page copying the user's footage off their machine - the
exact opposite of what polyfut.com promises. The video endpoint's 12-hex token
looks unguessable, and is, but it never had to be guessed: the catalogue hands
it out.

These tests pin the policy from both directions: a hostile origin gets nothing,
and the app's own origin keeps working.
"""

from __future__ import annotations

import pytest

import server


EVIL = "https://evil.example"
OURS = "http://127.0.0.1:5000"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("POLYFUT_ALLOW_FILE_ORIGIN", raising=False)
    return server.app.test_client()


# --------------------------------------------------------------- the policy
@pytest.mark.parametrize("origin", [
    "https://evil.example",
    "http://evil.example",
    "https://polyfut.com",          # our own SITE is still not our app
    "http://127.0.0.1.evil.com",    # suffix trick
    "http://localhost.evil.com",    # ditto
    "https://127.0.0.1@evil.com",   # userinfo trick
])
def test_foreign_origins_are_refused(origin):
    assert not server._origin_allowed(origin), f"{origin} must not be allowed"


@pytest.mark.parametrize("origin", [
    "http://127.0.0.1:5000",
    "http://127.0.0.1:54818",       # the app picks a random port
    "http://localhost:8137",
    "https://localhost:5000",
])
def test_local_origins_are_allowed(origin):
    assert server._origin_allowed(origin)


def test_no_origin_is_not_allowed_but_also_not_blocked():
    """Same-origin requests send no Origin. They need no CORS header, and the
    write guard must not reject them."""
    assert not server._origin_allowed("")


def test_file_origin_is_off_by_default(monkeypatch):
    """'null' is also what a sandboxed iframe on a hostile page sends."""
    monkeypatch.delenv("POLYFUT_ALLOW_FILE_ORIGIN", raising=False)
    assert not server._origin_allowed("null")
    monkeypatch.setenv("POLYFUT_ALLOW_FILE_ORIGIN", "1")
    assert server._origin_allowed("null")


# ------------------------------------------------------- reads are not leaked
def test_ai_config_is_not_readable_cross_origin(client):
    """This response carries the AI proxy token."""
    r = client.get("/api/ai_config", headers={"Origin": EVIL})
    assert "Access-Control-Allow-Origin" not in r.headers, (
        "a hostile page could read the AI proxy token")


def test_catalogue_is_not_readable_cross_origin(client):
    """The catalogue lists upload tokens, which unlock the videos."""
    r = client.get("/api/catalogue", headers={"Origin": EVIL})
    assert "Access-Control-Allow-Origin" not in r.headers


def test_the_video_exfiltration_chain_is_broken(client):
    """Both links of catalogue -> token -> video must be unreadable."""
    for path in ("/api/catalogue", "/api/video/0123456789ab"):
        r = client.get(path, headers={"Origin": EVIL})
        assert "Access-Control-Allow-Origin" not in r.headers, path


def test_our_own_page_still_works(client):
    r = client.get("/api/health", headers={"Origin": OURS})
    assert r.status_code == 200
    assert r.headers.get("Access-Control-Allow-Origin") == OURS


def test_same_origin_request_still_works(client):
    """No Origin header at all - the normal case in the packaged app."""
    r = client.get("/api/health")
    assert r.status_code == 200


def test_vary_origin_is_set(client):
    """Or a cache could hand one origin's response to another."""
    r = client.get("/api/health", headers={"Origin": OURS})
    assert "Origin" in r.headers.get("Vary", "")


# ------------------------------------------------------ writes are refused
def test_cross_site_delete_is_refused(client):
    r = client.delete("/api/catalogue/deadbeefcafe", headers={"Origin": EVIL})
    assert r.status_code == 403, "a hostile page could delete the user's analyses"


def test_cross_site_form_post_is_refused(client):
    """multipart/form-data is a "simple request": no preflight, so CORS alone
    would let this through and merely hide the reply."""
    r = client.post("/api/v2/process", data={"token": "0123456789ab"},
                    content_type="multipart/form-data", headers={"Origin": EVIL})
    assert r.status_code == 403


def test_cross_site_metadata_write_is_refused(client):
    r = client.post("/api/catalogue/deadbeefcafe/metadata",
                    json={"note": "owned"}, headers={"Origin": EVIL})
    assert r.status_code == 403


def test_same_origin_write_is_not_blocked(client):
    """The guard must not break the app's own writes. This 4xx/5xx comes from
    the endpoint's own validation, never from the origin guard."""
    r = client.post("/api/v2/process", data={}, content_type="multipart/form-data",
                    headers={"Origin": OURS})
    assert r.status_code != 403


def test_write_with_no_origin_is_not_blocked(client):
    r = client.post("/api/v2/process", data={}, content_type="multipart/form-data")
    assert r.status_code != 403
