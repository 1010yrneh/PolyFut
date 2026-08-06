"""Tests for the AI proxy's request validation and quota guard.

The endpoint itself can't be exercised without deploying, but the parts that
protect the shared Groq free tier — message validation and the per-IP rate
limit — are pure and worth pinning: they are what stops one extracted app token
from draining the whole organisation's daily quota.

``modal`` is a deploy-time dependency, not an app runtime one, so it isn't in
requirements.txt and isn't installed in the test environment. A minimal stub
lets the module import so the real logic underneath can be tested.
"""

from __future__ import annotations

import sys
import types

import pytest


def _install_fastapi_stub() -> None:
    """report_backend.py imports fastapi.Request at module level (the documented
    way to read headers/IP in a Modal fastapi_endpoint). fastapi is a real
    runtime dependency — installed in the Modal image and needed on any machine
    that runs `modal serve`/`modal deploy` — but isn't installed in this test
    sandbox, so a placeholder lets the module import for the parts of it that
    don't need a live HTTP framework."""
    if "fastapi" in sys.modules:
        return
    fastapi = types.ModuleType("fastapi")

    class _Request:
        pass

    fastapi.Request = _Request
    sys.modules["fastapi"] = fastapi

    responses = types.ModuleType("fastapi.responses")

    class _JSONResponse:
        def __init__(self, status_code=200, content=None):
            self.status_code = status_code
            self.content = content

    responses.JSONResponse = _JSONResponse
    sys.modules["fastapi.responses"] = responses


def _install_modal_stub() -> None:
    """Just enough of Modal's surface for report_backend.py to import."""
    if "modal" in sys.modules:
        return
    modal = types.ModuleType("modal")

    class _Image:
        @staticmethod
        def debian_slim(*a, **kw):
            return _Image()

        def pip_install(self, *a, **kw):
            return self

    class _App:
        def __init__(self, *a, **kw):
            pass

        def function(self, *a, **kw):
            return lambda fn: fn

    class _Secret:
        @staticmethod
        def from_name(*a, **kw):
            return object()

    class _Dict(dict):
        @classmethod
        def from_name(cls, *a, **kw):
            return cls()

    modal.App = _App
    modal.Image = _Image
    modal.Secret = _Secret
    modal.Dict = _Dict
    modal.fastapi_endpoint = lambda *a, **kw: (lambda fn: fn)
    modal.web_endpoint = modal.fastapi_endpoint
    sys.modules["modal"] = modal


_install_fastapi_stub()
_install_modal_stub()

from ai_backend import report_backend as rb  # noqa: E402


# --- message validation ------------------------------------------------------

def test_valid_history_passes_through_unchanged():
    msgs = [
        {"role": "system", "content": "You are a scout."},
        {"role": "user", "content": "Report please"},
        {"role": "assistant", "content": "Here it is"},
    ]
    out, err = rb._validate_messages(msgs)
    assert err is None
    assert out == msgs


def test_extra_keys_are_stripped():
    """Only role/content reach Groq — a client can't smuggle other fields into
    the upstream request."""
    out, err = rb._validate_messages(
        [{"role": "user", "content": "hi", "name": "x", "tool_calls": [1]}])
    assert err is None
    assert out == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("bad", [
    None, [], "not a list", [[]], [{"role": "user"}],
    [{"role": "root", "content": "hi"}],
    [{"role": "user", "content": ""}],
    [{"role": "user", "content": "   "}],
    [{"role": "user", "content": 42}],
])
def test_malformed_histories_are_rejected(bad):
    out, err = rb._validate_messages(bad)
    assert out is None and err


def test_message_count_is_capped():
    msgs = [{"role": "user", "content": "x"}] * (rb.MAX_MESSAGES + 1)
    out, err = rb._validate_messages(msgs)
    assert out is None
    assert "too many messages" in err


def test_total_size_is_capped():
    """A long match timeline is the realistic oversized input, and the shared
    per-minute token budget is what it would blow."""
    msgs = [{"role": "user", "content": "x" * (rb.MAX_TOTAL_CHARS // 2 + 10)}] * 2
    out, err = rb._validate_messages(msgs)
    assert out is None
    assert "too long" in err


def test_a_realistic_report_request_fits_comfortably():
    """Guard against setting the caps so tight that normal usage trips them."""
    timeline = ", ".join(f"[{m}:00] Pass Completed" for m in range(90))
    system = "You are a professional Premier League scout. " + "detail " * 400
    out, err = rb._validate_messages([
        {"role": "system", "content": "You are a football data scientist."},
        {"role": "user", "content": system + timeline},
    ])
    assert err is None, err
    assert len(out) == 2


# --- per-IP rate limiting ----------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_limits():
    rb.rate_limits.clear()
    yield
    rb.rate_limits.clear()


def test_rate_limit_allows_then_blocks():
    ip = "203.0.113.9"
    for i in range(rb.MAX_REPORTS_PER_IP_PER_HOUR):
        assert rb._rate_limited(ip) is False, f"blocked early at request {i}"
    assert rb._rate_limited(ip) is True


def test_rate_limit_is_per_ip():
    for _ in range(rb.MAX_REPORTS_PER_IP_PER_HOUR):
        rb._rate_limited("198.51.100.1")
    assert rb._rate_limited("198.51.100.1") is True
    assert rb._rate_limited("198.51.100.2") is False   # a different user is unaffected


def test_rate_limit_buckets_by_hour(monkeypatch):
    """The allowance refills each hour rather than being a permanent ban."""
    import time as real_time

    # _rate_limited imports time inside the function, so patching the module
    # attribute is enough.
    fake = {"now": 1_000_000.0}
    monkeypatch.setattr(real_time, "time", lambda: fake["now"])

    ip = "192.0.2.5"
    for _ in range(rb.MAX_REPORTS_PER_IP_PER_HOUR):
        rb._rate_limited(ip)
    assert rb._rate_limited(ip) is True
    fake["now"] += 3600                                 # next hour
    assert rb._rate_limited(ip) is False


# --- client IP extraction ----------------------------------------------------

class _Req:
    def __init__(self, headers=None, host=None):
        self.headers = headers or {}
        self.client = types.SimpleNamespace(host=host) if host else None


def test_client_ip_prefers_first_forwarded_hop():
    """Modal sits behind a proxy, so the socket address is the proxy's — keying
    the rate limit on it would throttle every user as one."""
    req = _Req({"x-forwarded-for": "203.0.113.7, 70.0.0.1"}, host="10.0.0.1")
    assert rb._client_ip(req) == "203.0.113.7"


def test_client_ip_falls_back_to_socket_then_unknown():
    assert rb._client_ip(_Req(host="10.0.0.4")) == "10.0.0.4"
    assert rb._client_ip(_Req()) == "unknown"


# --- response_format passthrough --------------------------------------------

def test_json_object_request_is_forwarded():
    assert rb._response_format_kwargs({"type": "json_object"}) == {
        "response_format": {"type": "json_object"}}


@pytest.mark.parametrize("bad", [
    None, {}, {"type": "text"}, {"type": "json_schema", "schema": {}},
    "json_object", 42,
])
def test_anything_else_is_dropped(bad):
    """A client can't smuggle an arbitrary schema/tool-call config to Groq
    through this field — only the exact json_object flag passes through."""
    assert rb._response_format_kwargs(bad) == {}


# --- config sanity -----------------------------------------------------------

def test_output_cap_is_large_enough_for_the_real_report():
    """The prompt asks for ~500 words across 9 sections. The plan's 800-token
    cap would truncate that mid-report."""
    assert rb.MAX_OUTPUT_TOKENS >= 1500


def test_no_credentials_are_hardcoded():
    """The Groq key and app token must come from Modal secrets, never source."""
    src = open(rb.__file__, encoding="utf-8").read()
    assert "gsk_" not in src
    assert 'POLYFUT_APP_TOKEN"]' not in src or "os.environ" in src
