"""PolyFut AI scout-report proxy — runs on Modal, holds the Groq key server-side.

The app used to call Groq directly from the browser with a key the user pasted
in themselves (`script.js` → `api.groq.com`). That meant every user had to make
a Groq account before they could get a report. This endpoint takes that job
over: the Groq key lives only in Modal's secret store, the client sends chat
messages and gets text back, and nothing key-shaped ever ships in the installer.

    [PolyFut app] --messages--> [this endpoint] --> [Groq] --> text

Deploy:  modal deploy ai_backend/report_backend.py
Test:    modal serve  ai_backend/report_backend.py   (temporary URL)

Two Modal secrets are required — see ai_backend/README.md:
  * ``groq-secret``       → GROQ_API_KEY   (the real credential, never leaves Modal)
  * ``polyfut-app-token`` → POLYFUT_APP_TOKEN (shared token, also embedded in the app)

The app token is deliberately NOT a literal in this file: it would end up in git,
and it is easier to rotate as a secret. It is not a security boundary — anyone
who unpacks the installer can read it — it exists to keep random internet
traffic that finds the public URL from draining the free Groq quota. The Groq
key is the thing that stays genuinely hidden.
"""

from __future__ import annotations

import modal
from fastapi import Request

app = modal.App("polyfut-report-backend")
# httpx<0.28 pinned deliberately: groq==0.11.0's internal client passes a
# `proxies=` kwarg to httpx.Client(), which httpx 0.28 removed — an unpinned
# install pulls httpx 0.28.x via fastapi's dependency chain and breaks Groq
# client construction with "unexpected keyword argument 'proxies'".
image = modal.Image.debian_slim().pip_install(
    "groq==0.11.0", "fastapi[standard]", "httpx<0.28"
)

# --- Limits ---------------------------------------------------------------- #
# Groq's free tier is shared at the ORGANISATION level, not per key: every
# PolyFut user draws from the same bucket. These caps exist so one client can't
# exhaust it for everyone. Tune after watching real usage on the Groq console.
MAX_REPORTS_PER_IP_PER_HOUR = 20
MAX_MESSAGES = 40            # a report + ~19 follow-up turns
MAX_TOTAL_CHARS = 24_000     # ≈6k tokens; a long match timeline is the big input
MAX_OUTPUT_TOKENS = 2_000
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.6    # matches what the client sent when it called Groq directly

_VALID_ROLES = {"system", "user", "assistant"}

# --- vision (kit-colour reads) -------------------------------------------
# The kit picker sends a handful of match stills and asks which two shirt
# colours the teams are wearing. This is the abuse ceiling for a public
# endpoint, deliberately left at Groq's documented 5 rather than tracking the
# current model's real limit — qwen3.6-27b rejects more than 3, which the
# client enforces (kit_vlm.MAX_IMAGES) so a swap to a roomier model needs no
# redeploy here. Stills are 640x360 JPEGs, far under the 20MB request cap, and
# every byte is shared free-tier quota.
MAX_IMAGES = 5
MAX_IMAGE_BYTES = 400_000        # per image, as base64 — ~300KB of JPEG
MAX_IMAGE_BYTES_TOTAL = 1_500_000
VISION_MODEL = "qwen/qwen3.6-27b"

# Per-IP counters, bucketed by hour. A Dict survives across containers, so the
# limit holds even when Modal scales out to several instances.
rate_limits = modal.Dict.from_name("polyfut-rate-limits", create_if_missing=True)


def _client_ip(request) -> str:
    """Caller's IP. Modal sits behind a proxy, so the socket address is the
    proxy's — the first x-forwarded-for hop is the real client."""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return getattr(getattr(request, "client", None), "host", "") or "unknown"


def _rate_limited(ip: str) -> bool:
    """True if this IP has already used its allowance this hour.

    Read-modify-write isn't atomic, so two simultaneous requests can both see
    the same count. That undercounts by a request or two under concurrency,
    which is fine: this is a quota guard, not a billing meter.
    """
    import time

    bucket = f"{ip}:{int(time.time() // 3600)}"
    used = rate_limits.get(bucket, 0)
    if used >= MAX_REPORTS_PER_IP_PER_HOUR:
        return True
    rate_limits[bucket] = used + 1
    return False


def _validate_vision_content(content) -> tuple[list | None, str | None, int, int]:
    """Validate one message's multi-part content.

    Returns (parts, error, n_images, image_bytes). Text parts are length-capped
    the same way plain string content is; image parts must be inline data: URIs,
    never remote URLs — this endpoint is public, and honouring a caller-supplied
    http(s) URL would turn it into a fetcher for arbitrary hosts.
    """
    if not content:
        return None, "empty content list", 0, 0
    parts, n_images, image_bytes = [], 0, 0
    for part in content:
        if not isinstance(part, dict):
            return None, "each content part must be an object", 0, 0
        kind = part.get("type")
        if kind == "text":
            text = part.get("text")
            if not isinstance(text, str) or not text.strip():
                return None, "text part needs non-empty text", 0, 0
            if len(text) > MAX_TOTAL_CHARS:
                return None, "text part too long", 0, 0
            parts.append({"type": "text", "text": text})
        elif kind == "image_url":
            holder = part.get("image_url")
            url = holder.get("url") if isinstance(holder, dict) else None
            if not isinstance(url, str) or not url.startswith("data:image/"):
                return None, "image_url must be an inline data:image/... URI", 0, 0
            if len(url) > MAX_IMAGE_BYTES:
                return None, "image too large", 0, 0
            n_images += 1
            image_bytes += len(url)
            parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            return None, f"unsupported content part: {kind!r}", 0, 0
    return parts, None, n_images, image_bytes


def _validate_messages(raw) -> tuple[list[dict] | None, str | None]:
    """Return (messages, error). Rejects anything that isn't a well-formed chat
    history, and caps size so one oversized match timeline can't consume the
    per-minute token budget everyone else shares."""
    if not isinstance(raw, list) or not raw:
        return None, "messages must be a non-empty list"
    if len(raw) > MAX_MESSAGES:
        return None, f"too many messages (max {MAX_MESSAGES}) — start a new report"

    out: list[dict] = []
    total = 0
    n_images = 0
    image_bytes = 0
    for msg in raw:
        if not isinstance(msg, dict):
            return None, "each message must be an object"
        role = msg.get("role")
        content = msg.get("content")
        if role not in _VALID_ROLES:
            return None, f"invalid role: {role!r}"
        # Vision: content may be a list of {"type":"text"} / {"type":"image_url"}
        # parts. Validated part by part rather than passed through, for the same
        # reason _response_format_kwargs exists — a caller that reaches this
        # endpoint must not be able to hand arbitrary structure to the upstream
        # API. Only data: URIs are accepted: a remote URL would make this
        # endpoint fetch whatever a caller names.
        if isinstance(content, list):
            parts, err, n_img, n_bytes = _validate_vision_content(content)
            if err:
                return None, err
            n_images += n_img
            image_bytes += n_bytes
            if n_images > MAX_IMAGES:
                return None, f"too many images (max {MAX_IMAGES})"
            if image_bytes > MAX_IMAGE_BYTES_TOTAL:
                return None, "images too large in total"
            out.append({"role": role, "content": parts})
            continue
        if not isinstance(content, str) or not content.strip():
            return None, "each message needs non-empty string content"
        total += len(content)
        if total > MAX_TOTAL_CHARS:
            return None, (
                f"conversation too long (max {MAX_TOTAL_CHARS} characters) — "
                f"generate a fresh report instead of continuing this thread"
            )
        out.append({"role": role, "content": content})
    return out, None


# Groq's documented values. An allowlist rather than a pass-through for the
# same reason _response_format_kwargs is one: a caller reaching this public
# endpoint must not be able to hand arbitrary fields to the upstream API.
_REASONING_EFFORTS = {"none", "default", "low", "medium", "high"}


def _reasoning_kwargs(effort) -> dict:
    """Forward a known reasoning_effort value, drop anything else.

    qwen3.6-27b reasons in a <think> block by default. For the kit-colour read
    that is pure cost: the thinking ran past a 2000-token completion budget and
    was truncated before the model ever emitted its answer, and the reserved
    budget also counts toward the request-size limit that was rejecting three
    images with a 413. "none" turns it off, leaving just the answer.

    Sent via ``extra_body`` rather than as a named argument: the pinned
    groq==0.11.0 predates the parameter and raises "got an unexpected keyword
    argument 'reasoning_effort'". extra_body goes straight into the request
    JSON, so this works without disturbing a version pin that exists to keep
    httpx off 0.28 (see the image definition above).
    """
    if isinstance(effort, str) and effort in _REASONING_EFFORTS:
        return {"extra_body": {"reasoning_effort": effort}}
    return {}


def _response_format_kwargs(fmt) -> dict:
    """Only a JSON-mode request is forwarded to Groq; anything else in this
    field is dropped rather than passed through to the upstream API unvalidated
    (e.g. a client couldn't smuggle a schema or tool-call config through here)."""
    if isinstance(fmt, dict) and fmt.get("type") == "json_object":
        return {"response_format": {"type": "json_object"}}
    return {}


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("groq-secret"),
        modal.Secret.from_name("polyfut-app-token"),
    ],
    # An LLM API call is pure network wait — no GPU, minimal CPU, so this stays
    # inside Modal's free credit. A generous timeout because a 70B model writing
    # a full report can take a while under free-tier load.
    timeout=180,
    max_containers=4,
)
@modal.fastapi_endpoint(method="POST")
def generate_report(payload: dict, request: Request):
    """Proxy a chat completion to Groq.

    Request:  {"app_token": str, "messages": [{"role","content"}, ...],
               "model": str?, "temperature": float?, "max_tokens": int?,
               "response_format": {"type": "json_object"}?}
    Response: {"report": str, "model": str, "usage": {...}}
    Errors:   {"error": str} with a real HTTP status (401/400/429/502).
    """
    import os

    from fastapi.responses import JSONResponse
    from groq import Groq

    def fail(status: int, message: str):
        # Modal endpoints don't turn a returned tuple into a status code — it
        # would be serialised as a JSON array with a 200. JSONResponse does.
        return JSONResponse(status_code=status, content={"error": message})

    expected = os.environ.get("POLYFUT_APP_TOKEN", "")
    if not expected:
        return fail(500, "server misconfigured: POLYFUT_APP_TOKEN secret is missing")
    # Constant-time compare so the token can't be recovered a byte at a time by
    # timing the responses.
    import hmac

    if not hmac.compare_digest(str(payload.get("app_token", "")), expected):
        return fail(401, "unauthorized")

    messages, err = _validate_messages(payload.get("messages"))
    if err:
        return fail(400, err)

    ip = _client_ip(request)
    if _rate_limited(ip):
        return fail(429, (
            f"You've generated {MAX_REPORTS_PER_IP_PER_HOUR} reports in the last "
            f"hour, which is this app's limit. Try again shortly."
        ))

    try:
        temperature = float(payload.get("temperature", DEFAULT_TEMPERATURE))
    except (TypeError, ValueError):
        temperature = DEFAULT_TEMPERATURE
    try:
        max_tokens = int(payload.get("max_tokens", MAX_OUTPUT_TOKENS))
    except (TypeError, ValueError):
        max_tokens = MAX_OUTPUT_TOKENS
    max_tokens = max(256, min(max_tokens, MAX_OUTPUT_TOKENS))
    model = str(payload.get("model") or DEFAULT_MODEL)
    kwargs = _response_format_kwargs(payload.get("response_format"))
    kwargs.update(_reasoning_kwargs(payload.get("reasoning_effort")))

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=max(0.0, min(temperature, 2.0)),
            max_tokens=max_tokens,
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 — every failure must reach the user readably
        detail = str(exc)
        status = getattr(exc, "status_code", None)
        if status == 429 or "rate_limit" in detail:
            # The shared org quota, not this user's. Say so — otherwise it reads
            # as "the app is broken" when it's "everyone is using it right now".
            # Carry Groq's own wording through: its limits are per-minute AND
            # per-day, and "wait a minute" is wrong advice for the daily one —
            # which is what an image request is most likely to exhaust, a few
            # stills costing far more tokens than a whole report.
            return fail(429, "The AI service is busy right now (shared "
                             f"free-tier limit). Upstream said: {detail[:300]}")
        if status == 401:
            return fail(502, "The AI service rejected our credentials. This is a "
                             "server-side problem, not something you can fix.")
        return fail(502, f"AI service error: {detail[:300]}")

    choice = completion.choices[0].message.content if completion.choices else ""
    if not choice:
        return fail(502, "The AI returned an empty response. Please try again.")

    usage = getattr(completion, "usage", None)
    return {
        "report": choice,
        "model": model,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        },
    }


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health():
    """Unauthenticated liveness probe — the client uses it to decide whether the
    proxy is reachable before offering AI features. Deliberately reveals
    nothing: no key state, no usage, no token."""
    return {"ok": True, "service": "polyfut-report-backend"}
