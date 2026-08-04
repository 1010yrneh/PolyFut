"""Transport for the kit-colour vision call.

Kept apart from kit_vlm so the prompt, parsing and snapping can be tested
without a network, a key, or anyone's quota.

Two routes, in order:

1. the app's own Modal proxy (``ai_config.json`` / POLYFUT_AI_PROXY_URL), which
   holds the shared Groq key server-side — the same path the scout report uses;
2. a Groq key in the environment (GROQ_API_KEY), for a developer running the
   pipeline directly.

If neither is configured this returns None and the caller keeps its local
k-means result. That is the expected state for most users: the key is optional
and everything except the scout report works without it.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Short: this runs while the user waits on the team-picker screen, and a slow
# answer is worth less than falling straight back to k-means.
TIMEOUT_SEC = 45

# --- circuit breaker ------------------------------------------------------
# Catching the error per call is not enough. Groq's free tier is shared at the
# ORGANISATION level, so once it is exhausted it stays exhausted for everyone,
# and without this every subsequent analysis would stall the team-picker screen
# for the full timeout before falling back to k-means — and burn another slot
# of the proxy's own hourly per-IP allowance doing it. After a refusal we stop
# asking and go straight to the local read.
QUOTA_COOLDOWN_SEC = 15 * 60      # 429/413: may recover, retry later
AUTH_COOLDOWN_SEC = 24 * 60 * 60  # 401/403: will not fix itself unattended
_blocked_until = 0.0
_block_reason = ""


def _block(seconds: float, reason: str) -> None:
    global _blocked_until, _block_reason
    until = time.time() + seconds
    if until > _blocked_until:
        _blocked_until, _block_reason = until, reason
    log.info("kit vision read paused for %.0f min (%s); using the local "
             "k-means kit colours meanwhile", seconds / 60, reason)


def blocked_for() -> float:
    """Seconds until the vision read is worth attempting again (0 = now)."""
    return max(0.0, _blocked_until - time.time())


def reset_block() -> None:
    """Test hook / manual clear."""
    global _blocked_until, _block_reason
    _blocked_until, _block_reason = 0.0, ""


def _post(url: str, payload: dict, headers: dict) -> dict | None:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          **headers})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")[:300]
        except Exception:
            pass
        log.info("kit VLM HTTP %s: %s", e.code, detail)
        if e.code in (401, 403):
            _block(AUTH_COOLDOWN_SEC, f"credentials rejected (HTTP {e.code})")
        elif e.code in (402, 413, 429):
            # 413 is Groq's "request too large", which it reports alongside the
            # rate-limit family; either way the next identical request fails too.
            _block(QUOTA_COOLDOWN_SEC, f"quota or size limit (HTTP {e.code})")
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
        log.info("kit VLM unreachable: %s", e)
        # Offline or the proxy is down — both persist for a while, and the user
        # should not wait out the timeout again on their next run.
        _block(QUOTA_COOLDOWN_SEC, f"unreachable ({type(e).__name__})")
    return None


def load_proxy_config(config_paths=None) -> tuple[str, str] | None:
    """(proxy_url, app_token) from env or ai_config.json, or None."""
    url = os.environ.get("POLYFUT_AI_PROXY_URL", "").strip()
    token = os.environ.get("POLYFUT_AI_APP_TOKEN", "").strip()
    if url and token:
        return url, token
    for path in config_paths or []:
        try:
            with open(path, encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (OSError, ValueError):
            continue
        url = str(cfg.get("proxy_url") or "").strip()
        token = str(cfg.get("app_token") or "").strip()
        # The shipped example file has placeholders in both fields; treat those
        # as "not configured" rather than firing a request that must 401.
        if url and token and "YOUR-MODAL" not in url and not token.startswith("PASTE_"):
            return url, token
    return None


def ask(messages: list[dict], model: str, *, config_paths=None) -> str | None:
    """Send a vision chat request; return the reply text, or None."""
    waiting = blocked_for()
    if waiting > 0:
        log.info("kit vision read skipped (%s; %.0f min left) — using k-means",
                 _block_reason, waiting / 60)
        return None

    payload = {
        "messages": messages,
        "model": model,
        "temperature": 0,          # a colour read is not a creative task
        # Room to think, not room to write. qwen3.6-27b reasons inside a
        # <think> block before answering, and that block counts against
        # max_tokens: at 300 it was consumed entirely by reasoning and Groq
        # rejected the empty result with json_validate_failed. The answer
        # itself is about 30 tokens.
        # Reasoning off. Left on, qwen3.6-27b spent more than 2000 completion
        # tokens narrating its way down the colour list and was truncated before
        # it ever answered. Its reasoning was correct — it identified the
        # referee's shirt and the grass unprompted — but this call needs the
        # conclusion, not the derivation. Off, the answer is ~30 tokens, which
        # also shrinks the reserved budget that counts toward the request-size
        # limit rejecting three images with a 413.
        "reasoning_effort": "none",
        "max_tokens": 300,
        # Deliberately NOT response_format=json_object. Groq validates that the
        # whole completion is JSON, and with reasoning ON that failed against
        # the <think> block (HTTP 400 json_validate_failed, empty
        # failed_generation, at every payload size). parse_choice tolerates
        # fences and surrounding prose, so JSON mode buys nothing it needs.
    }

    proxy = load_proxy_config(config_paths)
    if proxy:
        url, token = proxy
        data = _post(url, {**payload, "app_token": token}, {})
        if data:
            # The proxy returns {"report": str}; Groq returns OpenAI shape.
            text = data.get("report")
            if isinstance(text, str) and text.strip():
                return text
            log.info("kit VLM proxy replied without usable text: %s",
                     list(data)[:5])

    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        data = _post(GROQ_URL, payload, {"Authorization": f"Bearer {key}"})
        if data:
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                log.info("kit VLM: unexpected Groq response shape")

    if not proxy and not key:
        log.info("kit VLM not configured (no proxy, no GROQ_API_KEY) — "
                 "keeping the local k-means kit read")
    return None
