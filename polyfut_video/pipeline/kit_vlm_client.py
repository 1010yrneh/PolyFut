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
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Short: this runs while the user waits on the team-picker screen, and a slow
# answer is worth less than falling straight back to k-means.
TIMEOUT_SEC = 45


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
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
        log.info("kit VLM unreachable: %s", e)
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
    payload = {
        "messages": messages,
        "model": model,
        "temperature": 0,          # a colour read is not a creative task
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
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
