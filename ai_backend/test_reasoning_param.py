"""reasoning_effort is forwarded, but only known values.

The endpoint is public and holds a shared key, so every field that reaches the
upstream API is an allowlist — the same rule _response_format_kwargs follows.
"""

from __future__ import annotations

import pytest

from ai_backend.report_backend import _reasoning_kwargs


@pytest.mark.parametrize("effort", ["none", "default", "low", "medium", "high"])
def test_documented_values_are_forwarded(effort):
    """Through extra_body: the pinned groq==0.11.0 rejects reasoning_effort as
    a named argument, and that pin exists to keep httpx off 0.28."""
    assert _reasoning_kwargs(effort) == {
        "extra_body": {"reasoning_effort": effort}}


@pytest.mark.parametrize("bad", [
    "None", "NONE", "off", "maximum", "", "  none  ", 1, True, None,
    ["none"], {"effort": "none"}, {"$ne": None},
])
def test_anything_else_is_dropped(bad):
    assert _reasoning_kwargs(bad) == {}


def test_the_kit_read_turns_reasoning_off():
    """Pins the setting the live API forced: with reasoning on, the model spent
    its whole completion budget narrating and never emitted an answer."""
    from polyfut_video.pipeline import kit_vlm_client

    import inspect
    src = inspect.getsource(kit_vlm_client.ask)
    assert '"reasoning_effort": "none"' in src
    # The key itself, not the word in the comment explaining its absence:
    # JSON mode cannot be used while this model emits a <think> block.
    assert '"response_format":' not in src
