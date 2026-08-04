"""The proxy is a public endpoint holding a shared Groq key, so widening it to
accept images widens what an unknown caller can send. These pin the limits.

The endpoint itself needs Modal to run; _validate_messages and
_validate_vision_content are plain functions and are what actually decide what
reaches Groq, so they are tested directly.
"""

from __future__ import annotations

import pytest

from ai_backend.report_backend import (
    MAX_IMAGES, _validate_messages, _validate_vision_content,
)

TINY = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="


def _img(uri=TINY):
    return {"type": "image_url", "image_url": {"url": uri}}


# ------------------------------------------------------- still accepts text
def test_plain_text_conversations_are_unaffected():
    msgs, err = _validate_messages([{"role": "user", "content": "hello"}])
    assert err is None and msgs == [{"role": "user", "content": "hello"}]


# ------------------------------------------------------------ accepts images
def test_a_vision_message_is_accepted():
    msgs, err = _validate_messages([{
        "role": "user",
        "content": [{"type": "text", "text": "which two kits?"}, _img()],
    }])
    assert err is None
    assert msgs[0]["content"][1]["image_url"]["url"] == TINY


# ------------------------------------------------------------- the limits
def test_a_remote_image_url_is_refused():
    """Honouring a caller-supplied http URL would make this public endpoint
    fetch arbitrary hosts on request."""
    _parts, err, _n, _b = _validate_vision_content(
        [_img("https://example.com/pic.jpg")])
    assert err and "data:image" in err


@pytest.mark.parametrize("uri", [
    "file:///etc/passwd", "http://169.254.169.254/latest/meta-data/",
    "ftp://example.com/x.jpg", "javascript:alert(1)", "",
])
def test_only_inline_image_data_is_accepted(uri):
    _parts, err, _n, _b = _validate_vision_content([_img(uri)])
    assert err is not None


def test_too_many_images_are_refused():
    content = [{"type": "text", "text": "go"}] + [_img()] * (MAX_IMAGES + 1)
    _msgs, err = _validate_messages([{"role": "user", "content": content}])
    assert err and "too many images" in err


def test_images_split_across_messages_still_count_against_the_cap():
    msgs = [{"role": "user", "content": [_img()]} for _ in range(MAX_IMAGES + 1)]
    _out, err = _validate_messages(msgs)
    assert err and "too many images" in err


def test_an_oversized_image_is_refused():
    huge = "data:image/jpeg;base64," + ("A" * 500_000)
    _parts, err, _n, _b = _validate_vision_content([_img(huge)])
    assert err and "too large" in err


def test_total_image_size_is_capped():
    each = "data:image/jpeg;base64," + ("A" * 390_000)
    content = [_img(each) for _ in range(MAX_IMAGES)]
    _msgs, err = _validate_messages([{"role": "user", "content": content}])
    assert err and "too large in total" in err


@pytest.mark.parametrize("part", [
    {"type": "audio", "audio": {}},
    {"type": "tool_call", "name": "x"},
    {"type": "text"},
    {"type": "text", "text": "   "},
    {"type": "image_url", "image_url": "not-an-object"},
    {"type": "image_url"},
    "not-an-object",
])
def test_unknown_or_malformed_parts_are_refused(part):
    _parts, err, _n, _b = _validate_vision_content([part])
    assert err is not None


def test_empty_content_list_is_refused():
    _parts, err, _n, _b = _validate_vision_content([])
    assert err is not None
