"""Ask a vision model which two kit colours a match is being played in.

Why this exists
---------------
The k-means picker reads 15px torso crops in isolation. It has no idea that
eleven figures spread in a formation are a team, that a lone figure in red
between them is the referee, or that the row of shapes along the top edge is a
crowd. On an ISB v TAS broadcast it returned two purples for an orange team and
a navy one — and a wrong kit pair is not cosmetic: measured on that footage, the
team gate keeps 98% of your touches with the right pair and 16-26% with a wrong
one, because everything it decides is "the other team" is dropped outright.

A vision model has exactly the scene understanding the crops lack, and the job
is a handful of stills ONCE per match, so it costs nothing per contact and
nothing per frame. It must never enter the tracking loop.

Trust model
-----------
The model does not get to invent a colour. It is shown the frames alongside the
colours actually measured off players in this video, and asked to CHOOSE two of
them by label. A kit that is not on the pitch is therefore unrepresentable,
rather than merely filtered out afterwards.

That design is not fastidiousness, it is forced. Letting the model emit a free
hex and snapping it to the nearest measurement cannot be made safe: measured on
this footage, the worst *correct* naming (a model saying "#ffa500" for a shirt
measuring "#8f561b") sits dE76 91 away, while the closest *wrong* answer
("#ff0000") sits 71.6 away. Vivid-versus-dull spans more distance than
orange-versus-red, so no threshold separates them. Choosing from a list has no
such failure mode.

This is strictly an enhancement: no key, no network, a refusal, a malformed
reply or an out-of-range choice all fall back to the existing local path.
"""

from __future__ import annotations

import base64
import json
import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

VISION_MODEL = "qwen/qwen3.6-27b"
# Groq's vision docs say 5 images per request; this model rejects more than 3
# ("Too many images provided. This model supports up to 3 images"), measured
# against the live API. Trust the API over the docs.
MAX_IMAGES = 3
JPEG_QUALITY = 82
SEND_WIDTH = 640            # native for this footage; no gain in upscaling

_LABELS = "ABCDEFGH"

_PROMPT_HEAD = (
    "These are stills from one association-football match, followed by a list "
    "of colours that were measured off the players' shirts in this same video.\n"
    "Choose which TWO listed colours are the shirt colours of the two TEAMS.\n"
    "Rules:\n"
    "- Ignore the referee and assistant referees (often a distinct colour worn "
    "by one or two people who are not grouped as a team).\n"
    "- Ignore spectators, benches, coaching staff, tents and pitch markings.\n"
    "- Ignore goalkeepers; they wear a different colour from their own team.\n"
    "- Judge by the outfield players who appear in numbers and are spread "
    "across the pitch in a formation.\n"
    "- Some listed colours are wrong (grass, skin, a crowd) — that is why you "
    "are choosing rather than being told.\n"
    "- The colours are dulled by distance, sunlight and video compression, so "
    "pick the closest listed match rather than an ideal kit colour.\n"
)

_PROMPT_TAIL = (
    '\nReply with JSON only, using the letters above: '
    '{"team_one":"A","team_two":"B","confident":true|false}. '
    "Set confident to false if these frames do not clearly show two teams "
    "(a warm-up, a replay, a crowd shot), or if neither team's shirt colour "
    "appears in the list."
)


def describe_colour(hexstr: str) -> str:
    """A coarse plain-language name, so the list reads as colours to a model
    rather than as six opaque hex codes."""
    h = hexstr.lstrip("#")
    bgr = np.uint8([[[int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)]]])
    hh, s, v = (int(x) for x in cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0])
    if v < 50:
        return "very dark / near-black"
    if s < 40:
        return "white" if v > 180 else "grey"
    deg = hh * 2
    for lim, nm in ((20, "red"), (45, "orange"), (70, "yellow"),
                    (100, "yellow-green"), (160, "green"), (200, "cyan"),
                    (260, "blue"), (320, "purple/magenta"), (361, "red")):
        if deg < lim:
            base = nm
            break
    shade = "dark " if v < 110 else ("pale " if s < 90 else "")
    return f"{shade}{base}"


def build_palette_text(candidates: list[str]) -> str:
    """The lettered colour list the model chooses from."""
    lines = []
    for i, hx in enumerate(candidates[:len(_LABELS)]):
        lines.append(f"  {_LABELS[i]}. {hx}  ({describe_colour(hx)})")
    return "Colours measured on players in this video:\n" + "\n".join(lines)


def _spread(frames: list, n: int) -> list:
    """Pick n items spread across the list, not the first n.

    With only three slots the choice matters: the candidates are ordered by
    time, and three consecutive early ones can all be the same phase of the
    match (or all still the warm-up). Spreading them samples the game.
    """
    if len(frames) <= n:
        return list(frames)
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in dict.fromkeys(idx.tolist())]


def frames_to_data_uris(frames: list[np.ndarray]) -> list[str]:
    """JPEG-encode BGR frames as inline data: URIs, at most MAX_IMAGES of them."""
    uris = []
    for fr in _spread(list(frames), MAX_IMAGES):
        if fr is None or getattr(fr, "size", 0) == 0:
            continue
        if fr.shape[1] > SEND_WIDTH:
            h = int(round(fr.shape[0] * SEND_WIDTH / fr.shape[1]))
            fr = cv2.resize(fr, (SEND_WIDTH, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", fr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue
        uris.append("data:image/jpeg;base64,"
                    + base64.b64encode(buf.tobytes()).decode("ascii"))
    return uris


def build_messages(data_uris: list[str], candidates: list[str]) -> list[dict]:
    """The chat payload: prompt, then the frames, then the colour list.

    Separate from the transport so it can be tested without a network.
    """
    parts: list[dict] = [{"type": "text", "text": _PROMPT_HEAD}]
    for uri in data_uris:
        parts.append({"type": "image_url", "image_url": {"url": uri}})
    parts.append({"type": "text",
                  "text": build_palette_text(candidates) + _PROMPT_TAIL})
    return [{"role": "user", "content": parts}]


def parse_choice(text: str, candidates: list[str]) -> list[str] | None:
    """Resolve the model's two chosen letters to measured colours.

    Tolerates the fenced- and prose-wrapped JSON models emit even under an
    explicit JSON instruction. Refuses an unconfident answer, a letter outside
    the list, or both picks landing on the same colour.
    """
    if not text or not candidates:
        return None
    raw = text.strip()
    # qwen3.6-27b is a thinking model: it emits a <think>...</think> block
    # before the answer, and that block is prose which can easily contain
    # braces ("the JSON should be {team_one: ...}"). Scanning for the first
    # brace without removing it would parse the model's working-out instead of
    # its answer, so strip it first. An unterminated block means the reply was
    # cut off mid-thought and there is no answer to find.
    if "<think>" in raw:
        _before, _sep, after = raw.partition("</think>")
        if not _sep:
            log.info("kit vision read: reply ended inside its <think> block "
                     "(max_tokens too small?)")
            return None
        raw = after.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(raw[start:end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("confident") is False:
        log.info("kit vision read declined: model was not confident these "
                 "frames show two teams")
        return None

    usable = candidates[:len(_LABELS)]
    picked = []
    for key in ("team_one", "team_two"):
        v = obj.get(key)
        if not isinstance(v, str) or not v.strip():
            return None
        letter = v.strip().upper()[0]
        idx = _LABELS.find(letter)
        if idx < 0 or idx >= len(usable):
            log.info("kit vision read: choice %r is outside the offered list", v)
            return None
        picked.append(idx)
    if len(set(picked)) != 2:
        return None
    return [usable[i] for i in picked]
