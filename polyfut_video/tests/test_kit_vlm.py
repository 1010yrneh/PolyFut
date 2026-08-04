"""The vision kit read must never be able to make things worse.

It exists because k-means reads 15px crops with no scene context and returned
two purples for an orange team and a navy one. But a model's answer is not
evidence, so every claim here is about the guard rails: a bad, absent, or
out-of-range reply has to leave the caller with exactly the k-means result it
already had.

The model CHOOSES from colours measured in the video rather than emitting a
colour of its own. That is forced, not fastidious — see
test_free_hex_snapping_could_not_have_been_made_safe below, which pins the
measurement that ruled the alternative out.

Nothing here touches the network — the transport lives in kit_vlm_client.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from polyfut_video.pipeline import kit_vlm

# The fixed picker's real output on the ISB/TAS clip, plus decoys of the kind
# k-means actually produced there (the two purples) and generic scene colours.
MEASURED = ["#beb9e4", "#140f2b", "#8f561b", "#110a25", "#cfcfcf", "#3d7a2f"]
ORANGE_IDX, NAVY_IDX = 2, 3          # -> letters C and D


# ------------------------------------------------------------ parse_choice
def test_a_plain_choice_resolves_to_measured_colours():
    out = kit_vlm.parse_choice(
        '{"team_one":"C","team_two":"D","confident":true}', MEASURED)
    assert out == ["#8f561b", "#110a25"]


def test_fenced_json_is_parsed():
    """Models emit ```json fences even when told to reply with JSON only."""
    out = kit_vlm.parse_choice(
        '```json\n{"team_one":"C","team_two":"D"}\n```', MEASURED)
    assert out == ["#8f561b", "#110a25"]


def test_prose_wrapped_json_is_parsed():
    out = kit_vlm.parse_choice(
        'The teams are in orange and navy: {"team_one":"C","team_two":"D"} '
        '— the figure in red is the referee.', MEASURED)
    assert out == ["#8f561b", "#110a25"]


def test_lowercase_and_trailing_text_in_a_choice_are_tolerated():
    out = kit_vlm.parse_choice(
        '{"team_one":"c. orange","team_two":"d"}', MEASURED)
    assert out == ["#8f561b", "#110a25"]


@pytest.mark.parametrize("reply", [
    "", "   ", "no json here at all", "{unclosed",
    '{"team_one":"C"}',                            # only one team
    '{"team_one":"C","team_two":"C"}',             # same colour twice
    '{"team_one":"C","team_two":"Z"}',             # letter not offered
    '{"team_one":"C","team_two":""}',
    '{"team_one":"C","team_two":null}',
    '{"team_one":"#8f561b","team_two":"#110a25"}',  # a hex is not a choice
])
def test_an_unusable_reply_is_refused(reply):
    assert kit_vlm.parse_choice(reply, MEASURED) is None


def test_a_letter_past_the_end_of_a_short_list_is_refused():
    """F is a valid label but there is no F when only three colours are offered."""
    assert kit_vlm.parse_choice(
        '{"team_one":"A","team_two":"F"}', MEASURED[:3]) is None


def test_an_unconfident_answer_is_refused():
    """The prompt lets the model say the frames don't show two teams — that is
    the warm-up case, and guessing there is what caused the original bug."""
    assert kit_vlm.parse_choice(
        '{"team_one":"A","team_two":"B","confident":false}', MEASURED) is None


def test_no_candidates_means_no_answer():
    assert kit_vlm.parse_choice('{"team_one":"A","team_two":"B"}', []) is None


def test_a_choice_can_only_ever_be_a_measured_colour():
    """The anti-hallucination property, stated directly: whatever the model
    replies, the result is drawn from the list or is None."""
    for reply in ('{"team_one":"A","team_two":"B"}',
                  '{"team_one":"E","team_two":"F"}',
                  '{"team_one":"#ff0000","team_two":"#00ff00"}',
                  '{"team_one":"Q","team_two":"R"}'):
        out = kit_vlm.parse_choice(reply, MEASURED)
        assert out is None or all(c in MEASURED for c in out)


def test_the_reported_failure_is_correctable():
    """k-means offered two purples; both true kits are in the wider palette,
    so the model has something correct available to choose."""
    out = kit_vlm.parse_choice('{"team_one":"C","team_two":"D"}', MEASURED)
    assert out == ["#8f561b", "#110a25"]
    assert "#beb9e4" not in out and "#140f2b" not in out


# ------------------------------------------------- why choosing, not naming
def test_free_hex_snapping_could_not_have_been_made_safe():
    """Pins the measurement that forced the choose-from-a-list design.

    A model naming a dull sunlit orange shirt as vivid "#ffa500" lands FURTHER
    from the true measurement than plain red does. Any single distance
    threshold therefore either rejects correct answers or accepts wrong ones.
    """
    def de(a, b):
        def lab(hx):
            h = hx.lstrip("#")
            bgr = np.uint8([[[int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)]]])
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        return float(np.linalg.norm(lab(a) - lab(b)))

    true_but_far = de("#ffa500", "#8f561b")     # correct kit, vividly named
    wrong_but_near = de("#ff0000", "#8f561b")   # red, not on this pitch
    assert true_but_far > wrong_but_near, (
        f"vivid-correct {true_but_far:.0f} should exceed wrong-but-close "
        f"{wrong_but_near:.0f}; if this ever stops holding, snapping a free "
        f"hex to the nearest measurement becomes viable again"
    )


# ------------------------------------------------------------ palette text
def test_the_palette_lists_every_candidate_with_a_letter():
    text = kit_vlm.build_palette_text(MEASURED)
    for letter, hx in zip("ABCDEF", MEASURED):
        assert f"{letter}. {hx}" in text


def test_candidates_are_described_in_words_not_just_hex():
    assert "orange" in kit_vlm.describe_colour("#8f561b")
    assert "dark" in kit_vlm.describe_colour("#110a25")
    assert kit_vlm.describe_colour("#cfcfcf") in ("grey", "white")
    assert "green" in kit_vlm.describe_colour("#3d7a2f")


# ------------------------------------------------------------- the payload
def test_images_are_capped_at_the_api_limit():
    frames = [np.full((36, 64, 3), i * 10, np.uint8) for i in range(12)]
    assert len(kit_vlm.frames_to_data_uris(frames)) == kit_vlm.MAX_IMAGES <= 5


def test_empty_and_broken_frames_are_skipped():
    frames = [None, np.zeros((0, 0, 3), np.uint8),
              np.full((36, 64, 3), 90, np.uint8)]
    assert len(kit_vlm.frames_to_data_uris(frames)) == 1


def test_frames_are_encoded_as_inline_data_uris():
    uris = kit_vlm.frames_to_data_uris([np.full((36, 64, 3), 120, np.uint8)])
    assert uris[0].startswith("data:image/jpeg;base64,")


def test_message_is_prompt_then_images_then_the_colour_list():
    uris = kit_vlm.frames_to_data_uris([np.full((36, 64, 3), 120, np.uint8)] * 3)
    msgs = kit_vlm.build_messages(uris, MEASURED)
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    parts = msgs[0]["content"]
    assert parts[0]["type"] == "text"
    assert [p["type"] for p in parts[1:4]] == ["image_url"] * 3
    assert parts[4]["type"] == "text" and "A. #beb9e4" in parts[4]["text"]


def test_the_prompt_excludes_the_people_that_broke_the_kmeans_read():
    p = (kit_vlm._PROMPT_HEAD + kit_vlm._PROMPT_TAIL).lower()
    for who in ("referee", "spectator", "goalkeeper", "warm-up"):
        assert who in p, who
