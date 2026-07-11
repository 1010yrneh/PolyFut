"""Tests for adaptive review grouping (kit colour / other-team / appearance)."""

from types import SimpleNamespace

import numpy as np

from polyfut_v2.pipeline.grouping import assign_player_groups


def _item(rank, jersey_hsv, crop):
    contact = SimpleNamespace(jersey_hsv=jersey_hsv, torso_crop=crop)
    return SimpleNamespace(rank=rank, scored=SimpleNamespace(contact=contact))


def _solid(bgr):
    img = np.zeros((20, 12, 3), np.uint8)
    img[:] = bgr
    return img


def test_other_team_flagged_and_kept_separate():
    seed = SimpleNamespace(kit_hsv=np.array([0, 200, 150], np.float32))  # red kit
    items = [
        _item(0, [5, 200, 150], _solid((0, 0, 200))),     # you: red
        _item(1, [3, 190, 140], _solid((0, 0, 205))),     # teammate: red, same look
        _item(2, [4, 195, 150], _solid((0, 200, 0))),     # teammate: red, green shirt-print
        _item(3, [120, 200, 150], _solid((200, 0, 0))),   # opponent: blue
        _item(4, [118, 210, 150], _solid((205, 0, 0))),   # opponent: blue
    ]
    g = assign_player_groups(items, seed)

    assert g[3]["is_other_team"] and g[4]["is_other_team"]
    assert not any(g[r]["is_other_team"] for r in (0, 1, 2))
    # opponents share a kit group distinct from your team's
    assert g[3]["kit_group"] == g[4]["kit_group"]
    assert g[3]["kit_group"] != g[0]["kit_group"]
    # opponents get no appearance group (only your team is clustered)
    assert g[3]["appearance_group"] == -1


def test_same_kit_appearance_groups():
    seed = SimpleNamespace(kit_hsv=np.array([0, 200, 150], np.float32))
    a = _solid((0, 0, 200))
    items = [
        _item(0, [4, 200, 150], a),                       # you
        _item(1, [4, 200, 150], a.copy()),                # look-alike → same group
        _item(2, [4, 200, 150], _solid((0, 180, 0))),     # different look → other group
    ]
    g = assign_player_groups(items, seed)
    assert g[0]["appearance_group"] == g[1]["appearance_group"] >= 0
    assert g[2]["appearance_group"] != g[0]["appearance_group"]


def test_no_seed_kit_disables_other_team():
    seed = SimpleNamespace(kit_hsv=None)
    items = [
        _item(0, [5, 200, 150], _solid((0, 0, 200))),
        _item(1, [120, 200, 150], _solid((200, 0, 0))),   # would-be opponent
    ]
    g = assign_player_groups(items, seed)
    assert not g[0]["is_other_team"] and not g[1]["is_other_team"]


def test_missing_colour_stays_same_team():
    seed = SimpleNamespace(kit_hsv=np.array([0, 200, 150], np.float32))
    items = [_item(0, None, None)]                         # no colour, no crop
    g = assign_player_groups(items, seed)
    assert g[0] == {"kit_group": -1, "is_other_team": False, "appearance_group": -1}
