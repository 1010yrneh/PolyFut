"""The kit-vision opt-out has to actually stick.

This setting decides whether stills from the user's match leave their machine.
It is on by default, so the only thing standing between that default and a user
who does not want it is this precedence chain - and every link is a place where
a wrong answer is silent. Nothing errors if the setting is ignored; frames just
keep being sent.

The ordering that matters most: the user's saved choice must beat the packaged
ai_config.json. ai_config.json lives in the install directory, which is
read-only on an all-users install, so if the shipped default won, the users
least able to edit files would be the ones unable to opt out.
"""

from __future__ import annotations

import json

import pytest

import server


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Point DATA_ROOT at a temp dir so settings.json is per-test."""
    monkeypatch.setattr(server, "DATA_ROOT", tmp_path)
    monkeypatch.delenv("POLYFUT_KIT_VISION", raising=False)
    return tmp_path


@pytest.fixture
def no_ai_config(monkeypatch):
    """No ai_config.json opinion, so the default and prefs are what decide."""
    monkeypatch.setattr(server, "_load_ai_config",
                        lambda: {"proxy_url": "", "app_token": ""})


# ------------------------------------------------------------ the default
def test_on_by_default(prefs, no_ai_config):
    assert server._kit_vision_enabled() is True


# ------------------------------------------------------------ precedence
def test_user_choice_beats_packaged_config(prefs, monkeypatch):
    """The whole point: a read-only install must still be refusable."""
    monkeypatch.setattr(server, "_load_ai_config",
                        lambda: {"proxy_url": "u", "app_token": "t",
                                 "kit_vision": True})
    server._save_prefs({"kit_vision": False})
    assert server._kit_vision_enabled() is False


def test_packaged_config_applies_when_user_has_not_chosen(prefs, monkeypatch):
    monkeypatch.setattr(server, "_load_ai_config",
                        lambda: {"proxy_url": "u", "app_token": "t",
                                 "kit_vision": False})
    assert server._kit_vision_enabled() is False


@pytest.mark.parametrize("env,expected", [
    ("0", False), ("false", False), ("off", False), ("no", False),
    ("1", True), ("true", True), ("on", True), ("yes", True),
])
def test_env_var_overrides_a_saved_choice(prefs, no_ai_config, monkeypatch,
                                          env, expected):
    server._save_prefs({"kit_vision": not expected})
    monkeypatch.setenv("POLYFUT_KIT_VISION", env)
    assert server._kit_vision_enabled() is expected


# ------------------------------------------------------- persistence
def test_choice_survives_a_restart(prefs, no_ai_config):
    server._save_prefs({"kit_vision": False})
    # A fresh read is what the next process would do - no caching in between.
    assert server._load_prefs()["kit_vision"] is False
    assert server._kit_vision_enabled() is False


def test_saving_preserves_unrelated_settings(prefs, no_ai_config):
    server._save_prefs({"something_else": 42})
    server._save_prefs({"kit_vision": False})
    doc = server._load_prefs()
    assert doc["something_else"] == 42 and doc["kit_vision"] is False


def test_corrupt_settings_file_does_not_break_the_app(prefs, no_ai_config):
    """A bad settings file must fall back to defaults, not take the app down."""
    (prefs / "settings.json").write_text("{not json", encoding="utf-8")
    assert server._load_prefs() == {}
    assert server._kit_vision_enabled() is True


def test_settings_file_holding_a_list_is_ignored(prefs, no_ai_config):
    (prefs / "settings.json").write_text("[1, 2]", encoding="utf-8")
    assert server._load_prefs() == {}


# ------------------------------------------------------------- endpoint
@pytest.fixture
def client(prefs, monkeypatch):
    monkeypatch.setattr(server, "_load_ai_config",
                        lambda: {"proxy_url": "u", "app_token": "t"})
    return server.app.test_client()


def test_get_reports_the_effective_state(client):
    body = client.get("/api/settings").get_json()
    assert body["kit_vision"] is True
    assert body["available"] is True and body["locked"] is False


def test_post_turns_it_off_and_persists(client, prefs):
    body = client.post("/api/settings", json={"kit_vision": False}).get_json()
    assert body["kit_vision"] is False
    saved = json.loads((prefs / "settings.json").read_text(encoding="utf-8"))
    assert saved["kit_vision"] is False
    # And a fresh GET agrees, rather than only the response being right.
    assert client.get("/api/settings").get_json()["kit_vision"] is False


def test_post_rejects_a_non_boolean(client):
    r = client.post("/api/settings", json={"kit_vision": "false"})
    assert r.status_code == 400


def test_post_refuses_when_pinned_by_env(client, monkeypatch):
    """Accepting a save the env var would override would show a false state."""
    monkeypatch.setenv("POLYFUT_KIT_VISION", "1")
    r = client.post("/api/settings", json={"kit_vision": False})
    assert r.status_code == 409
    assert r.get_json()["locked"] is True
    assert server._kit_vision_enabled() is True


def test_available_false_when_no_proxy_configured(prefs, monkeypatch):
    monkeypatch.setattr(server, "_load_ai_config",
                        lambda: {"proxy_url": "", "app_token": ""})
    body = server.app.test_client().get("/api/settings").get_json()
    assert body["available"] is False


# -------------------------------------------------- the hook actually obeys
def test_hook_is_none_when_switched_off(prefs, no_ai_config):
    """The setting is only real if the pipeline hook respects it."""
    server._save_prefs({"kit_vision": False})
    assert server._kit_vision_hook(lambda: object()) is None
