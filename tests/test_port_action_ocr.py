"""The port action label is the exposure-independent port confirmation.

A dark/night harbour fails every brightness-based port anchor (measured 0 of 4
on the reporting machine), which used to make the page classify as BATTLE and
then as LOADING. Both wrong answers stalled the lifecycle: the battle branch
waited forever for a match clock, and ``wait_while_loading`` spun for 120 s
before the port workflow could even start.
"""

import cv2
import numpy as np
import pytest

import port_navigator
from bot import BattleBot
from core.ocr import OcrToken
from core.ui import PORT_BATTLE_BUTTON, ScreenState
from core.vision import Vision


def port_frame(*, with_button=True):
    """A dark frame whose only port evidence is the filled action button."""
    image = np.full((720, 1280, 3), 18, np.uint8)
    if with_button:
        x1, y1, x2, y2 = PORT_BATTLE_BUTTON.pixels(1280, 720)
        cv2.rectangle(image, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), (30, 60, 140), -1)
    return image


class RecordingBackend:
    def __init__(self, tokens=None):
        self.tokens = tokens or []
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        return list(self.tokens)


def wired_vision(tokens=None):
    vision = Vision()
    vision.text_backend = RecordingBackend(tokens)
    return vision


def test_anchor_failure_falls_back_to_the_rendered_action_label(monkeypatch):
    monkeypatch.setattr(
        port_navigator, "port_battle_action_point", lambda image, backend: (640, 20)
    )
    vision = wired_vision()
    image = port_frame()
    assert vision._port_anchor_votes(image) != {}
    assert sum(bool(v) for v in vision._port_anchor_votes(image).values()) < 3
    assert vision.in_port(image) is True


def test_missing_action_label_keeps_the_frame_out_of_port(monkeypatch):
    monkeypatch.setattr(
        port_navigator, "port_battle_action_point", lambda image, backend: None
    )
    vision = wired_vision()
    assert vision.in_port(port_frame()) is False


def test_without_a_button_no_ocr_is_spent(monkeypatch):
    calls = []

    def action_point(image, backend):
        calls.append(backend)
        return (640, 20)

    monkeypatch.setattr(port_navigator, "port_battle_action_point", action_point)
    vision = wired_vision()
    assert vision.in_port(port_frame(with_button=False)) is False
    assert calls == []


def test_unwired_vision_stays_pure_visual(monkeypatch):
    monkeypatch.setattr(
        port_navigator, "port_battle_action_point", lambda image, backend: (640, 20)
    )
    vision = Vision()
    assert vision.text_backend is None
    assert vision.in_port(port_frame()) is False


def test_port_verdict_is_cached_per_frame(monkeypatch):
    calls = []

    def action_point(image, backend):
        calls.append(image)
        return (640, 20)

    monkeypatch.setattr(port_navigator, "port_battle_action_point", action_point)
    vision = wired_vision()
    image = port_frame()
    assert vision.in_port(image) is True
    assert vision.in_port(image) is True
    assert len(calls) == 1


def test_ocr_failure_stays_fail_closed(monkeypatch):
    def broken(image, backend):
        raise RuntimeError("OCR unavailable")

    monkeypatch.setattr(port_navigator, "port_battle_action_point", broken)
    vision = wired_vision()
    assert vision.in_port(port_frame()) is False


def test_dark_harbour_classifies_as_port_not_loading(monkeypatch):
    monkeypatch.setattr(
        port_navigator, "port_battle_action_point", lambda image, backend: (640, 20)
    )
    vision = wired_vision()
    image = port_frame()
    assert vision.classify_screen(image) == ScreenState.PORT
    assert vision.in_loading(image) is False


def test_battle_bot_shares_its_ocr_backend_with_vision():
    backend = RecordingBackend([OcrToken("加入战斗", 0.99, ())])
    vision = Vision()
    bot = BattleBot(
        1,
        {"strategy": {}},
        vision=vision,
        gamepad=object(),
        distance_reader=type("Reader", (), {"backend": backend})(),
    )
    assert vision.text_backend is backend


def test_battle_bot_tolerates_a_vision_stub_without_the_hook():
    # Third-party/adaptor vision objects must not break construction.
    BattleBot(
        1,
        {"strategy": {}},
        vision=object(),
        gamepad=object(),
        distance_reader=type("Reader", (), {"backend": None})(),
    )


def test_wait_while_loading_returns_immediately_on_a_confirmed_port(monkeypatch):
    """The 120 s port stall lived in this helper, not in the scene router."""
    import main

    grabs = []

    def grab(_self, hwnd, allow_stale=False):
        grabs.append(hwnd)
        return port_frame()

    bot = type("Bot", (), {})()
    bot.hwnd = 1
    bot.vision = type("V", (), {"grab": grab})()
    bot.vision.classify_screen = staticmethod(lambda image: ScreenState.PORT)
    monkeypatch.setattr(main, "ensure_capture_foreground", lambda b: True)
    monkeypatch.setattr(main.time, "sleep", lambda _s: None)

    assert main.wait_while_loading(bot, timeout=5.0, should_stop=lambda: True) is not None
    assert len(grabs) == 1


def test_wait_while_loading_keeps_waiting_on_a_real_loading_page(monkeypatch):
    import main

    states = iter([ScreenState.LOADING, ScreenState.LOADING, ScreenState.PORT])
    grabs = []

    def grab(_self, hwnd, allow_stale=False):
        grabs.append(hwnd)
        return port_frame()

    bot = type("Bot", (), {})()
    bot.hwnd = 1
    bot.vision = type("V", (), {"grab": grab})()
    bot.vision.classify_screen = staticmethod(lambda image: next(states))
    monkeypatch.setattr(main, "ensure_capture_foreground", lambda b: True)
    monkeypatch.setattr(main.time, "sleep", lambda _s: None)

    assert main.wait_while_loading(bot, timeout=30.0) is not None
    assert len(grabs) == 3
