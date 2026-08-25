"""
tests/test_ui.py — Sprint 8: Premium UI Polish Unit Tests

Tests UI rendering enhancements from Sprint 8:
    - Glassmorphism Key drawing with cv2.addWeighted
    - Stats bar drawing (Word count & WPM calculations)
    - SPEAK key pulse ring drawing when is_speaking=True
    - Overall Keyboard.draw() integration with visual overlays

RUN WITH:
    .venv/bin/python -m unittest tests.test_ui -v
"""

import sys
import os
import time
import unittest
import numpy as np

# Add src/ to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from keyboard import (
    Keyboard,
    Key,
    GLASS_ALPHA,
    COLOR_PULSE,
    COLOR_STATS_BG,
    COLOR_STATS_TEXT,
    STATS_BAR_Y1,
    STATS_BAR_Y2,
)


class TestGlassmorphismKey(unittest.TestCase):
    """Test glassmorphism rendering on keys."""

    def test_key_draw_on_black_frame(self):
        """Drawing a key should blend onto frame without exceptions."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        key = Key("A", 100, 100, 58, 50)
        key.draw(frame, hovered=False)
        # Verify that pixels within key area are modified
        roi = frame[100:150, 100:158]
        self.assertGreater(int(np.sum(roi)), 0)

    def test_key_draw_hovered(self):
        """Drawing a hovered key should alter the frame."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        key = Key("A", 100, 100, 58, 50)
        key.draw(frame, hovered=True)
        roi = frame[100:150, 100:158]
        self.assertGreater(int(np.sum(roi)), 0)

    def test_glass_alpha_valid_range(self):
        """GLASS_ALPHA should be between 0.0 and 1.0."""
        self.assertGreater(GLASS_ALPHA, 0.0)
        self.assertLess(GLASS_ALPHA, 1.0)


class TestStatsBar(unittest.TestCase):
    """Test stats bar calculations and drawing."""

    def setUp(self):
        self.keyboard = Keyboard()

    def test_draw_stats_bar_empty_text(self):
        """Drawing stats bar with empty text should execute cleanly."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.keyboard.typed_text = ""
        self.keyboard._draw_stats_bar(frame, session_start=time.time())
        # Check that stats bar region is non-zero
        roi = frame[STATS_BAR_Y1:STATS_BAR_Y2, :]
        self.assertGreater(int(np.sum(roi)), 0)

    def test_draw_stats_bar_with_words(self):
        """Drawing stats bar with words should render word count."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.keyboard.typed_text = "HELLO WORLD TEST AIRTYPE"
        # Simulate session started 60 seconds ago
        session_start = time.time() - 60.0
        self.keyboard._draw_stats_bar(frame, session_start=session_start)
        roi = frame[STATS_BAR_Y1:STATS_BAR_Y2, :]
        self.assertGreater(int(np.sum(roi)), 0)

    def test_draw_stats_bar_no_session_start(self):
        """Passing None for session_start should not crash."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.keyboard.typed_text = "TESTING"
        self.keyboard._draw_stats_bar(frame, session_start=None)
        roi = frame[STATS_BAR_Y1:STATS_BAR_Y2, :]
        self.assertGreater(int(np.sum(roi)), 0)


class TestSpeakPulse(unittest.TestCase):
    """Test pulsing ring animation rendering."""

    def setUp(self):
        self.keyboard = Keyboard()

    def test_draw_speak_pulse_renders(self):
        """_draw_speak_pulse should modify frame pixels around SPEAK key."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.keyboard._draw_speak_pulse(frame)
        self.assertGreater(int(np.sum(frame)), 0)


class TestFullKeyboardDraw(unittest.TestCase):
    """Test the full Keyboard.draw() method with all overlays."""

    def setUp(self):
        self.keyboard = Keyboard()

    def test_full_draw_idle(self):
        """draw() when not speaking should execute without error."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.keyboard.draw(frame, finger_pos=(200, 500), is_speaking=False, session_start=time.time())
        self.assertGreater(int(np.sum(frame)), 0)

    def test_full_draw_speaking(self):
        """draw() when speaking should render keys, stats, and pulse."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.keyboard.draw(frame, finger_pos=None, is_speaking=True, session_start=time.time() - 30)
        self.assertGreater(int(np.sum(frame)), 0)


if __name__ == "__main__":
    unittest.main()
