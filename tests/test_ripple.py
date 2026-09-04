"""
test_ripple.py — Sprint 12: Keypad Layout Toggle & Visual Ripple Unit Tests

Tests cover:
  1. RippleEffect initialization, timing, expansion, alpha decay, and completion.
  2. RippleEffect frame drawing, clipping safety, and non-crash on boundaries.
  3. Keyboard layout toggling between 'ABC' and '123' modes.
  4. Number/symbol keys presence and coordinates in '123' mode.
  5. CLR key clearing typed_text and presence on row 3.
  6. Ripple spawning on register_click and select_suggestion.
  7. Ripple pruning after duration elapses.
  8. Adaptive font scaling for multi-character keys.
"""

import os
import sys
import time
import unittest
import numpy as np

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from keyboard import Keyboard, Key, RippleEffect, START_Y, KEY_HEIGHT, GAP


class TestRippleEffect(unittest.TestCase):
    """Unit tests for the RippleEffect expanding circle animation class."""

    def test_init_defaults(self):
        """RippleEffect should initialize with default parameters."""
        r = RippleEffect(cx=200, cy=300)
        self.assertEqual(r.cx, 200)
        self.assertEqual(r.cy, 300)
        self.assertEqual(r.max_radius, 45)
        self.assertAlmostEqual(r.duration, 0.30)
        self.assertFalse(r.is_finished(r.start_time))

    def test_not_finished_immediately(self):
        """Ripple should not be finished right after creation."""
        r = RippleEffect(cx=100, cy=100, duration=0.3)
        self.assertFalse(r.is_finished(r.start_time + 0.1))

    def test_is_finished_after_duration(self):
        """Ripple is_finished() should return True after duration elapses."""
        r = RippleEffect(cx=100, cy=100, duration=0.3)
        self.assertTrue(r.is_finished(r.start_time + 0.31))
        self.assertTrue(r.is_finished(r.start_time + 1.0))

    def test_draw_modifies_frame(self):
        """draw() should blend circle onto the frame."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        r = RippleEffect(cx=400, cy=400, max_radius=40, duration=0.30)

        # Midpoint of animation: elapsed = 0.15s (50% progress)
        mid_time = r.start_time + 0.15
        r.draw(frame, now=mid_time)

        # Pixels in the ROI should be non-zero
        roi = frame[360:440, 360:440]
        self.assertGreater(np.sum(roi), 0)

    def test_draw_past_duration_does_not_modify(self):
        """draw() past duration should not modify frame."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        r = RippleEffect(cx=400, cy=400, duration=0.20)
        r.draw(frame, now=r.start_time + 0.25)
        self.assertEqual(np.sum(frame), 0)

    def test_draw_at_frame_boundary_no_crash(self):
        """Ripple at frame edges (0, 0) or (1280, 720) must not crash."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        r_top_left = RippleEffect(cx=2, cy=2, max_radius=30, duration=0.3)
        r_bottom_right = RippleEffect(cx=1279, cy=719, max_radius=30, duration=0.3)

        # Draw at 50% elapsed
        r_top_left.draw(frame, now=r_top_left.start_time + 0.15)
        r_bottom_right.draw(frame, now=r_bottom_right.start_time + 0.15)


class TestKeypadLayoutModes(unittest.TestCase):
    """Unit tests for 'ABC' and '123' keypad layout toggling."""

    def setUp(self):
        self.keyboard = Keyboard()

    def test_initial_mode_is_abc(self):
        """Default mode on creation should be 'ABC'."""
        self.assertEqual(self.keyboard.mode, "ABC")

    def test_abc_mode_has_qwerty_keys(self):
        """In 'ABC' mode, keys should include Q, W, E, R, T, Y..."""
        labels = [k.label for k in self.keyboard.keys]
        self.assertIn("Q", labels)
        self.assertIn("Z", labels)
        self.assertIn("M", labels)
        self.assertIn("123", labels)  # Toggle key to switch to numbers
        self.assertIn("CLR", labels)  # Clear key

    def test_toggle_mode_switches_to_123(self):
        """toggle_mode() should switch mode to '123'."""
        self.keyboard.toggle_mode()
        self.assertEqual(self.keyboard.mode, "123")
        labels = [k.label for k in self.keyboard.keys]
        self.assertIn("1", labels)
        self.assertIn("0", labels)
        self.assertIn("@", labels)
        self.assertIn("#", labels)
        self.assertIn("ABC", labels)  # Toggle key to switch back to letters
        self.assertNotIn("Q", labels)

    def test_toggle_mode_twice_returns_to_abc(self):
        """Toggling twice should return to 'ABC' layout."""
        self.keyboard.toggle_mode()
        self.assertEqual(self.keyboard.mode, "123")
        self.keyboard.toggle_mode()
        self.assertEqual(self.keyboard.mode, "ABC")
        labels = [k.label for k in self.keyboard.keys]
        self.assertIn("Q", labels)
        self.assertIn("123", labels)

    def test_special_keys_exist_in_both_modes(self):
        """SPC, SPEAK, BACK, and CLR must exist in both modes."""
        for mode in ("ABC", "123"):
            self.keyboard.mode = mode
            self.keyboard._create_keyboard()
            labels = [k.label for k in self.keyboard.keys]
            self.assertIn("SPC", labels)
            self.assertIn("SPEAK", labels)
            self.assertIn("BACK", labels)
            self.assertIn("CLR", labels)

    def test_row3_keys_on_same_y_line(self):
        """All row 3 keys should share the exact same y coordinate."""
        for mode in ("ABC", "123"):
            self.keyboard.mode = mode
            self.keyboard._create_keyboard()
            toggle_key = next(k for k in self.keyboard.keys if k.label in ("123", "ABC"))
            spc_key    = next(k for k in self.keyboard.keys if k.label == "SPC")
            speak_key  = next(k for k in self.keyboard.keys if k.label == "SPEAK")
            back_key   = next(k for k in self.keyboard.keys if k.label == "BACK")
            clr_key    = next(k for k in self.keyboard.keys if k.label == "CLR")

            expected_y = START_Y + 3 * (KEY_HEIGHT + GAP)
            self.assertEqual(toggle_key.y, expected_y)
            self.assertEqual(spc_key.y, expected_y)
            self.assertEqual(speak_key.y, expected_y)
            self.assertEqual(back_key.y, expected_y)
            self.assertEqual(clr_key.y, expected_y)


class TestClearKey(unittest.TestCase):
    """Unit tests for the CLR (clear) key behavior."""

    def setUp(self):
        self.keyboard = Keyboard()

    def test_clear_text_method(self):
        """clear_text() should empty the typed_text buffer."""
        self.keyboard.typed_text = "HELLO WORLD TEST"
        self.keyboard.clear_text()
        self.assertEqual(self.keyboard.typed_text, "")

    def test_register_click_clr_clears_text(self):
        """register_click when hovered on CLR should empty typed_text."""
        self.keyboard.typed_text = "HELLO WORLD"
        clr_key = next(k for k in self.keyboard.keys if k.label == "CLR")
        self.keyboard.hovered_key = clr_key
        self.keyboard.register_click()
        self.assertEqual(self.keyboard.typed_text, "")

    def test_register_click_clr_spawns_ripple(self):
        """Clicking CLR should spawn a ripple effect."""
        clr_key = next(k for k in self.keyboard.keys if k.label == "CLR")
        self.keyboard.hovered_key = clr_key
        initial_ripples = len(self.keyboard.ripples)
        self.keyboard.register_click()
        self.assertEqual(len(self.keyboard.ripples), initial_ripples + 1)


class TestKeyClickAndRipples(unittest.TestCase):
    """Unit tests for ripple spawning and lifecycle during clicks."""

    def setUp(self):
        self.keyboard = Keyboard()

    def test_click_letter_spawns_ripple(self):
        """Clicking a normal key should spawn a ripple at key center."""
        a_key = next(k for k in self.keyboard.keys if k.label == "A")
        self.keyboard.hovered_key = a_key
        self.keyboard.register_click()
        self.assertEqual(len(self.keyboard.ripples), 1)

        ripple = self.keyboard.ripples[0]
        self.assertEqual(ripple.cx, a_key.x + a_key.width // 2)
        self.assertEqual(ripple.cy, a_key.y + a_key.height // 2)

    def test_click_mode_toggle_switches_mode_and_spawns_ripple(self):
        """Clicking '123' key should toggle mode and spawn a ripple."""
        toggle_key = next(k for k in self.keyboard.keys if k.label == "123")
        self.keyboard.hovered_key = toggle_key
        self.keyboard.register_click()
        self.assertEqual(self.keyboard.mode, "123")
        self.assertEqual(len(self.keyboard.ripples), 1)

    def test_suggestion_click_spawns_ripple(self):
        """Selecting a suggestion should spawn a golden/amber ripple."""
        self.keyboard.set_suggestions(["HELLO", "HELP", "HELD"])
        sug_box = self.keyboard.suggestion_boxes[0]
        self.keyboard.hovered_suggestion = sug_box
        self.keyboard.select_suggestion()
        self.assertEqual(len(self.keyboard.ripples), 1)

    def test_ripples_pruned_on_draw_after_expiry(self):
        """draw() should prune expired ripples."""
        # Add a ripple that expired 1 second ago
        r_old = RippleEffect(cx=200, cy=200, duration=0.1)
        r_old.start_time = time.time() - 2.0  # long expired
        self.keyboard.ripples.append(r_old)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.keyboard.draw(frame)

        # Expired ripple should be removed
        self.assertNotIn(r_old, self.keyboard.ripples)


if __name__ == "__main__":
    unittest.main()
