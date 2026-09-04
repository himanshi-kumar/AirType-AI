"""
test_multihand.py — Sprint 11: Multi-Hand Typing Tests

Tests cover:
  1. MultiPinchDetector independent per-hand cooldowns.
  2. MultiPinchDetector lazy detector creation.
  3. tick_inactive cleanup for disappeared hands.
  4. Keyboard multi-finger drawing (finger_positions parameter).
  5. get_all_hands API contract expectations.
"""

import sys
import os
import unittest

# Add the src/ directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gesture import PinchDetector, MultiPinchDetector, CLICK_COOLDOWN_FRAMES


class TestMultiPinchDetectorCreation(unittest.TestCase):
    """Verify MultiPinchDetector creates per-hand detectors lazily."""

    def test_empty_initially(self):
        """No detectors should exist before any update() call."""
        mpd = MultiPinchDetector()
        self.assertEqual(len(mpd._detectors), 0)

    def test_creates_detector_on_first_update(self):
        """First update for a hand should create its detector."""
        mpd = MultiPinchDetector()
        mpd.update(0, (100, 100), (200, 200))  # hand 0, no pinch
        self.assertIn(0, mpd._detectors)

    def test_creates_separate_detectors(self):
        """Each hand index gets its own PinchDetector."""
        mpd = MultiPinchDetector()
        mpd.update(0, (100, 100), (200, 200))
        mpd.update(1, (300, 300), (400, 400))
        self.assertIn(0, mpd._detectors)
        self.assertIn(1, mpd._detectors)
        self.assertIsNot(mpd._detectors[0], mpd._detectors[1])


class TestMultiPinchIndependentCooldowns(unittest.TestCase):
    """Each hand should have independent rising-edge and cooldown state."""

    def test_hand0_cooldown_doesnt_block_hand1(self):
        """
        Hand 0 fires a click (enters cooldown).
        Hand 1 should still be able to fire immediately.
        """
        mpd = MultiPinchDetector()

        # Hand 0: pinch (close fingers)
        click0 = mpd.update(0, (100, 100), (105, 105))  # within threshold
        self.assertTrue(click0, "Hand 0 should click on first pinch")

        # Hand 0 is now in cooldown
        self.assertGreater(mpd._detectors[0].cooldown, 0)

        # Hand 1: pinch (close fingers) — should NOT be blocked by hand 0
        click1 = mpd.update(1, (300, 300), (305, 305))
        self.assertTrue(click1, "Hand 1 should click independently of hand 0's cooldown")

    def test_same_hand_respects_cooldown(self):
        """Same hand should NOT fire again during cooldown."""
        mpd = MultiPinchDetector()

        # First pinch
        click1 = mpd.update(0, (100, 100), (105, 105))
        self.assertTrue(click1)

        # Release
        mpd.update(0, (100, 100), (200, 200))  # far apart

        # Try to pinch again immediately — should be blocked by cooldown
        click2 = mpd.update(0, (100, 100), (105, 105))
        self.assertFalse(click2, "Same hand should be blocked during cooldown")


class TestMultiPinchRisingEdge(unittest.TestCase):
    """Rising edge detection per hand."""

    def test_holding_pinch_doesnt_repeat(self):
        """Holding a pinch should only fire once (rising edge)."""
        mpd = MultiPinchDetector()

        # Frame 1: Start pinch
        click1 = mpd.update(0, (100, 100), (105, 105))
        self.assertTrue(click1)

        # Frame 2-5: Still pinching — should NOT fire again
        for _ in range(5):
            click = mpd.update(0, (100, 100), (105, 105))
            self.assertFalse(click, "Held pinch should not re-fire")


class TestTickInactive(unittest.TestCase):
    """Test tick_inactive cleanup for disappeared hands."""

    def test_inactive_hand_cooldown_decrements(self):
        """Hands not in the frame should still have cooldowns decremented."""
        mpd = MultiPinchDetector()

        # Fire hand 0
        mpd.update(0, (100, 100), (105, 105))
        initial_cooldown = mpd._detectors[0].cooldown

        # Hand 0 disappears — tick with empty active set
        mpd.tick_inactive(set())

        # Cooldown should have decremented
        if 0 in mpd._detectors:
            self.assertLess(mpd._detectors[0].cooldown, initial_cooldown)

    def test_is_any_pinching(self):
        """is_any_pinching should report True when any hand is pinching."""
        mpd = MultiPinchDetector()

        # No hands — not pinching
        self.assertFalse(mpd.is_any_pinching())

        # Hand 0 pinching
        mpd.update(0, (100, 100), (105, 105))
        self.assertTrue(mpd.is_any_pinching())


class TestKeyboardMultiFingerDraw(unittest.TestCase):
    """Verify Keyboard.draw() accepts multiple finger positions."""

    def test_draw_with_finger_positions_list(self):
        """draw() should accept finger_positions kwarg without error."""
        import numpy as np
        from keyboard import Keyboard

        kb = Keyboard()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Should not raise with a list of finger positions
        kb.draw(
            frame,
            finger_positions=[(300, 500), (800, 500)],
            fps=30.0,
        )

    def test_draw_with_empty_finger_list(self):
        """draw() with empty finger_positions should not crash."""
        import numpy as np
        from keyboard import Keyboard

        kb = Keyboard()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        kb.draw(frame, finger_positions=[], fps=25.0)

    def test_draw_backward_compat(self):
        """draw() with single finger_pos should still work."""
        import numpy as np
        from keyboard import Keyboard

        kb = Keyboard()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        kb.draw(frame, finger_pos=(500, 500))


class TestFPSDisplay(unittest.TestCase):
    """Verify FPS parameter is accepted by draw()."""

    def test_fps_none_works(self):
        """fps=None should not crash."""
        import numpy as np
        from keyboard import Keyboard

        kb = Keyboard()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        kb.draw(frame, fps=None)

    def test_fps_value_works(self):
        """fps with a float value should display without error."""
        import numpy as np
        from keyboard import Keyboard

        kb = Keyboard()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        kb.draw(frame, fps=29.7)


if __name__ == "__main__":
    unittest.main()
