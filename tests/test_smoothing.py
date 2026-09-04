"""
test_smoothing.py — Sprint 10: Landmark Smoothing (EMA) Tests

Tests cover:
  1. EMA convergence behavior.
  2. First-frame returns raw value (no lag).
  3. Jittery input produces smoother output.
  4. Multiple landmark IDs tracked independently.
  5. Reset clears history.
  6. Alpha parameter validation.
  7. Mathematical correctness of EMA formula.
"""

import sys
import os
import unittest

# Add the src/ directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from smoothing import LandmarkSmoother


class TestEMAFirstFrame(unittest.TestCase):
    """First-frame behavior: no history → return raw value."""

    def test_first_frame_returns_raw(self):
        """The very first update should return the exact raw coordinates."""
        smoother = LandmarkSmoother(alpha=0.45)
        result = smoother.update(8, 100, 200)
        self.assertEqual(result, (100, 200))

    def test_first_frame_different_landmarks(self):
        """Each landmark's first frame should return raw independently."""
        smoother = LandmarkSmoother(alpha=0.5)
        r1 = smoother.update(8, 100, 200)   # index finger tip
        r2 = smoother.update(4, 300, 400)   # thumb tip
        self.assertEqual(r1, (100, 200))
        self.assertEqual(r2, (300, 400))


class TestEMASmoothing(unittest.TestCase):
    """Verify EMA produces smoother output from jittery input."""

    def test_constant_input_converges(self):
        """Constant input should converge to the constant value."""
        smoother = LandmarkSmoother(alpha=0.45)
        # Send 100 frames of constant position
        for _ in range(100):
            result = smoother.update(8, 500, 300)
        self.assertEqual(result, (500, 300))

    def test_jitter_reduction(self):
        """
        EMA should reduce output variance compared to input variance.
        Jittery input oscillating ±5px should produce smoother output.
        """
        smoother = LandmarkSmoother(alpha=0.45)
        base_x, base_y = 500, 300
        jitter = [0, 5, -3, 4, -5, 2, -4, 3, -2, 5, -3, 4, -5, 2, -1]

        raw_outputs = []
        smooth_outputs = []

        for j in jitter:
            raw_x = base_x + j
            raw_y = base_y + j
            raw_outputs.append(raw_x)
            sx, sy = smoother.update(8, raw_x, raw_y)
            smooth_outputs.append(sx)

        # Calculate variance of raw vs smooth
        raw_mean = sum(raw_outputs) / len(raw_outputs)
        smooth_mean = sum(smooth_outputs) / len(smooth_outputs)
        raw_var = sum((x - raw_mean) ** 2 for x in raw_outputs) / len(raw_outputs)
        smooth_var = sum((x - smooth_mean) ** 2 for x in smooth_outputs) / len(smooth_outputs)

        # Smooth output should have LOWER variance than raw input
        self.assertLess(smooth_var, raw_var,
                        f"Smoothing should reduce variance: raw={raw_var:.2f}, smooth={smooth_var:.2f}")

    def test_step_response_convergence(self):
        """After a step change, EMA should converge within ~10 frames."""
        smoother = LandmarkSmoother(alpha=0.45)

        # Stabilize at position (100, 100)
        for _ in range(20):
            smoother.update(8, 100, 100)

        # Step change to (200, 200) — should converge within ~10 frames
        for _ in range(20):
            result = smoother.update(8, 200, 200)

        # After 20 frames at α=0.45, should be very close to (200, 200)
        # (1-0.45)^20 ≈ 0.0001 — virtually zero residual
        self.assertAlmostEqual(result[0], 200, delta=1)
        self.assertAlmostEqual(result[1], 200, delta=1)


class TestEMAFormula(unittest.TestCase):
    """Verify the mathematical correctness of the EMA formula."""

    def test_alpha_1_no_smoothing(self):
        """α = 1.0 should produce raw values (no smoothing)."""
        smoother = LandmarkSmoother(alpha=1.0)
        smoother.update(8, 100, 100)   # first frame

        # Second frame with completely different value
        result = smoother.update(8, 200, 200)
        self.assertEqual(result, (200, 200))

    def test_manual_ema_calculation(self):
        """Verify output matches manual EMA computation."""
        alpha = 0.5
        smoother = LandmarkSmoother(alpha=alpha)

        # Frame 1: x=100 → smoothed = 100 (first frame)
        r1 = smoother.update(8, 100, 100)
        self.assertEqual(r1, (100, 100))

        # Frame 2: x=200 → smoothed = 0.5 * 200 + 0.5 * 100 = 150
        r2 = smoother.update(8, 200, 200)
        self.assertEqual(r2, (150, 150))

        # Frame 3: x=200 → smoothed = 0.5 * 200 + 0.5 * 150 = 175
        r3 = smoother.update(8, 200, 200)
        self.assertEqual(r3, (175, 175))


class TestIndependentLandmarks(unittest.TestCase):
    """Each landmark ID should be tracked independently."""

    def test_independent_tracking(self):
        """Smoothing one landmark should not affect another."""
        smoother = LandmarkSmoother(alpha=0.5)

        # Initialize both landmarks
        smoother.update(8, 100, 100)   # index finger
        smoother.update(4, 500, 500)   # thumb

        # Move index finger only
        r_index = smoother.update(8, 200, 200)
        r_thumb = smoother.update(4, 500, 500)  # thumb stays put

        # Index should move toward 200
        self.assertEqual(r_index, (150, 150))
        # Thumb should stay at 500 (unchanged input)
        self.assertEqual(r_thumb, (500, 500))

    def test_tracked_landmarks_list(self):
        """tracked_landmarks should list all tracked landmark IDs."""
        smoother = LandmarkSmoother(alpha=0.5)
        smoother.update(8, 100, 100)
        smoother.update(4, 200, 200)
        smoother.update(0, 300, 300)

        tracked = set(smoother.tracked_landmarks)
        self.assertEqual(tracked, {8, 4, 0})


class TestReset(unittest.TestCase):
    """Verify reset clears smoothing history."""

    def test_reset_single_landmark(self):
        """Resetting a specific landmark should clear only that one."""
        smoother = LandmarkSmoother(alpha=0.5)
        smoother.update(8, 100, 100)
        smoother.update(4, 200, 200)

        smoother.reset(8)

        self.assertNotIn(8, smoother.tracked_landmarks)
        self.assertIn(4, smoother.tracked_landmarks)

    def test_reset_all(self):
        """Resetting all landmarks should clear everything."""
        smoother = LandmarkSmoother(alpha=0.5)
        smoother.update(8, 100, 100)
        smoother.update(4, 200, 200)

        smoother.reset()

        self.assertEqual(len(smoother.tracked_landmarks), 0)

    def test_after_reset_returns_raw(self):
        """After reset, the next update should return raw (like first frame)."""
        smoother = LandmarkSmoother(alpha=0.5)
        smoother.update(8, 100, 100)
        smoother.update(8, 200, 200)  # smoothed = 150

        smoother.reset(8)

        result = smoother.update(8, 300, 300)
        self.assertEqual(result, (300, 300))  # raw value, no history


class TestAlphaValidation(unittest.TestCase):
    """Verify alpha parameter validation."""

    def test_alpha_zero_raises(self):
        """α = 0.0 should raise ValueError (output would never change)."""
        with self.assertRaises(ValueError):
            LandmarkSmoother(alpha=0.0)

    def test_alpha_negative_raises(self):
        """Negative α should raise ValueError."""
        with self.assertRaises(ValueError):
            LandmarkSmoother(alpha=-0.5)

    def test_alpha_above_one_raises(self):
        """α > 1.0 should raise ValueError."""
        with self.assertRaises(ValueError):
            LandmarkSmoother(alpha=1.5)

    def test_alpha_one_valid(self):
        """α = 1.0 should be valid (no smoothing = raw pass-through)."""
        smoother = LandmarkSmoother(alpha=1.0)
        self.assertEqual(smoother.alpha, 1.0)

    def test_alpha_small_valid(self):
        """Small positive α should be valid (heavy smoothing)."""
        smoother = LandmarkSmoother(alpha=0.01)
        self.assertEqual(smoother.alpha, 0.01)


if __name__ == "__main__":
    unittest.main()
