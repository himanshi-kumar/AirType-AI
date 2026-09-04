"""
tests/test_audio.py — Sprint 7: Sound Feedback Unit Tests

Tests the SoundPlayer class from audio.py:
    - Initialization and pre-synthesis
    - Sine wave array validity
    - Sweep array validity
    - Chord array validity
    - Fade-out envelope application
    - All playback methods are callable without errors

RUN WITH:
    .venv/bin/python -m unittest tests.test_audio -v
"""

import sys
import os
import unittest
import numpy as np

# Add src/ to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from audio import SoundPlayer, SAMPLE_RATE


class TestSoundPlayerInit(unittest.TestCase):
    """Test SoundPlayer initializes and pre-synthesizes all sounds correctly."""

    def setUp(self):
        self.player = SoundPlayer(volume=0.4)

    def test_init_no_error(self):
        """SoundPlayer() should initialize without raising."""
        player = SoundPlayer()
        self.assertIsNotNone(player)

    def test_all_sounds_pre_synthesized(self):
        """All 7 sound arrays should exist after __init__."""
        self.assertIsNotNone(self.player._keypress_sound)
        self.assertIsNotNone(self.player._space_sound)
        self.assertIsNotNone(self.player._backspace_sound)
        self.assertIsNotNone(self.player._suggestion_sound)
        self.assertIsNotNone(self.player._speak_sound)
        self.assertIsNotNone(self.player._clear_sound)
        self.assertIsNotNone(self.player._mode_sound)

    def test_all_sounds_are_numpy_arrays(self):
        """All pre-synthesized sounds should be np.ndarray instances."""
        sounds = [
            self.player._keypress_sound,
            self.player._space_sound,
            self.player._backspace_sound,
            self.player._suggestion_sound,
            self.player._speak_sound,
            self.player._clear_sound,
            self.player._mode_sound,
        ]
        for s in sounds:
            self.assertIsInstance(s, np.ndarray)

    def test_all_sounds_are_float32(self):
        """All sounds should be float32 (required by sounddevice)."""
        sounds = [
            self.player._keypress_sound,
            self.player._space_sound,
            self.player._backspace_sound,
            self.player._suggestion_sound,
            self.player._speak_sound,
            self.player._clear_sound,
            self.player._mode_sound,
        ]
        for s in sounds:
            self.assertEqual(s.dtype, np.float32)


class TestSoundSynthesis(unittest.TestCase):
    """Test the internal synthesis methods produce valid audio arrays."""

    def setUp(self):
        self.player = SoundPlayer(volume=0.5)

    def test_sine_correct_length(self):
        """Sine wave should have exactly SAMPLE_RATE * duration samples."""
        duration = 0.05
        wave = self.player._sine(freq=440, duration=duration)
        expected_len = int(SAMPLE_RATE * duration)
        self.assertEqual(len(wave), expected_len)

    def test_sine_within_amplitude(self):
        """Sine wave amplitude should not exceed the set volume."""
        wave = self.player._sine(freq=440, duration=0.05)
        self.assertLessEqual(float(np.max(np.abs(wave))), 0.5 + 1e-5)

    def test_sweep_correct_length(self):
        """Frequency sweep should have the correct number of samples."""
        duration = 0.07
        wave = self.player._sweep(330, 165, duration)
        expected_len = int(SAMPLE_RATE * duration)
        self.assertEqual(len(wave), expected_len)

    def test_sweep_within_amplitude(self):
        """Sweep amplitude should not exceed the set volume."""
        wave = self.player._sweep(523, 784, 0.12)
        self.assertLessEqual(float(np.max(np.abs(wave))), 0.5 + 1e-5)

    def test_chord_correct_length(self):
        """Chord should have the correct number of samples."""
        duration = 0.15
        wave = self.player._chord([440, 550], duration)
        expected_len = int(SAMPLE_RATE * duration)
        self.assertEqual(len(wave), expected_len)

    def test_chord_within_amplitude(self):
        """Chord amplitude (normalized) should not exceed the set volume."""
        wave = self.player._chord([440, 550, 660], 0.1)
        self.assertLessEqual(float(np.max(np.abs(wave))), 0.5 + 1e-5)

    def test_fade_reduces_tail(self):
        """Fade-out should make the last sample close to zero."""
        # Constant signal: all samples = 1.0
        constant = np.ones(SAMPLE_RATE, dtype=np.float32)
        faded = self.player._apply_fade(constant, fade_ms=10.0)
        # Last sample should be near zero after fade
        self.assertAlmostEqual(float(faded[-1]), 0.0, places=3)
        # First sample should be unchanged
        self.assertAlmostEqual(float(faded[0]), 1.0, places=3)

    def test_fade_preserves_length(self):
        """Fade should not change the length of the array."""
        wave = self.player._sine(440, 0.1)
        faded = self.player._apply_fade(wave)
        self.assertEqual(len(wave), len(faded))


class TestSoundPlayerPlayback(unittest.TestCase):
    """Test that all public playback methods run without raising errors."""

    def setUp(self):
        self.player = SoundPlayer(volume=0.01)  # very quiet for CI tests

    def test_play_keypress_no_error(self):
        """play_keypress() should not raise."""
        self.player.play_keypress()

    def test_play_space_no_error(self):
        """play_space() should not raise."""
        self.player.play_space()

    def test_play_backspace_no_error(self):
        """play_backspace() should not raise."""
        self.player.play_backspace()

    def test_play_suggestion_no_error(self):
        """play_suggestion() should not raise."""
        self.player.play_suggestion()

    def test_play_speak_no_error(self):
        """play_speak() should not raise."""
        self.player.play_speak()

    def test_play_clear_no_error(self):
        """play_clear() should not raise."""
        self.player.play_clear()

    def test_play_mode_switch_no_error(self):
        """play_mode_switch() should not raise."""
        self.player.play_mode_switch()


if __name__ == "__main__":
    unittest.main()
