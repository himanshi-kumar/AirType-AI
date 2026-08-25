"""
tests/test_spelling.py — Sprint 5: Auto-Correct Unit Tests

Tests the new spelling correction features added in Sprint 5:
    - QWERTY-weighted substitution cost
    - Damerau-Levenshtein distance (with transposition)
    - Fuzzy suggestion fallback in _predict_completions
    - Auto-correct on space via get_autocorrect()
    - Full pipeline: keyboard.register_click() with predictor

RUN WITH:
    .venv/bin/python -m pytest tests/test_spelling.py -v
    OR
    .venv/bin/python -m unittest tests.test_spelling -v
"""

import sys
import os
import unittest

# Add src/ to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prediction import (
    WordPredictor,
    get_substitution_cost,
    damerau_levenshtein_distance,
    KEY_CENTERS,
)
from keyboard import Keyboard


class TestSubstitutionCost(unittest.TestCase):
    """Test QWERTY-aware substitution cost function."""

    def test_identical_chars_cost_zero(self):
        """Same character → cost = 0."""
        self.assertEqual(get_substitution_cost("A", "A"), 0.0)
        self.assertEqual(get_substitution_cost("Z", "Z"), 0.0)

    def test_adjacent_keys_low_cost(self):
        """Adjacent keys should have low substitution cost (< 1.0)."""
        cost = get_substitution_cost("W", "E")
        self.assertGreater(cost, 0.0)
        self.assertLess(cost, 1.0)

    def test_distant_keys_high_cost(self):
        """Distant keys should have higher cost than adjacent keys."""
        adjacent_cost = get_substitution_cost("W", "E")
        distant_cost = get_substitution_cost("Q", "M")
        self.assertGreater(distant_cost, adjacent_cost)

    def test_unknown_char_default_cost(self):
        """Characters not in QWERTY layout → cost = 1.0."""
        self.assertEqual(get_substitution_cost("1", "A"), 1.0)

    def test_all_keys_have_centers(self):
        """Every QWERTY letter should have coordinates."""
        for letter in "QWERTYUIOPASDFGHJKLZXCVBNM":
            self.assertIn(letter, KEY_CENTERS)


class TestDamerauLevenshteinDistance(unittest.TestCase):
    """Test QWERTY-weighted Damerau-Levenshtein distance."""

    def test_identical_strings(self):
        """Same string → distance = 0."""
        self.assertEqual(damerau_levenshtein_distance("HELLO", "HELLO"), 0.0)

    def test_single_insertion(self):
        """One insertion → distance = 1.0."""
        self.assertEqual(damerau_levenshtein_distance("HELO", "HELLO"), 1.0)

    def test_single_deletion(self):
        """One deletion → distance = 1.0."""
        self.assertEqual(damerau_levenshtein_distance("HELLO", "HELO"), 1.0)

    def test_adjacent_substitution_low_cost(self):
        """Substituting adjacent key (W→E) costs less than 1.0."""
        dist = damerau_levenshtein_distance("THW", "THE")
        self.assertGreater(dist, 0.0)
        self.assertLess(dist, 1.0)

    def test_transposition_detected(self):
        """Swapping adjacent chars should use transposition (cost = 1.0).
        Note: 'AB' → 'BA' would add a first-letter mismatch penalty (+1.5),
        so we use 'ACB' → 'ABC' which keeps 'A' as first letter."""
        dist = damerau_levenshtein_distance("ACB", "ABC")
        self.assertEqual(dist, 1.0)

    def test_hpap_to_happy(self):
        """HPAP → HAPPY: transposition + insertion = 2.0."""
        dist = damerau_levenshtein_distance("HPAP", "HAPPY")
        self.assertEqual(dist, 2.0)

    def test_first_letter_penalty(self):
        """Different first letter adds 1.5 penalty."""
        dist_same_first = damerau_levenshtein_distance("THE", "THAT")
        dist_diff_first = damerau_levenshtein_distance("ZHE", "THE")
        self.assertGreater(dist_diff_first, dist_same_first)

    def test_empty_strings(self):
        """Empty → non-empty should equal length."""
        self.assertEqual(damerau_levenshtein_distance("", "ABC"), 3.0)
        self.assertEqual(damerau_levenshtein_distance("ABC", ""), 3.0)
        self.assertEqual(damerau_levenshtein_distance("", ""), 0.0)


class TestGetAutocorrect(unittest.TestCase):
    """Test the get_autocorrect method of WordPredictor."""

    def setUp(self):
        self.predictor = WordPredictor()

    def test_valid_word_returns_none(self):
        """A word already in the vocabulary should return None (no correction)."""
        self.assertIsNone(self.predictor.get_autocorrect("HELLO"))
        self.assertIsNone(self.predictor.get_autocorrect("THE"))

    def test_empty_returns_none(self):
        """Empty string should return None."""
        self.assertIsNone(self.predictor.get_autocorrect(""))

    def test_thw_corrects_to_the(self):
        """THW (adjacent key typo) → THE."""
        result = self.predictor.get_autocorrect("THW")
        self.assertEqual(result, "THE")

    def test_hapy_corrects_to_happy(self):
        """HAPY (missing letter) → HAPPY."""
        result = self.predictor.get_autocorrect("HAPY")
        self.assertEqual(result, "HAPPY")

    def test_hpap_corrects_to_happy(self):
        """HPAP (transposition + missing) → HAPPY."""
        result = self.predictor.get_autocorrect("HPAP")
        self.assertEqual(result, "HAPPY")

    def test_computw_corrects_to_computer(self):
        """COMPUTW (adjacent key typo) → COMPUTER."""
        result = self.predictor.get_autocorrect("COMPUTW")
        self.assertEqual(result, "COMPUTER")

    def test_gibberish_returns_none(self):
        """Completely unrecognizable input should return None."""
        result = self.predictor.get_autocorrect("XZQWJ")
        self.assertIsNone(result)


class TestFuzzySuggestions(unittest.TestCase):
    """Test that _predict_completions falls back to fuzzy matching."""

    def setUp(self):
        self.predictor = WordPredictor()

    def test_thw_suggests_the(self):
        """THW has no prefix matches → fuzzy should suggest THE."""
        suggestions = self.predictor.get_suggestions("THW")
        self.assertIn("THE", suggestions)

    def test_hapy_suggests_happy(self):
        """HAPY prefix matches nothing exactly → fuzzy finds HAPPY."""
        suggestions = self.predictor.get_suggestions("HAPY")
        self.assertIn("HAPPY", suggestions)

    def test_prefix_match_still_works(self):
        """Normal prefix matching should still work as before."""
        suggestions = self.predictor.get_suggestions("COM")
        self.assertIn("COME", suggestions)

    def test_exact_prefix_returns_completions(self):
        """HA should return prefix matches like HAVE, HAPPY, etc."""
        suggestions = self.predictor.get_suggestions("HA")
        self.assertTrue(len(suggestions) > 0)
        for word in suggestions:
            self.assertTrue(word.startswith("HA"))


class TestKeyboardAutocorrect(unittest.TestCase):
    """Test the full auto-correct pipeline via keyboard.register_click()."""

    def setUp(self):
        self.keyboard = Keyboard()
        self.predictor = WordPredictor()

    def _simulate_spc_press(self):
        """Simulate hovering over SPC and calling register_click."""
        # Find the SPC key
        for key in self.keyboard.keys:
            if key.label == "SPC":
                self.keyboard.hovered_key = key
                break
        self.keyboard.register_click(self.predictor)

    def test_thw_space_corrects_to_the(self):
        """Typing 'THW' + SPC → 'THE '."""
        self.keyboard.typed_text = "THW"
        self._simulate_spc_press()
        self.assertEqual(self.keyboard.typed_text, "THE ")

    def test_valid_word_not_changed(self):
        """Typing 'HELLO' + SPC → 'HELLO ' (no correction)."""
        self.keyboard.typed_text = "HELLO"
        self._simulate_spc_press()
        self.assertEqual(self.keyboard.typed_text, "HELLO ")

    def test_multiword_correction(self):
        """Typing 'I AM THW' + SPC → 'I AM THE '."""
        self.keyboard.typed_text = "I AM THW"
        self._simulate_spc_press()
        self.assertEqual(self.keyboard.typed_text, "I AM THE ")

    def test_double_space_no_crash(self):
        """Typing 'HELLO ' + SPC → 'HELLO  ' (empty last word, no crash)."""
        self.keyboard.typed_text = "HELLO "
        self._simulate_spc_press()
        self.assertEqual(self.keyboard.typed_text, "HELLO  ")

    def test_empty_text_space(self):
        """Empty text + SPC → ' ' (just a space, no crash)."""
        self.keyboard.typed_text = ""
        self._simulate_spc_press()
        self.assertEqual(self.keyboard.typed_text, " ")

    def test_register_click_without_predictor(self):
        """register_click(None) should work like Sprint 4 (no crash)."""
        self.keyboard.typed_text = "THW"
        for key in self.keyboard.keys:
            if key.label == "SPC":
                self.keyboard.hovered_key = key
                break
        self.keyboard.register_click()  # no predictor passed
        self.assertEqual(self.keyboard.typed_text, "THW ")


if __name__ == "__main__":
    unittest.main()
""", "Description": "Unit tests covering every aspect of Sprint 5: substitution cost, DL distance, autocorrect, fuzzy suggestions, and keyboard integration", "Overwrite": false, "TargetFile": "/Users/himanshi/Desktop/AirType-AI/tests/test_spelling.py
"""
