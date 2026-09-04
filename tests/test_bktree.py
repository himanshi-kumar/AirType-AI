"""
test_bktree.py — Sprint 9: BK-Tree, Bayesian Ranking & Optimized Autocorrect Tests

Tests cover:
  1. BK-Tree construction and search correctness.
  2. BK-Tree search matches brute-force results (correctness guarantee).
  3. Length pre-filtering.
  4. Bayesian frequency-weighted ranking (tie-breaking).
  5. Expanded vocabulary integration.
  6. Backward compatibility with existing autocorrect/prediction API.
"""

import math
import sys
import os
import unittest

# Add the src/ directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prediction import (
    BKTree,
    BKTreeNode,
    WordPredictor,
    damerau_levenshtein_distance,
    WORD_FREQUENCIES,
    BAYESIAN_ALPHA,
)


class TestBKTreeConstruction(unittest.TestCase):
    """Verify BK-Tree builds correctly from vocabulary."""

    def test_empty_tree(self):
        """An empty BK-Tree should have no root and size 0."""
        tree = BKTree()
        self.assertIsNone(tree.root)
        self.assertEqual(tree.size, 0)

    def test_single_insert(self):
        """Inserting one word sets root and size = 1."""
        tree = BKTree()
        tree.insert("HELLO")
        self.assertIsNotNone(tree.root)
        self.assertEqual(tree.root.word, "HELLO")
        self.assertEqual(tree.size, 1)

    def test_multiple_inserts(self):
        """Insert multiple words and verify size."""
        tree = BKTree()
        words = ["HELLO", "HELP", "WORLD", "WORD", "WORK"]
        for w in words:
            tree.insert(w)
        self.assertEqual(tree.size, len(words))

    def test_from_vocabulary(self):
        """Build from a vocabulary dict and verify all words are indexed."""
        vocab = {"THE": 100, "THEY": 71, "THEM": 40, "THEN": 39, "THAN": 30}
        tree = BKTree.from_vocabulary(vocab)
        self.assertEqual(tree.size, len(vocab))
        # Root should be the highest-frequency word (inserted first)
        self.assertEqual(tree.root.word, "THE")

    def test_from_full_vocabulary(self):
        """Build from the full ~2,500 word vocabulary without errors."""
        tree = BKTree.from_vocabulary(WORD_FREQUENCIES)
        self.assertGreater(tree.size, 500)
        # Root should be "THE" (highest frequency = 100)
        self.assertEqual(tree.root.word, "THE")


class TestBKTreeSearch(unittest.TestCase):
    """Verify BK-Tree search returns correct results."""

    def setUp(self):
        """Build a small tree for testing."""
        self.vocab = {
            "THE": 100, "THEY": 71, "THEM": 40, "THEN": 39,
            "THAN": 30, "THIS": 76, "THAT": 89, "THOSE": 25,
            "THESE": 38, "THINK": 51, "THREE": 20,
        }
        self.tree = BKTree.from_vocabulary(self.vocab)

    def test_exact_match(self):
        """Searching for an exact word should find it at distance 0."""
        results = self.tree.search("THE", 0.0)
        words = [w for w, _ in results]
        self.assertIn("THE", words)

    def test_close_matches(self):
        """Searching with small max_dist should find close words."""
        results = self.tree.search("THE", 2.0)
        words = [w for w, _ in results]
        # THE itself should be found (distance 0)
        self.assertIn("THE", words)
        # At least some close matches should be found
        self.assertGreater(len(words), 1)

    def test_no_results_for_tight_threshold(self):
        """Searching with max_dist=0 for a non-vocabulary word → empty."""
        results = self.tree.search("ZZZZZ", 0.0)
        self.assertEqual(len(results), 0)

    def test_empty_tree_search(self):
        """Searching an empty tree returns empty list."""
        tree = BKTree()
        results = tree.search("HELLO", 5.0)
        self.assertEqual(results, [])

    def test_all_results_within_threshold(self):
        """Every result should have distance <= max_dist."""
        max_dist = 3.0
        results = self.tree.search("THW", max_dist)
        for word, dist in results:
            self.assertLessEqual(
                dist, max_dist,
                f"Word '{word}' has distance {dist} > max_dist {max_dist}"
            )


class TestBKTreeMatchesBruteForce(unittest.TestCase):
    """
    Critical correctness test: BK-Tree search must return the SAME results
    as brute-force scanning every word. The BK-Tree is an optimization —
    it must not miss any valid matches.
    """

    def setUp(self):
        """Build a tree from a medium-sized vocabulary for testing."""
        # Use a subset of the real vocabulary for faster tests
        self.vocab = dict(list(WORD_FREQUENCIES.items())[:200])
        self.tree = BKTree.from_vocabulary(self.vocab)

    def _brute_force_search(self, query, max_dist):
        """Reference implementation: check every word."""
        results = []
        for word in self.vocab:
            dist = damerau_levenshtein_distance(query, word)
            if dist <= max_dist:
                results.append((word, dist))
        return results

    def test_brute_force_match_thw(self):
        """BK-Tree results for 'THW' should match brute force."""
        query, max_dist = "THW", 2.0
        bk_results = set(w for w, _ in self.tree.search(query, max_dist))
        bf_results = set(w for w, _ in self._brute_force_search(query, max_dist))
        self.assertEqual(bk_results, bf_results,
                         f"BK-Tree missed or added words for '{query}'")

    def test_brute_force_match_computw(self):
        """BK-Tree results for 'COMPUTW' should match brute force."""
        query, max_dist = "COMPUTW", 3.0
        bk_results = set(w for w, _ in self.tree.search(query, max_dist))
        bf_results = set(w for w, _ in self._brute_force_search(query, max_dist))
        self.assertEqual(bk_results, bf_results,
                         f"BK-Tree missed or added words for '{query}'")

    def test_brute_force_match_hapy(self):
        """BK-Tree results for 'HAPY' should match brute force."""
        query, max_dist = "HAPY", 2.5
        bk_results = set(w for w, _ in self.tree.search(query, max_dist))
        bf_results = set(w for w, _ in self._brute_force_search(query, max_dist))
        self.assertEqual(bk_results, bf_results,
                         f"BK-Tree missed or added words for '{query}'")


class TestBayesianRanking(unittest.TestCase):
    """Verify Bayesian frequency-weighted ranking breaks ties correctly."""

    def test_common_word_wins_tie(self):
        """When edit distances are equal, the more common word should win."""
        # Create a vocabulary where THE (freq=100) and THY (freq=5) exist
        vocab = {"THE": 100, "THY": 5, "HELLO": 50}
        predictor = WordPredictor(vocabulary=vocab)

        # "THW" is equidistant from THE and THY on QWERTY
        result = predictor.get_autocorrect("THW")
        # THE should win because it's far more frequent
        self.assertEqual(result, "THE")

    def test_bayesian_score_formula(self):
        """Verify the Bayesian scoring formula produces expected values."""
        freq_high = 100
        freq_low = 5
        dist = 1.0

        score_high = dist - BAYESIAN_ALPHA * math.log(freq_high + 1)
        score_low = dist - BAYESIAN_ALPHA * math.log(freq_low + 1)

        # Higher frequency word should have LOWER (better) score
        self.assertLess(score_high, score_low)

    def test_distance_overrides_frequency(self):
        """A much closer word should win even if it's less frequent."""
        vocab = {"HELLO": 10, "HELP": 50, "HELD": 30}
        predictor = WordPredictor(vocabulary=vocab)

        # "HELPP" → "HELP" should win (distance 1) over "HELLO" (distance > 1)
        result = predictor.get_autocorrect("HELPP")
        self.assertEqual(result, "HELP")


class TestLengthPreFiltering(unittest.TestCase):
    """Verify length pre-filtering works in autocorrect."""

    def test_short_word_doesnt_match_long(self):
        """A 2-char typo shouldn't correct to a 12-char word."""
        vocab = {"AT": 77, "APPRECIATION": 10, "AN": 65}
        predictor = WordPredictor(vocabulary=vocab)
        # "AX" (2 chars) should match "AT" or "AN", not "APPRECIATION"
        result = predictor.get_autocorrect("AX")
        self.assertIn(result, ["AT", "AN"])

    def test_length_buckets_populated(self):
        """Length buckets should contain all vocabulary words."""
        predictor = WordPredictor()
        total = sum(len(words) for words in predictor._length_buckets.values())
        self.assertEqual(total, len(predictor.vocabulary))


class TestExpandedVocabulary(unittest.TestCase):
    """Verify the expanded vocabulary has the expected size and content."""

    def test_vocabulary_size(self):
        """Vocabulary should have at least 500 words (expanded from ~300)."""
        self.assertGreater(len(WORD_FREQUENCIES), 500)

    def test_essential_words_present(self):
        """Core English words must be in the vocabulary."""
        essential = ["THE", "AND", "IS", "TO", "OF", "A", "IN", "THAT",
                     "HAVE", "FOR", "NOT", "WITH", "YOU", "HE", "SHE",
                     "GOOD", "BAD", "HAPPY", "HELLO", "COMPUTER"]
        for word in essential:
            self.assertIn(word, WORD_FREQUENCIES,
                          f"Essential word '{word}' missing from vocabulary")

    def test_all_words_uppercase(self):
        """Every vocabulary word must be uppercase."""
        for word in WORD_FREQUENCIES:
            self.assertEqual(word, word.upper(),
                             f"Word '{word}' is not uppercase")

    def test_all_frequencies_positive(self):
        """Every frequency score must be a positive integer."""
        for word, freq in WORD_FREQUENCIES.items():
            self.assertGreater(freq, 0,
                               f"Word '{word}' has non-positive frequency {freq}")


class TestAutocorrectBackwardCompat(unittest.TestCase):
    """Ensure Sprint 9 changes don't break existing autocorrect behavior."""

    def setUp(self):
        self.predictor = WordPredictor()

    def test_valid_word_returns_none(self):
        """A correctly spelled word should return None (no correction)."""
        self.assertIsNone(self.predictor.get_autocorrect("THE"))
        self.assertIsNone(self.predictor.get_autocorrect("HELLO"))
        self.assertIsNone(self.predictor.get_autocorrect("COMPUTER"))

    def test_empty_word_returns_none(self):
        """Empty input should return None."""
        self.assertIsNone(self.predictor.get_autocorrect(""))

    def test_close_typo_corrected(self):
        """A close typo should be corrected to a real word."""
        result = self.predictor.get_autocorrect("COMPUTR")
        self.assertIsNotNone(result)
        self.assertIn(result, WORD_FREQUENCIES)

    def test_gibberish_returns_none(self):
        """Completely unrecognizable input should return None."""
        result = self.predictor.get_autocorrect("XYZQWJK")
        self.assertIsNone(result)

    def test_transposition_corrected(self):
        """Adjacent character swaps should be corrected."""
        result = self.predictor.get_autocorrect("HLEPO")
        # Should find HELLO or HELP — both are close via transposition
        self.assertIsNotNone(result)
        self.assertIn(result, WORD_FREQUENCIES)


class TestPredictionBackwardCompat(unittest.TestCase):
    """Ensure Sprint 9 changes don't break existing prediction behavior."""

    def setUp(self):
        self.predictor = WordPredictor()

    def test_prefix_completion(self):
        """Prefix completion should return words starting with the prefix."""
        suggestions = self.predictor.get_suggestions("HAP")
        self.assertGreater(len(suggestions), 0)
        # At least one suggestion should start with "HAP"
        has_prefix_match = any(s.startswith("HAP") for s in suggestions)
        self.assertTrue(has_prefix_match, f"No prefix match in {suggestions}")

    def test_next_word_prediction(self):
        """After a space, bigram prediction should work."""
        suggestions = self.predictor.get_suggestions("I ")
        self.assertGreater(len(suggestions), 0)
        # "AM" should be in suggestions after "I "
        self.assertIn("AM", suggestions)

    def test_empty_input(self):
        """Empty input should return top-N most frequent words."""
        suggestions = self.predictor.get_suggestions("")
        self.assertGreater(len(suggestions), 0)
        self.assertIn("THE", suggestions)

    def test_returns_at_most_n(self):
        """Should never return more than N suggestions."""
        suggestions = self.predictor.get_suggestions("T", n=3)
        self.assertLessEqual(len(suggestions), 3)

    def test_fuzzy_completion_for_typo(self):
        """A typo prefix should still get suggestions via fuzzy backoff."""
        suggestions = self.predictor.get_suggestions("THW")
        # "THW" has no exact prefix match but fuzzy should find "THE", "THEY", etc.
        self.assertGreater(len(suggestions), 0)


if __name__ == "__main__":
    unittest.main()
