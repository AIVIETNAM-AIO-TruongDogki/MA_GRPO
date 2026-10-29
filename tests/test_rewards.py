"""
Unit Tests for MA-GRPO Reward Engine (Phase 1).
Verifies mathematical soundness, boundary conditions, and anti-hacking properties.
"""

import math
import unittest
import numpy as np

from src.rewards import (
    calculate_length_reward,
    count_words,
    calculate_anti_context_ratio,
    extract_head_words,
    extract_tail_words,
    calculate_burstiness,
    calculate_vocabulary_entropy,
)


class TestMAGRPORewards(unittest.TestCase):

    def test_word_count(self):
        text = "The quick brown fox jumps over the lazy dog."
        self.assertEqual(count_words(text), 9)
        self.assertEqual(count_words(""), 0)
        self.assertEqual(count_words("   "), 0)

    def test_length_reward_exact(self):
        target = 1200
        text = "word " * 1200
        score = calculate_length_reward(text, target)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_length_reward_half(self):
        target = 1200
        text = "word " * 600
        score = calculate_length_reward(text, target)
        self.assertAlmostEqual(score, 0.5, places=4)

    def test_length_reward_excess(self):
        target = 1200
        text = "word " * 2400
        score = calculate_length_reward(text, target)
        self.assertAlmostEqual(score, 0.0, places=4)

    def test_anti_context_ratio_verbatim_copy(self):
        """Verbatim copying of the summary should produce a near-zero novelty score."""
        summary = "Alice went down the rabbit hole and discovered a magical secret door in the garden."
        rollout = "Alice went down the rabbit hole and discovered a magical secret door in the garden."
        ratio = calculate_anti_context_ratio(rollout, summary)
        self.assertAlmostEqual(ratio, 0.0, places=4)

    def test_anti_context_ratio_novel_text(self):
        """Completely new narrative should have high novelty (1.0)."""
        summary = "Alice went down the rabbit hole and discovered a secret door."
        rollout = "The deep space explorer adjusted his helmet and stepped onto the crimson surface of Mars."
        ratio = calculate_anti_context_ratio(rollout, summary)
        self.assertAlmostEqual(ratio, 1.0, places=4)

    def test_vocabulary_entropy_repetitive(self):
        """Repetitive text (mode collapse) should produce zero or near-zero entropy."""
        repetitive_text = "danger " * 500
        entropy = calculate_vocabulary_entropy(repetitive_text)
        self.assertAlmostEqual(entropy, 0.0, places=4)

    def test_vocabulary_entropy_diverse(self):
        """Rich vocabulary usage should produce high normalized Shannon entropy."""
        diverse_words = [f"word_{i}" for i in range(100)]
        diverse_text = " ".join(diverse_words)
        entropy = calculate_vocabulary_entropy(diverse_text)
        self.assertGreater(entropy, 0.95)

    def test_burstiness_monotone_vs_dynamic(self):
        """Monotone robotic pacing should score lower than natural varied sentence pacing."""
        # Monotone: All sentences have exactly 5 words
        monotone_text = (
            "This is a short sentence. Here is another short sentence. "
            "They all have five words. Nothing changes in this pacing."
        )
        # Dynamic: Mix of very short (2-3 words) and long descriptive (20+ words) sentences
        dynamic_text = (
            "Stop. The rain lashed against the cracked window pane as Detective Miller struggled "
            "to decipher the faded bloodstains smeared across the mahogany desk. He hesitated. "
            "Could this be the final clue left behind by the elusive syndicate?"
        )
        score_monotone = calculate_burstiness(monotone_text)
        score_dynamic = calculate_burstiness(dynamic_text)
        self.assertGreater(score_dynamic, score_monotone)

    def test_snippets_extraction(self):
        words = [f"w{i}" for i in range(100)]
        text = " ".join(words)
        tail = extract_tail_words(text, n_words=10)
        head = extract_head_words(text, n_words=10)
        self.assertEqual(head, " ".join([f"w{i}" for i in range(10)]))
        self.assertEqual(tail, " ".join([f"w{i}" for i in range(90, 100)]))


if __name__ == "__main__":
    unittest.main()
