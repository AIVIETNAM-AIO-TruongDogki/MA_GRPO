"""
Unit Tests for MA-GRPO Reward Engine (Phase 1).
Verifies mathematical soundness, boundary conditions, and anti-hacking properties.
"""

import math
import unittest
import numpy as np

from src.rewards import (
    calculate_length_reward,
    extract_length_bounds,
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

    def test_length_reward_piecewise_formula(self):
        # Explicit bounds: L_lower=1000, L_upper=1200, L_max=1600
        # 1. Inside target range [L_lower, L_upper] -> 1.0
        text_inside = "word " * 1100
        self.assertAlmostEqual(
            calculate_length_reward(text_inside, l_lower=1000, l_upper=1200, l_max=1600), 1.0, places=4
        )
        self.assertAlmostEqual(
            calculate_length_reward("word " * 1000, l_lower=1000, l_upper=1200, l_max=1600), 1.0, places=4
        )
        self.assertAlmostEqual(
            calculate_length_reward("word " * 1200, l_lower=1000, l_upper=1200, l_max=1600), 1.0, places=4
        )

        # 2. Below L_lower -> len(o) / L_lower
        text_short = "word " * 500
        score_short = calculate_length_reward(text_short, l_lower=1000, l_upper=1200, l_max=1600)
        self.assertAlmostEqual(score_short, 0.5, places=4)

        # 3. Above L_upper -> (L_max - len(o)) / (L_max - L_upper)
        # Here L_max=1600, L_upper=1200 => denominator = 400. At 1400: (1600 - 1400) / 400 = 0.5
        text_long = "word " * 1400
        score_long = calculate_length_reward(text_long, l_lower=1000, l_upper=1200, l_max=1600)
        self.assertAlmostEqual(score_long, 0.5, places=4)

        # 4. At L_max -> 0.0
        text_max = "word " * 1600
        score_max = calculate_length_reward(text_max, l_lower=1000, l_upper=1200, l_max=1600)
        self.assertAlmostEqual(score_max, 0.0, places=4)

        # 5. Above L_max -> clamped to 0.0
        text_excess = "word " * 2000
        score_excess = calculate_length_reward(text_excess, l_lower=1000, l_upper=1200, l_max=1600)
        self.assertAlmostEqual(score_excess, 0.0, places=4)

    def test_extract_length_bounds(self):
        prompt1 = "Write between 1300–1400 words telling a story..."
        low1, up1, max1 = extract_length_bounds(prompt1)
        self.assertEqual(low1, 1300)
        self.assertEqual(up1, 1400)
        self.assertGreater(max1, up1)

        prompt2 = "The story must be between 1,550 and 1,600 words."
        low2, up2, max2 = extract_length_bounds(prompt2)
        self.assertEqual(low2, 1550)
        self.assertEqual(up2, 1600)

        prompt3 = "The story should be at least 900 words long."
        low3, up3, max3 = extract_length_bounds(prompt3)
        self.assertEqual(low3, 900)
        self.assertGreater(up3, 900)

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
