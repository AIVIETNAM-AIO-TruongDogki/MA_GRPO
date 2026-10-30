"""
Unit Tests for MA-GRPO Reward Engine (Phase 1).
Verifies mathematical soundness, boundary conditions, and anti-hacking properties.
"""

import math
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import numpy as np
except ImportError:
    np = None

from src.rewards import (
    calculate_length_reward,
    extract_length_bounds,
    count_words,
    calculate_anti_context_ratio,
    extract_entity_trigrams,
    extract_head_words,
    extract_tail_words,
    calculate_burstiness,
    calculate_gaussian_burstiness,
    calculate_sigmoid_fluency,
    calculate_vocabulary_entropy,
    reward_length,
    CoherenceRewardModel,
    QualityRewardModel,
    MAGRPORewardEvaluator,
    MAGRPORewardEvaluatorV2,
    CompositeRewardEngine,
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
        """Verbatim copying of a summary without proper nouns should produce exactly 0.0 novelty."""
        summary = "the animal went down the deep hole and discovered a magical secret door in the garden."
        rollout = "the animal went down the deep hole and discovered a magical secret door in the garden."
        ratio = calculate_anti_context_ratio(rollout, summary)
        self.assertAlmostEqual(ratio, 0.0, places=4)

        # Verbatim copying with an entity still yields near-zero (<0.1) novelty
        summary_ent = "Alice went down the rabbit hole and discovered a magical secret door in the garden."
        rollout_ent = "Alice went down the rabbit hole and discovered a magical secret door in the garden."
        ratio_ent = calculate_anti_context_ratio(rollout_ent, summary_ent)
        self.assertLess(ratio_ent, 0.1)

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


    def test_standard_reward_length(self):
        # Target 1200: safe zone [1020, 1380] -> 1.0
        r_exact = reward_length(["word " * 1200], target_length=1200)[0]
        self.assertAlmostEqual(r_exact, 1.0, places=4)

        r_safe_min = reward_length(["word " * 1020], target_length=1200)[0]
        self.assertAlmostEqual(r_safe_min, 1.0, places=4)

        r_safe_max = reward_length(["word " * 1380], target_length=1200)[0]
        self.assertAlmostEqual(r_safe_max, 1.0, places=4)

        # Under-length: 510 words / 1020 = 0.5
        r_under = reward_length(["word " * 510], target_length=1200)[0]
        self.assertAlmostEqual(r_under, 0.5, places=4)

        # Over-length: (2400 - 1890) / (2400 - 1380) = 510 / 1020 = 0.5
        r_over = reward_length(["word " * 1890], target_length=1200)[0]
        self.assertAlmostEqual(r_over, 0.5, places=4)

        # Extreme over-length: >= 2400 words -> 0.0
        r_extreme = reward_length(["word " * 2500], target_length=1200)[0]
        self.assertAlmostEqual(r_extreme, 0.0, places=4)

    def test_standard_quality_model(self):
        qm = QualityRewardModel()
        prose = "The detective entered the silent room. Dust particles danced in the pale moonlight that filtered through the cracked blinds."
        scores = qm([prose])
        self.assertEqual(len(scores), 1)
        self.assertGreater(scores[0], 0.0)
        self.assertLessEqual(scores[0], 1.0)

    def test_standard_evaluator_and_engine(self):
        evaluator = MAGRPORewardEvaluator(device="cpu")
        fl, fc, fq = evaluator.get_reward_functions()
        self.assertEqual(len(fl(["word " * 1200], target_length=1200)), 1)
        self.assertEqual(len(fc(["A new beginning."], prev_chapters=["An old ending."], summaries=["Summary."])), 1)
        self.assertEqual(len(fq(["A high quality sentence here."])), 1)

        engine = CompositeRewardEngine()
        res = engine.compute_batch_rewards(
            completions=["A nicely written chapter test rollout with good variety and vocabulary."],
            target_words=10,
            prev_chapter_tail="Ending of previous chapter.",
            context_summary="Context summary.",
        )
        self.assertIn("total_rewards", res)
        self.assertIn("r_length", res)
        self.assertIn("r_coherence", res)
        self.assertIn("r_quality", res)

    def test_v2_entity_protection(self):
        # Summary mentions character "Detective Miller" and place "London"
        summary = "Detective Miller arrived at London to investigate the incident."
        entities = extract_entity_trigrams(summary)
        # Should detect "detective miller arrived", "arrived at london", etc.
        self.assertTrue(len(entities) > 0)

        # Rollout containing character names should not be penalized to zero
        rollout = "Detective Miller stepped into the rain and looked across London carefully."
        ratio = calculate_anti_context_ratio(rollout, summary)
        self.assertGreater(ratio, 0.4)

    def test_v2_sigmoid_fluency(self):
        good_text = "The quick brown fox jumps over the lazy dog and enjoys a pleasant morning walk."
        score = calculate_sigmoid_fluency(good_text)
        self.assertGreater(score, 0.5)

    def test_v2_gaussian_burstiness(self):
        text = "Short sentence. Here is a longer sentence with descriptive details that adds texture to the storytelling pacing. Another short one."
        score = calculate_gaussian_burstiness(text, b_target=11.0, tau_b=4.0)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
