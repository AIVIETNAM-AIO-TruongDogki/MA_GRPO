import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_base_vs_checkpoint import (
    CANONICAL_5_BENCHMARK_PROMPTS,
    CANONICAL_8MODEL_BASELINES,
    calculate_distinct_3,
    calculate_distinct_4,
    calculate_length_score,
    clean_generated_prose,
    extract_think_and_prose,
    format_ma_grpo_prompt,
    load_test_prompts,
)


class TestCompareScript(unittest.TestCase):
    def test_canonical_5_prompts_exact_match(self):
        prompts = load_test_prompts("8models", num_samples=5)
        self.assertEqual(len(prompts), 5)
        self.assertEqual(prompts[0]["id"], "example_000")
        self.assertEqual(prompts[1]["id"], "example_001")
        self.assertEqual(prompts[2]["id"], "example_002")
        self.assertEqual(prompts[3]["id"], "example_003")
        self.assertEqual(prompts[4]["id"], "example_004")

        # Canonical target lengths
        self.assertEqual(prompts[0]["target_length"], 1350)
        self.assertEqual(prompts[1]["target_length"], 1575)
        self.assertEqual(prompts[2]["target_length"], 1900)
        self.assertEqual(prompts[3]["target_length"], 1600)
        self.assertEqual(prompts[4]["target_length"], 900)

    def test_canonical_3_prompts_mean_length(self):
        prompts = load_test_prompts("8models", num_samples=3)
        self.assertEqual(len(prompts), 3)
        target_lens = [p["target_length"] for p in prompts]
        self.assertAlmostEqual(sum(target_lens) / len(target_lens), 1608.33, places=1)

    def test_format_ma_grpo_prompt(self):
        p = format_ma_grpo_prompt("A myth story about Horus.", 1350)
        self.assertIn("Total Target Length: Approximately 1350 words", p)
        self.assertIn("Scene 1: Opening exposition and world details (~338 words)", p)
        self.assertIn("Scene 2: Inciting tension and primary dialogue (~608 words)", p)
        self.assertIn("<think>...</think>", p)

    def test_extract_think_and_prose(self):
        raw = "<think>Planning Scene 1 (300 words), Scene 2 (500 words).</think>Once upon a time in a city far away..."
        res = extract_think_and_prose(raw)
        self.assertIn("Planning Scene 1", res["think"])
        self.assertEqual(res["prose"], "Once upon a time in a city far away...")

    def test_extract_think_and_prose_no_think(self):
        raw = "Once upon a time without think tags."
        res = extract_think_and_prose(raw)
        self.assertEqual(res["think"], "")
        self.assertEqual(res["prose"], "Once upon a time without think tags.")

    def test_length_and_distinct_metrics(self):
        sl_100 = calculate_length_score(1500, 1500)
        self.assertEqual(sl_100, 100.0)

        sl_50 = calculate_length_score(750, 1500)
        self.assertEqual(sl_50, 50.0)

        text = "The quick brown fox jumps over the lazy dog repeatedly"
        d3 = calculate_distinct_3(text)
        d4 = calculate_distinct_4(text)
        self.assertGreater(d3, 0.0)
        self.assertGreater(d4, 0.0)

    def test_canonical_baselines_structure(self):
        self.assertEqual(len(CANONICAL_8MODEL_BASELINES), 8)
        models = [row["Model / Method"] for row in CANONICAL_8MODEL_BASELINES]
        self.assertIn("AGENTS' ROOM", models)
        self.assertIn("STORYWRITER", models)
        self.assertIn("Self-Lengthen", models)
        self.assertIn("LongWriter-llama3.1-8b", models)
        self.assertIn("MA-GRPO-50", models)
        self.assertIn("MA-GRPO-100", models)


if __name__ == "__main__":
    unittest.main()
