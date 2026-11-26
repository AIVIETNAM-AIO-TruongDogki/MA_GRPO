"""Offline unit tests for scripts/test_base_length.py (no GPU required)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.test_base_length import (
    build_variant_payload,
    evaluate_generated_output,
    select_diverse_samples,
    summarize_variant,
)


class TestBaseLengthDiagnostic(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_row = {
            "sample_id": "test_ch01",
            "premise_id": "test",
            "chapter_index": 1,
            "target_words": 1000,
            "chapter_title": "Chapter 1: Test",
            "chapter_goal": "Test goal.",
            "context_summary": "Summary.",
            "prev_chapter_tail": "",
            "prompt": (
                "You are an elite literary novelist.\n"
                "Total Target Length: Approximately 1000 words.\n\n"
                "[SCENE BREAKDOWN & WORD BUDGETS]:\n"
                "- Scene 1 (~250 words) [Opening]: Action.\n"
                "- Scene 2 (~450 words) [Escalation]: Conflict.\n"
                "- Scene 3 (~300 words) [Turn]: Turn.\n\n"
                "CRITICAL WRITING INSTRUCTIONS:\n"
                "Begin with <think> now:\n"
            ),
        }

    def test_all_four_variants_build_cleanly(self) -> None:
        msgs_a, raw_a = build_variant_payload(self.sample_row, "A_v2_raw_no_chat")
        self.assertIsNone(msgs_a)
        self.assertIn("Begin with <think> now:", raw_a)

        msgs_b, raw_b = build_variant_payload(self.sample_row, "B_v2_chat_baseline")
        self.assertIsNone(raw_b)
        self.assertEqual(len(msgs_b), 1)
        self.assertEqual(msgs_b[0]["role"], "user")

        msgs_c, raw_c = build_variant_payload(self.sample_row, "C_chat_budget_short_think")
        self.assertIsNone(raw_c)
        self.assertEqual(len(msgs_c), 2)
        self.assertIn("3 full, detailed paragraphs", msgs_c[1]["content"])
        self.assertIn("5 full, detailed paragraphs", msgs_c[1]["content"])
        self.assertIn("4 full, detailed paragraphs", msgs_c[1]["content"])
        self.assertIn("MAX 80-100 words", msgs_c[1]["content"])

        msgs_d, raw_d = build_variant_payload(self.sample_row, "D_chat_budget_no_think")
        self.assertIsNone(raw_d)
        self.assertEqual(len(msgs_d), 2)
        self.assertIn("DO NOT output <think> tags", msgs_d[1]["content"])

    def test_evaluator_and_summary_metrics(self) -> None:
        prose = " ".join(f"word{i}" for i in range(980)) + "."
        raw = f"<think>\n- Scene 1: 3 paras\n- Scene 2: 5 paras\n</think>\n\n{prose}"
        rec = evaluate_generated_output(self.sample_row, raw, gen_tokens=1350, max_new_tokens=2300, variant="C_chat_budget_short_think")
        self.assertTrue(rec["v_length"])
        self.assertTrue(rec["v_format"])
        self.assertTrue(rec["v_no_loop"])
        self.assertTrue(rec["is_compliant"])
        self.assertFalse(rec["clipped"])

        summary = summarize_variant("C_chat_budget_short_think", [rec], elapsed_s=12.3)
        self.assertEqual(summary["compliant_pct"], 100.0)
        self.assertEqual(summary["clipped_pct"], 0.0)

    def test_select_diverse_samples_from_real_dataset(self) -> None:
        ds = REPO_ROOT / "data" / "grpo_train_condition_d.jsonl"
        if ds.exists():
            picked = select_diverse_samples(ds, 4)
            self.assertEqual(len(picked), 4)
            chapters = [int(r["chapter_index"]) for r in picked]
            self.assertEqual(chapters, [1, 3, 5, 7])

    def test_v3_train_eval_chat_template_alignment_and_idempotency(self) -> None:
        from src.pipeline import (
            VLLMWritingPolicy,
            WRITER_SYSTEM_PROMPT_COT,
            annotate_scene_paragraph_budgets,
            build_writer_chat_messages,
        )

        # 1. Calling build_writer_chat_messages on v2 raw prompt vs freshly constructed prompt
        msgs_from_v2 = build_writer_chat_messages(self.sample_row["prompt"], target_words=1000, enable_cot=True)
        self.assertEqual(len(msgs_from_v2), 2)
        self.assertEqual(msgs_from_v2[0]["role"], "system")
        self.assertEqual(msgs_from_v2[0]["content"], WRITER_SYSTEM_PROMPT_COT)
        self.assertEqual(msgs_from_v2[1]["role"], "user")
        self.assertIn("3 full, detailed paragraphs", msgs_from_v2[1]["content"])
        self.assertIn("5 full, detailed paragraphs", msgs_from_v2[1]["content"])
        self.assertIn("4 full, detailed paragraphs", msgs_from_v2[1]["content"])
        self.assertNotIn("CRITICAL WRITING INSTRUCTIONS:", msgs_from_v2[1]["content"])

        # 2. Idempotency: re-wrapping eval's construct_prompt output produces identical messages
        policy = VLLMWritingPolicy(use_mock=True)
        ch_plan = {
            "chapter_title": "Chapter 1: Test",
            "chapter_goal": "Test goal.",
            "target_words": 1000,
            "characters_active": ["Alex"],
            "scene_outline": [
                "Scene 1 (~250 words) [Opening]: Action.",
                "Scene 2 (~450 words) [Escalation]: Conflict.",
                "Scene 3 (~300 words) [Turn]: Turn.",
            ],
        }
        eval_user_prompt = policy.construct_prompt(ch_plan, {"plot_summary": "Init"}, enable_cot=True)
        msgs_once = build_writer_chat_messages(eval_user_prompt, target_words=1000, enable_cot=True)
        msgs_twice = build_writer_chat_messages(msgs_once[1]["content"], target_words=1000, enable_cot=True)
        self.assertEqual(msgs_once, msgs_twice)
        self.assertEqual(msgs_once[1]["content"].count("full, detailed paragraphs"), 3)
        self.assertEqual(
            annotate_scene_paragraph_budgets(msgs_once[1]["content"]),
            msgs_once[1]["content"],
        )


if __name__ == "__main__":
    unittest.main()
