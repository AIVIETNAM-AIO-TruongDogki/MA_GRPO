"""Unit tests for Condition D (Macro-Only) Batch GRPO Dataset Generator."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.train import (
        extract_context_from_prompt,
        extract_tail_from_prompt,
        extract_target_words_from_text,
    )
except ModuleNotFoundError:
    # CPU-only test environments without torch: mirror exact scripts/train.py regexes
    def extract_target_words_from_text(prompt_text: str, fallback: int = 1400) -> int:
        clean = (prompt_text or "").replace(",", "")
        m = re.search(r"Total Target Length:\s*Approximately\s*(\d+)\s*words", clean, re.IGNORECASE)
        return int(m.group(1)) if m else fallback

    def extract_context_from_prompt(prompt_text: str) -> str:
        m = re.search(r"\[CONTEXT SUMMARY\]:\s*([\s\S]*?)(?=\n\n\[|\n\[CHAPTER GOAL\])", prompt_text or "")
        return m.group(1).strip() if m else ""

    def extract_tail_from_prompt(prompt_text: str) -> str:
        m = re.search(r"\[PRECEDING SCENE CONTINUATION.*?\]:\s*([\s\S]*?)(?=\n\n\[|\n\[CHAPTER GOAL\])", prompt_text or "")
        return m.group(1).strip() if m else ""
from src.dataset_generator import (
    ConditionDGRPODatasetGenerator,
    load_premises_pool,
    normalize_premise_text,
)


def _fake_macro_outline(num_events: int = 8) -> str:
    blocks = []
    for i in range(1, num_events + 1):
        dep = f"E{i - 1}" if i > 1 else "None"
        blocks.append(
            f"Event {i}: Turning Point {i}\n"
            f"Setting: Sector {i} Harbor\n"
            f"Character: Mira, Rowan\n"
            f"Action: Mira uncovers ledger fragment {i} with Rowan.\n"
            f"Conflict: Harbor guards tighten the blockade around Sector {i}.\n"
            f"Plot Twist: Fragment {i} implicates the council archivist.\n"
            f"Depends On: {dep}"
        )
    return "\n\n".join(blocks)


class TestConditionDGRPODatasetGenerator(unittest.TestCase):
    def test_load_premises_pool_excludes_test_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_file = tmp_path / "train_premises.jsonl"
            test_file = tmp_path / "test_premises.jsonl"

            train_rows = [
                {"id": "p1", "genre": "Sci-Fi", "premise": "A diver discovers a sunken clockwork archive."},
                {"id": "p2", "genre": "Fantasy", "premise": "An apprentice cartographer maps a moving island."},
                {"id": "p3", "genre": "Mystery", "premise": "A violinist hears a secret code in a symphony."},
            ]
            # Exact text overlap (different punctuation/casing) for p2
            test_rows = [
                {"id": "t99", "premise": "an apprentice cartographer maps a moving island!"},
            ]
            train_file.write_text("\n".join(json.dumps(r) for r in train_rows) + "\n", encoding="utf-8")
            test_file.write_text("\n".join(json.dumps(r) for r in test_rows) + "\n", encoding="utf-8")

            selected, excluded = load_premises_pool(
                premises_files=[train_file],
                include_canonical=False,
                exclude_files=[test_file],
                num_premises=10,
                target_words_choices=(8000, 9000, 10000),
                num_chapters=8,
            )
            self.assertEqual(excluded, 1)
            self.assertEqual([s["id"] for s in selected], ["p1", "p3"])
            self.assertEqual([s["target_words"] for s in selected], [8000, 9000])

    def test_load_real_mops_pool_80_premises_zero_test_overlap(self) -> None:
        mops_path = PROJECT_ROOT / "data" / "mops_premises.jsonl"
        test_path = PROJECT_ROOT / "data" / "tell-me-a-story-test.jsonl"
        if not mops_path.exists():
            self.skipTest("data/mops_premises.jsonl not present")

        selected, _ = load_premises_pool(
            premises_files=[mops_path],
            include_canonical=True,
            exclude_files=[test_path],
            num_premises=80,
            target_words_choices=(8000, 9000, 10000),
            num_chapters=8,
        )
        self.assertEqual(len(selected), 80)
        self.assertTrue(all(item["num_chapters"] == 8 for item in selected))

        if test_path.exists():
            test_norms = set()
            for line in test_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                for k in ("premise", "inputs", "prompt"):
                    if row.get(k):
                        test_norms.add(normalize_premise_text(row[k]))
            for item in selected:
                self.assertNotIn(normalize_premise_text(item["premise"]), test_norms)

    def test_batch_generator_matches_condition_d_and_train_parsers(self) -> None:
        batch_call_sizes: List[int] = []

        def fake_batch_llm(requests: List[Tuple[str, str]], max_tokens: int, temperature: float) -> List[str]:
            batch_call_sizes.append(len(requests))
            outputs: List[str] = []
            for sys_p, _usr_p in requests:
                if "Macro Event Graph" in sys_p:
                    outputs.append(_fake_macro_outline(8))
                elif "Context Coordinator" in sys_p:
                    outputs.append(
                        json.dumps(
                            {
                                "long_term_memory": {
                                    "major_events": ["Mira and Rowan secured the ledger fragment."],
                                    "character_states": {"Mira": "At the harbor vault, alert and holding the ledger."},
                                    "unresolved_plot_threads": ["Identify who ordered the blockade."],
                                    "revealed_secrets": ["The council archivist forged the manifest."],
                                },
                                "short_term_memory": {
                                    "recent_chapter_summary": "Mira and Rowan infiltrated the harbor vault and retrieved the forged ledger despite the guards.",
                                    "immediate_turning_point": "Alarms ring as the inner gate seals behind them.",
                                },
                            }
                        )
                    )
                else:
                    outputs.append(
                        "Rain hammered the slate roofs of Sector Harbor as Mira crouched beside the iron bollard. "
                        "Rowan whispered from the archway, holding the brass lantern low so the guards on the pier would not spot them. "
                        "Inside the vault, the ledger pages smelled of salt and cedar oil, revealing forged dock seals signed by the archivist. "
                        "Before Mira could copy the final cipher, the harbor bell clanged twice and boots echoed along the wet cobblestones."
                    )
            return outputs

        premises = [
            {"id": f"story_{i:02d}", "genre": "Mystery", "premise": f"Story {i} about a stolen harbor ledger.", "target_words": 8000 + i * 1000, "num_chapters": 8, "source": "unit_test"}
            for i in range(3)
        ]

        with tempfile.TemporaryDirectory() as tmp:
            gen = ConditionDGRPODatasetGenerator(batch_generate_fn=fake_batch_llm)
            report = gen.generate_dataset(
                premises=premises,
                output_dir=tmp,
                num_chapters=8,
                batch_size=3,
                resume=False,
            )

            self.assertEqual(report["condition"], "D")
            self.assertFalse(report["enable_micro_subevents"])
            self.assertEqual(report["completed_premises"], 3)
            self.assertEqual(report["valid_chapter_samples"], 24)
            self.assertEqual(report["mock_tail_count"], 0)
            self.assertEqual(report["non_empty_tail_ratio_ch2_plus"], 1.0)
            # Every batched LLM stage should have been called with all 3 stories together
            self.assertTrue(batch_call_sizes)
            self.assertTrue(all(sz == 3 for sz in batch_call_sizes))

            jsonl_path = Path(tmp) / "grpo_train_condition_d.jsonl"
            rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 24)

            for r in rows:
                self.assertEqual(r["condition"], "D")
                self.assertNotIn("Narrative Arc Stage:", r["prompt"])
                self.assertNotIn("story_arc", r)
                self.assertIn("<think>", r["prompt"])
                self.assertIn("</think>", r["prompt"])
                self.assertIn("STRICT LENGTH & FORMAT REQUIREMENTS:", r["prompt"])
                self.assertIn("[CONTEXT SUMMARY]:", r["prompt"])
                self.assertIn("[CHAPTER GOAL]:", r["prompt"])
                self.assertEqual(extract_target_words_from_text(r["prompt"]), r["target_words"])
                self.assertTrue(bool(extract_context_from_prompt(r["prompt"])))
                if r["chapter_index"] > 1:
                    self.assertIn("Rain hammered the slate roofs", r["prev_chapter_tail"])
                    self.assertTrue(bool(extract_tail_from_prompt(r["prompt"])))
                    self.assertNotIn("The journey continued into the subsequent sector.", r["prev_chapter_tail"])

    def test_resume_skips_completed_and_cleans_partial_premises(self) -> None:
        premises = [
            {"id": "res_01", "genre": "Sci-Fi", "premise": "A station orbit decays above Titan.", "target_words": 8000, "num_chapters": 4, "source": "unit_test"},
            {"id": "res_02", "genre": "Fantasy", "premise": "A clockmaker mends a frozen valley.", "target_words": 9000, "num_chapters": 4, "source": "unit_test"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            gen = ConditionDGRPODatasetGenerator(batch_generate_fn=None)
            # First run: generate only res_01
            r1 = gen.generate_dataset(premises=premises[:1], output_dir=tmp, num_chapters=4, batch_size=2, resume=True)
            self.assertEqual(r1["valid_chapter_samples"], 4)

            # Simulate an interrupted partial write for res_02 (only 1 of 4 chapters)
            jsonl_path = Path(tmp) / "grpo_train_condition_d.jsonl"
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"premise_id": "res_02", "chapter_index": 1, "total_chapters": 4, "prompt": "partial"}) + "\n")

            # Second run with resume=True should keep res_01 intact and cleanly regenerate all 4 chapters of res_02
            r2 = gen.generate_dataset(premises=premises, output_dir=tmp, num_chapters=4, batch_size=2, resume=True)
            self.assertEqual(r2["completed_premises"], 2)
            self.assertEqual(r2["valid_chapter_samples"], 8)
            self.assertEqual(r2["newly_generated_samples"], 4)

            # Third run with resume=True should be a no-op
            r3 = gen.generate_dataset(premises=premises, output_dir=tmp, num_chapters=4, batch_size=2, resume=True)
            self.assertEqual(r3["valid_chapter_samples"], 8)
            self.assertEqual(r3["newly_generated_samples"], 0)

    def test_cli_mock_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "generate_grpo_dataset.py"),
                "--backend", "mock",
                "--num-premises", "2",
                "--num-chapters", "4",
                "--batch-size", "2",
                "--output-dir", tmp,
                "--resume",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, msg=f"CLI failed:\n{proc.stdout}\n{proc.stderr}")
            out_file = Path(tmp) / "grpo_train_condition_d.jsonl"
            self.assertTrue(out_file.exists())
            rows = [json.loads(l) for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(rows), 8)


if __name__ == "__main__":
    unittest.main()
