"""
Advanced Unit Tests for Stage 1 (Macro Planning) & Stage 2 (Context Coordination).
Verifies:
    1. OutlineAgent strict mode vs dynamic premise-driven fallback (no hard-coded template).
    2. SubEventAgent DAG dependency sanitization, cycle breaking & dynamic inter-event linkage.
    3. PlanCriticAgent 5 research criteria:
       - Event coverage
       - Temporal consistency
       - Character consistency
       - Word budget matching (sum(scene_budgets) == target_words)
       - Event ordering (monotonic chronological progression)
    4. Two-Tier Memory in CoordinatorAgent (long_term_memory + short_term_memory).
    5. 0-token Artifact Cleaner & Paragraph Deduplication Gate in VLLMWritingPolicy.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.coordinator import CoordinatorAgent
from src.planning import (
    OutlineAgent,
    PlanningAgent,
    SubEventAgent,
    PlanCriticAgent,
    validate_and_sanitize_dependencies,
    ensure_event_ids,
)


class TestStage1AndStage2Advanced(unittest.TestCase):

    def setUp(self):
        self.outline_agent = OutlineAgent()
        self.sub_event_agent = SubEventAgent()
        self.critic = PlanCriticAgent()
        self.planning_agent = PlanningAgent()
        self.coordinator = CoordinatorAgent()

    def test_outline_agent_strict_mode_rejects_without_llm(self):
        """In strict dataset mode, OutlineAgent must raise ValueError rather than using a static template."""
        strict_agent = OutlineAgent(strict_mode=True)
        with self.assertRaises(ValueError) as ctx:
            strict_agent.create_event_graph(
                premise="A lone chronomancer attempts to repair a fractured timeline.",
                genre="Sci-Fi",
                num_events=6,
            )
        self.assertIn("Strict Mode", str(ctx.exception))

    def test_outline_agent_dynamic_fallback_respects_num_events_and_premise(self):
        """When not strict, dynamic fallback must produce exact num_events without static Elena Vance."""
        premise = "Marcus the alchemist discovers a living mercury entity in Alexandria."
        graph = self.outline_agent.create_event_graph(
            premise=premise,
            genre="Historical Fantasy",
            num_events=7,
        )
        self.assertEqual(len(graph["events"]), 7)
        self.assertEqual(graph["num_events"], 7)
        # Check that premise character "Marcus" was extracted into the events
        events_str = str(graph["events"])
        self.assertIn("Marcus", events_str)

    def test_sub_event_agent_dag_dependency_sanitization_and_cycle_breaking(self):
        """Verify that dangling IDs are stripped, cycles are broken, and ordering is valid DAG."""
        mock_events = [
            {"event_id": "E1", "title": "Start", "setting": "Base", "character": "Protagonist", "action": "Act", "conflict": "Conf", "plot_twist": "Twist"},
            {"event_id": "E2", "title": "Second", "setting": "Cave", "character": "Protagonist", "action": "Act2", "conflict": "Conf2", "plot_twist": "Twist2"},
        ]
        raw_sub_events = [
            {"sub_event_id": "E1.1", "event_id": "E1", "depends_on": ["E1.1", "NON_EXISTENT_ID"]},  # Self dep + dangling
            {"sub_event_id": "E1.2", "event_id": "E1", "depends_on": ["E1.1"]},
            {"sub_event_id": "E2.1", "event_id": "E2", "depends_on": ["E2.2"]},  # Forward cycle!
            {"sub_event_id": "E2.2", "event_id": "E2", "depends_on": ["E2.1"]},
        ]
        sanitized = validate_and_sanitize_dependencies(raw_sub_events, mock_events)

        # Self-dependency and non-existent IDs must be gone from E1.1
        self.assertNotIn("E1.1", sanitized[0]["depends_on"])
        self.assertNotIn("NON_EXISTENT_ID", sanitized[0]["depends_on"])

        # E1.2 depends on E1.1
        self.assertIn("E1.1", sanitized[1]["depends_on"])

        # E2.1 must depend on actual last beat of E1 (E1.2), not forward to E2.2
        self.assertIn("E1.2", sanitized[2]["depends_on"])
        self.assertNotIn("E2.2", sanitized[2]["depends_on"])

    def test_plan_critic_5_research_criteria(self):
        """Test PlanCriticAgent on the 5 research validation criteria."""
        story_plan = {
            "sub_events": [
                {"sub_event_id": "E1.1", "event_id": "E1", "depends_on": []},
                {"sub_event_id": "E1.2", "event_id": "E1", "depends_on": ["E1.1"]},
                {"sub_event_id": "E2.1", "event_id": "E2", "depends_on": ["E1.2"]},
            ],
            "completed_sub_event_ids": ["E1.1"],
            "event_graph": {"events": [{"event_id": "E1", "character": "Kaelen"}, {"event_id": "E2", "character": "Kaelen"}]},
        }

        # Case 1: Valid chapter plan
        valid_plan = {
            "chapter_index": 2,
            "target_words": 1500,
            "scene_budgets": [700, 800],
            "sub_events": [
                {"sub_event_id": "E1.2", "event_id": "E1", "depends_on": ["E1.1"]},
                {"sub_event_id": "E2.1", "event_id": "E2", "depends_on": ["E1.2"]},
            ],
            "characters_active": ["Kaelen"],
        }
        res_valid = self.critic.review(valid_plan, story_plan)
        self.assertTrue(res_valid["approved"])
        self.assertTrue(res_valid["criteria_breakdown"]["coverage_valid"])
        self.assertTrue(res_valid["criteria_breakdown"]["temporal_valid"])
        self.assertTrue(res_valid["criteria_breakdown"]["budget_valid"])
        self.assertTrue(res_valid["criteria_breakdown"]["ordering_valid"])

        # Case 2: Budget mismatch failure
        invalid_budget_plan = dict(valid_plan)
        invalid_budget_plan["scene_budgets"] = [500, 500]  # Sum = 1000 != target 1500
        res_budget = self.critic.review(invalid_budget_plan, story_plan)
        self.assertFalse(res_budget["approved"])
        self.assertFalse(res_budget["criteria_breakdown"]["budget_valid"])

        # Case 3: Temporal dependency violation (E2.1 scheduled before E1.2 is completed)
        invalid_temporal_plan = {
            "chapter_index": 2,
            "target_words": 1000,
            "scene_budgets": [1000],
            "sub_events": [
                {"sub_event_id": "E2.1", "event_id": "E2", "depends_on": ["E1.2"]},  # E1.2 is not completed!
            ],
            "characters_active": ["Kaelen"],
        }
        res_temporal = self.critic.review(invalid_temporal_plan, story_plan)
        self.assertFalse(res_temporal["approved"])
        self.assertFalse(res_temporal["criteria_breakdown"]["temporal_valid"])

    def test_coordinator_two_tier_memory_structure(self):
        """Verify Coordinator produces two-tier memory with cumulative long-term facts."""
        past_chapters = [
            "Chapter 1: Commander Vane discovered the alien distress transmitter in the ice canyon.",
            "Chapter 2: Vane repaired the transmitter while combating subterranean tremors.",
        ]
        past_plans = [
            {"chapter_index": 1, "chapter_title": "Discovery", "chapter_goal": "Find transmitter", "exit_state": "Transmitter secured"},
            {"chapter_index": 2, "chapter_title": "Repair", "chapter_goal": "Activate transmitter", "exit_state": "Signal pulsing"},
        ]

        context = self.coordinator.condense_context(
            next_chapter_index=3,
            past_chapters=past_chapters,
            past_plans=past_plans,
            characters=["Commander Vane"],
        )

        # Must have both memory tiers
        self.assertIn("long_term_memory", context)
        self.assertIn("short_term_memory", context)

        # Long term memory has cumulative milestones
        self.assertIn("major_events", context["long_term_memory"])
        self.assertEqual(len(context["long_term_memory"]["major_events"]), 2)
        self.assertIn("character_states", context["long_term_memory"])
        self.assertIn("Commander Vane", context["long_term_memory"]["character_states"])

        # Short term memory has recent summary and tail excerpt
        self.assertIn("recent_chapter_summary", context["short_term_memory"])
        self.assertIn("recent_tail_excerpt", context["short_term_memory"])

        # Top-level compatibility fields exist
        self.assertIn("plot_summary", context)
        self.assertIn("recent_tail_excerpt", context)

    def test_pilot_dataset_generator_run(self):
        """Verify PilotDatasetGenerator runs end-to-end on premises and exports jsonl."""
        from src.dataset_generator import PilotDatasetGenerator
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            gen = PilotDatasetGenerator(strict_mode=False)
            custom_premises = [
                {
                    "id": "test_premise_01",
                    "genre": "Sci-Fi",
                    "premise": "Dr. Sarah discovers a rogue planet emitting a signal.",
                    "target_words": 2000,
                    "num_chapters": 2,
                }
            ]
            report = gen.generate_pilot_dataset(premises=custom_premises, num_premises=1, output_dir=temp_dir)
            self.assertEqual(report["total_premises_submitted"], 1)
            self.assertEqual(report["valid_chapter_samples_generated"], 2)
            self.assertEqual(report["quarantined_failure_count"], 0)
            self.assertTrue(Path(report["artifacts"]["pilot_samples_file"]).exists())
            self.assertTrue(Path(report["artifacts"]["quarantine_failures_file"]).exists())
        finally:
            shutil.rmtree(temp_dir)

    def test_artifact_cleaner_and_paragraph_deduplication_gate(self):
        """Verify 0-token Artifact Cleaner and Paragraph Deduplication Gate (+ higher-temp retry)."""
        from src.pipeline import (
            VLLMWritingPolicy,
            clean_generation_artifacts,
            deduplicate_repeated_paragraphs,
        )

        dirty = (
            "<think>Internal plan</think>\n"
            "</think> [INST] Mara stepped into the storm. </assistant><|im_end|>\n\n\n"
            "Lightning split the pier.</s>"
        )
        cleaned = clean_generation_artifacts(dirty)
        for bad in ("<think>", "</think>", "</assistant>", "[INST]", "<|im_end|>", "</s>"):
            self.assertNotIn(bad, cleaned)
        self.assertIn("Mara stepped into the storm.", cleaned)
        self.assertIn("Lightning split the pier.", cleaned)

        long_para = (
            "Mara carried the brass lantern down the flooded stone staircase while cold canal water "
            "lapped against the iron bulkhead and distant bells echoed across the Venetian lagoon "
            "beneath the heavy autumn storm clouds gathering overhead."
        )
        short_para = "She paused and listened closely."
        text_with_dup = f"{long_para}\n\n{short_para}\n\n{long_para}\n\n{short_para}"
        deduped, removed = deduplicate_repeated_paragraphs(text_with_dup, min_words=30)
        self.assertEqual(removed, 1)
        self.assertEqual(deduped.count("Mara carried the brass lantern"), 1)
        # Short paragraphs (<= 30 words) are preserved
        self.assertEqual(deduped.count(short_para), 2)

        # Higher-temperature retry triggers when dedup drops length below 75% of target_words
        recorded_temps = []
        unique_para_2 = (
            "Inside the dry restoration loft, Lorenzo unfolded the perforated copper scroll beneath "
            "the magnifier glass and traced the hidden tidal sluice coordinates etched along the margin "
            "before footsteps echoed sharply along the wooden landing outside."
        )

        def _fake_gen(_prompt, _max_tok, temp, _top_p):
            recorded_temps.append(temp)
            if len(recorded_temps) == 1:
                return f"<think>draft</think>\n{long_para}\n\n{long_para} </assistant>"
            return f"<think>retry</think>\n{long_para}\n\n{unique_para_2}"

        policy = VLLMWritingPolicy(use_mock=False, custom_generate_fn=_fake_gen)
        out = policy.generate_chapter_prose(
            chapter_plan={"chapter_title": "Ch 1", "chapter_goal": "Unlock vault", "target_words": 60},
            context_summary={"plot_summary": "Start"},
            temperature=0.80,
        )
        self.assertEqual(len(recorded_temps), 2)
        self.assertGreater(recorded_temps[1], recorded_temps[0])
        self.assertIn("Lorenzo unfolded", out)
        self.assertNotIn("</assistant>", out)

    def test_coordinator_single_llm_call_no_validator(self):
        """CoordinatorAgent synthesizes two-tier memory in 1 LLM call without any validator layer."""
        calls = []

        def _mock_coord_llm(sys_p, _usr_p):
            calls.append(sys_p)
            return (
                '{"long_term_memory": {"major_events": ["Found vault"], '
                '"character_states": {"Mara": "Alert"}, "unresolved_plot_threads": ["Cipher"], '
                '"revealed_secrets": []}, "short_term_memory": {"recent_chapter_summary": '
                '"Mara unlocked the submerged vault and escaped the rising tide.", '
                '"immediate_turning_point": "Guards arrive at the canal."}}'
            )

        coord = CoordinatorAgent(llm_generate_fn=_mock_coord_llm)
        ctx = coord.condense_context(
            next_chapter_index=2,
            past_chapters=["Chapter 1 prose about Mara in Venice."],
            past_plans=[{"chapter_index": 1, "chapter_title": "Ch 1", "chapter_goal": "Find vault", "exit_state": "Escaped"}],
            characters=["Mara"],
        )
        # Exactly 1 LLM call (Coordinator only, zero validator calls)
        self.assertEqual(len(calls), 1)
        self.assertIn("submerged vault", ctx["plot_summary"])

    def test_clean_tail_cumulative_ltm_none_twist_and_intra_para_dedup(self):
        """Verify sentence-aligned tail, cumulative capped LTM in prompt, no 'Land turn: None', and intra-paragraph sentence dedup."""
        from src.coordinator import compact_long_term_memory, extract_clean_tail
        from src.pipeline import VLLMWritingPolicy, deduplicate_repeated_paragraphs

        # 1. Clean tail removes dangling mid-sentence end and mid-sentence start
        sentences = [
            f"Sentence number {i} describes Mara exploring the flooded Venetian archive with her brass lantern."
            for i in range(1, 22)
        ]
        dangling_prose = " ".join(sentences) + " As she reached for the door and"
        tail = extract_clean_tail(dangling_prose, max_words=180)
        self.assertTrue(tail.endswith("."), msg=f"Tail ended mid-sentence: {tail[-40:]}")
        self.assertTrue(tail[0].isupper(), msg=f"Tail started mid-sentence: {tail[:40]}")
        self.assertLessEqual(len(tail.split()), 180)

        # 2. Cumulative LTM merges earlier milestones & enforces hard caps (<=8 events, <=6 chars, <=4 threads)
        prev_ltm = {
            "major_events": [f"Ch {i}: Milestone event {i} occurred." for i in range(1, 6)],
            "character_states": {"Mara": "In Venice", "Lorenzo": "At workshop"},
            "unresolved_plot_threads": ["Old mystery"],
            "revealed_secrets": ["Grandfather built the clockwork"],
        }
        new_ltm = {
            "major_events": ["Ch 6: Sabotage uncovered at the Arsenal."],
            "character_states": {"Mara": "At Arsenal sluice", "Bellini": "Investigating"},
            "unresolved_plot_threads": ["Stop the midnight flood"],
            "revealed_secrets": ["Contarini rigged the gates"],
        }
        merged = compact_long_term_memory(new_ltm, previous_ltm=prev_ltm)
        self.assertEqual(len(merged["major_events"]), 6)
        self.assertIn("Ch 1: Milestone event 1 occurred.", merged["major_events"])
        self.assertIn("Ch 6: Sabotage uncovered at the Arsenal.", merged["major_events"])
        self.assertIn("Lorenzo", merged["character_states"])
        self.assertEqual(merged["character_states"]["Mara"], "At Arsenal sluice")

        # 3. VLLMWritingPolicy.construct_prompt includes [LONG-TERM STORY MEMORY] for Ch2+ only
        policy = VLLMWritingPolicy(use_mock=True)
        ch_plan = {"chapter_title": "Chapter 2", "chapter_goal": "Decode scroll", "target_words": 1000}
        prompt_ch1 = policy.construct_prompt(ch_plan, {"completed_chapters": 0, "plot_summary": "Init", "long_term_memory": merged})
        prompt_ch2 = policy.construct_prompt(ch_plan, {"completed_chapters": 1, "plot_summary": "Ch1 summary", "long_term_memory": merged})
        self.assertNotIn("[LONG-TERM STORY MEMORY]:", prompt_ch1)
        self.assertIn("[LONG-TERM STORY MEMORY]:", prompt_ch2)
        self.assertIn("Ch 1: Milestone event 1 occurred.", prompt_ch2)

        # 4. PlanningAgent Condition D never emits "Land turn: None"
        eg = {
            "premise": "Test",
            "genre": "Fantasy",
            "events": [
                {
                    "event_id": "E1",
                    "title": "Quest Begins",
                    "setting": "Meridia underwater kingdom",
                    "character": "Arion",
                    "action": "Arion investigates disappearances",
                    "conflict": "Merfolk doubt his warnings",
                    "plot_twist": "None",
                    "depends_on": [],
                }
            ],
        }
        plan_d = self.planning_agent.partition_into_chapters(eg, total_target_words=1000, num_chapters=1, enable_micro_subevents=False)
        scene_text = "\n".join(plan_d["chapters"][0]["scene_outline"])
        self.assertNotIn("Land turn: None", scene_text)

        # 5. Intra-paragraph repeated sentence loop (like Fantasy Ch4) is stripped
        looped_para = (
            "Arion's spear struck true and Lila's spells provided brief respite against the sea serpent's scales. "
            "The storm above raged on with terrifying fury across the abyssal trench. "
            "Arion's spear struck true and Lila's spells provided brief respite against the sea serpent's scales."
        )
        deduped, removed = deduplicate_repeated_paragraphs(looped_para)
        self.assertGreaterEqual(removed, 1)
        self.assertEqual(deduped.count("Arion's spear struck true"), 1)


if __name__ == "__main__":
    unittest.main()
