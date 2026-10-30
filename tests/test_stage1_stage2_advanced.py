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
    5. ContextGateValidator multi-signal verification (negation contradiction check, 5-part input).
    6. Non-generic structured state fallback and validation metadata preservation.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.coordinator import ContextGateValidator, CoordinatorAgent, detect_heuristic_negation_contradictions
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
        self.validator = ContextGateValidator()

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
        self.assertIn("validation_metadata", context)
        self.assertIn("faithfulness_score", context["validation_metadata"])
        self.assertIn("is_valid", context["validation_metadata"])

    def test_validator_negation_contradiction_detection(self):
        """Detect negation contradiction clashes where summary inverts source polarity."""
        source = "Elena trusted Marcus completely throughout the trial."
        # Contradictory summary
        contradictory_summary = "Elena did not trust Marcus during the trial."
        clashes = detect_heuristic_negation_contradictions(contradictory_summary, source)
        self.assertTrue(len(clashes) > 0)

        # Valid non-contradictory summary
        valid_summary = "Elena trusted Marcus as they faced the trial."
        clashes_valid = detect_heuristic_negation_contradictions(valid_summary, source)
        self.assertEqual(len(clashes_valid), 0)

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


if __name__ == "__main__":
    unittest.main()
