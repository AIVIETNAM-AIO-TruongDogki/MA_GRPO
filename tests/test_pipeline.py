"""
Unit Tests for MA-GRPO Phase 2 (StoryWriter Event Graph Integration).
Verifies:
    1. StoryWriter event regex parser (Setting, Character, Action, Conflict, Plot Twist)
    2. Event Graph construction & K-Chapter partitioning
    3. Coordinator ReIO context condensation & Context Gate Validator (V_s)
    4. End-to-End story generation simulation
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.coordinator import ContextGateValidator, CoordinatorAgent
from src.pipeline import VLLMWritingPolicy, MAGRPOPipeline
from src.planning import OutlineAgent, PlanningAgent, parse_storywriter_events
from src.rewards import count_words


class TestMAGRPOPhase2StoryWriter(unittest.TestCase):

    def setUp(self):
        self.outline_agent = OutlineAgent()
        self.planning_agent = PlanningAgent()
        self.coordinator = CoordinatorAgent()
        self.pipeline = MAGRPOPipeline()

    def test_storywriter_event_parser(self):
        sample_text = """Event 1: Inception
Setting: Secret laboratory in the Arctic.
Character: Dr. Aris Thorne (Geneticist)
Action: Aris clones an extinct bioluminescent flora.
Conflict: The laboratory containment power grid starts failing.
Plot Twist: The plant emits a cognitive frequency that disables electronic systems.

Event 2: The Thaw
Setting: Exterior icy tundra.
Character: Aris Thorne, Rescue Pilot Sonya
Action: Escaping the collapsing complex on snowmobiles.
Conflict: A blizzard wipes out navigation beacons.
Plot Twist: Sonya's distress transponder is receiving replies from inside the glacier.
"""
        events = parse_storywriter_events(sample_text)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["title"], "Inception")
        self.assertIn("Arctic", events[0]["setting"])
        self.assertIn("Aris Thorne", events[0]["character"])
        self.assertIn("bioluminescent", events[0]["action"])
        self.assertIn("failing", events[0]["conflict"])
        self.assertIn("cognitive frequency", events[0]["plot_twist"])

    def test_outline_agent_event_graph(self):
        premise = "A deep-sea drilling rig penetrates a crystalline dome beneath the ocean trench."
        graph = self.outline_agent.create_event_graph(premise=premise, genre="Sci-Fi Horror", num_events=4)

        self.assertIsInstance(graph, dict)
        self.assertIn("events", graph)
        self.assertGreaterEqual(len(graph["events"]), 4)
        for ev in graph["events"]:
            self.assertIn("setting", ev)
            self.assertIn("action", ev)
            self.assertIn("conflict", ev)
            self.assertIn("plot_twist", ev)

    def test_planning_agent_partitioning(self):
        premise = "Cyberpunk courier carries a sentient encryption key."
        graph = self.outline_agent.create_event_graph(premise=premise, genre="Cyberpunk")

        total_words = 6000
        num_chapters = 5
        blueprint = self.planning_agent.partition_into_chapters(
            event_graph=graph,
            total_target_words=total_words,
            num_chapters=num_chapters,
        )

        self.assertEqual(len(blueprint["chapters"]), num_chapters)
        self.assertEqual(blueprint["chapter_target_words"], 1200)
        for ch in blueprint["chapters"]:
            self.assertEqual(ch["target_words"], 1200)
            self.assertGreater(len(ch["scene_outline"]), 0)

    def test_rolling_planner_consumes_each_sub_event_once(self):
        graph = self.outline_agent.create_event_graph(
            premise="A cartographer maps a city that rearranges itself every midnight.",
            genre="Fantasy",
        )
        story_plan = self.planning_agent.create_story_plan(graph, total_target_words=4000, num_chapters=4)
        planned_ids = []
        for _ in range(4):
            chapter = self.planning_agent.plan_next_chapter(story_plan, {"plot_summary": "Validated state."})
            planned_ids.extend(item["sub_event_id"] for item in chapter["sub_events"])
            self.planning_agent.mark_chapter_complete(story_plan, chapter)

        self.assertEqual(len(planned_ids), len(set(planned_ids)))
        self.assertEqual(set(planned_ids), set(story_plan["completed_sub_event_ids"]))
        self.assertEqual(story_plan["remaining_target_words"], 0)

    def test_context_gate_validator(self):
        validator = ContextGateValidator(threshold=0.80)
        summary = {
            "chapter_index": 2,
            "completed_chapters": 1,
            "plot_summary": "The explorer discovered the crystalline cavern and activated the relay.",
            "character_statuses": ["Elena: Weary"],
            "pending_threads": ["Decipher the signal"],
            "recent_tail_excerpt": "The light dimmed slowly.",
        }
        prev_text = "The explorer ventured into the crystalline cavern and initiated the primary relay."

        result = validator.validate_summary(summary, prev_text)
        self.assertIsInstance(result, dict)
        self.assertIn("composite_score", result)
        self.assertIn("is_valid", result)
        self.assertGreaterEqual(result["composite_score"], 0.0)
        self.assertLessEqual(result["composite_score"], 1.0)

    def test_coordinator_reio_continuity(self):
        ch1_text = "The launch sequence initiated. Elena secured her harness as the engines roared."
        plan_ch1 = {
            "chapter_index": 1,
            "chapter_title": "Chapter 1: Ignition",
            "target_words": 1000,
            "chapter_goal": "Achieve orbit safely.",
        }

        summary = self.coordinator.condense_context(
            next_chapter_index=2,
            past_chapters=[ch1_text],
            past_plans=[plan_ch1],
            characters=["Elena Vance"],
        )

        self.assertEqual(summary["chapter_index"], 2)
        self.assertEqual(summary["completed_chapters"], 1)
        self.assertIn("Elena secured her harness", summary["recent_tail_excerpt"])

    def test_extract_cot_and_prose(self):
        from src.pipeline import extract_cot_and_prose
        sample_with_cot = "<think>\nPlan Scene 1 (~400 words).\nPlan Scene 2 (~600 words).\n</think>\n\nThis is the actual story prose."
        cot, prose = extract_cot_and_prose(sample_with_cot)
        self.assertIn("Plan Scene 1", cot)
        self.assertEqual(prose, "This is the actual story prose.")

        sample_without_cot = "Plain prose without reasoning tokens."
        cot_empty, prose_plain = extract_cot_and_prose(sample_without_cot)
        self.assertEqual(cot_empty, "")
        self.assertEqual(prose_plain, sample_without_cot)

    def test_scene_level_budgeting_in_chapter_plan(self):
        graph = self.outline_agent.create_event_graph(
            premise="A deep sea base experiences communication failure.",
            genre="Sci-Fi",
        )
        story_plan = self.planning_agent.create_story_plan(graph, total_target_words=3000, num_chapters=3)
        chapter = self.planning_agent.plan_next_chapter(story_plan, {"plot_summary": "Initial status"})

        self.assertIn("scene_budgets", chapter)
        self.assertIn("scenes", chapter)
        self.assertEqual(sum(chapter["scene_budgets"]), chapter["target_words"])
        for s in chapter["scenes"]:
            self.assertGreater(s["target_words"], 0)
            self.assertIn("beat_goal", s)

    def test_strip_think_tags_in_rewards(self):
        from src.rewards import count_words, strip_think_tags, calculate_length_reward
        raw_text = "<think>\nHere are 100 reasoning words about plot and pacing.\n</think>\n\nActual narrative with exactly seven words here."
        stripped = strip_think_tags(raw_text)
        self.assertEqual(stripped, "Actual narrative with exactly seven words here.")
        self.assertEqual(count_words(raw_text), 7)
        # Length reward evaluates strictly the 7 prose words against target 7
        self.assertAlmostEqual(calculate_length_reward(raw_text, 7), 1.0, places=3)

    def test_end_to_end_pipeline(self):
        premise = "An ancient subterranean machine wakes up beneath the Sahara desert."
        result = self.pipeline.generate_story(
            premise=premise,
            genre="Archaeological Sci-Fi",
            total_target_words=3000,
            num_chapters=3,
        )

        self.assertEqual(len(result["chapters"]), 3)
        self.assertGreater(result["actual_total_words"], 2500)
        self.assertEqual(len(result["context_summaries"]), 3)
        self.assertIn("=", result["full_text"])
        # Verify that output full_text contains strictly clean prose, not leaked <think> tags
        self.assertNotIn("<think>", result["full_text"])


if __name__ == "__main__":
    unittest.main()
