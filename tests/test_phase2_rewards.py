"""Unit & E2E CLI tests for Phase 2 Expansion Reward Architecture (EXP-00 .. EXP-05)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rewards.aggregation import (
    Phase2RewardEngine,
    aggregate_global_arithmetic,
    aggregate_global_geometric,
    aggregate_local_arithmetic,
    aggregate_local_geometric,
    aggregate_phase2_reward,
    load_reward_config,
)
from rewards.base import ConstraintPenalties, EvaluationInput, GlobalScores, LocalScores
from rewards.evaluators import ConstraintEvaluator, MockDeterministicJudgeBackend
from rewards.validation import (
    analyze_reward_distributions_and_correlations,
    compute_human_alignment_report,
    evaluate_perturbation_discrimination,
)


def _make_clean_chapter_input() -> EvaluationInput:
    prose = (
        "<think>Plan opening in Venice archive, confront inspector, end on cipher discovery.</think>\n\n"
        "Rain hammered the submerged Venetian vault as Archivist Mara Valeri forced open the corroded iron bulkhead. "
        "Inside the flooded chamber, she located the sealed bronze cylinder bearing her grandfather's cipher mark and "
        "carefully lifted the copper scrolls out of the brackish canal water.\n\n"
        "Before she could climb the stone staircase, Inspector Bellini blocked the doorway with two maritime guards, "
        "demanding she surrender the cylinder under emergency decree. Realizing Bellini already knew about the sluice "
        "override, Mara feigned compliance while slipping the second perforated scroll into Lorenzo's satchel.\n\n"
        "When Lorenzo triggered a diversion by overturning an oil lantern onto the wet flagstones, Mara vaulted over "
        "the railing into the waiting skiff. Glancing back at the smoking arch, she realized the maritime council was "
        "actively covering up a planned flood that would erase the ledger evidence before dawn."
    )
    return EvaluationInput(
        generated_chapter=prose,
        target_words=150,
        chapter_title="Chapter 2: Flooded Vault Confrontation",
        chapter_goal="Mara retrieves the bronze cylinder from the Venetian vault, confronts Inspector Bellini, and escapes with Lorenzo.",
        scene_outline=[
            "Scene 1: Mara unlocks the submerged Venetian archive and recovers the bronze cylinder.",
            "Scene 2: Inspector Bellini ambushes Mara demanding the cylinder under maritime law.",
            "Scene 3: Lorenzo creates a diversion so Mara escapes with the second scroll.",
        ],
        characters_active=["Mara Valeri", "Inspector Bellini", "Lorenzo"],
        context_summary="Mara discovered clues pointing to an ancient tidal sluice hidden beneath San Marco.",
        prev_chapter_tail="Holding the brass key tightly, Mara descended the wet steps toward the iron bulkhead.",
        long_term_memory={
            "major_events": ["Ch 1: Mara uncovered her grandfather's hidden archive key."],
            "character_states": {"Mara": "Alert in Venice", "Lorenzo": "Loyal locksmith ally"},
        },
        chapter_index=2,
        premise_id="venice_test_01",
    )


class TestPhase2ExpansionRewards(unittest.TestCase):
    def test_python_constraint_evaluator_penalties(self) -> None:
        con_eval = ConstraintEvaluator(use_embedder=False, max_think_words=400)
        clean_in = _make_clean_chapter_input()
        clean_pen = con_eval.evaluate(clean_in, contradictions=[])
        self.assertEqual(clean_pen.p_length, 0.0)
        self.assertEqual(clean_pen.p_contra, 0.0)
        self.assertEqual(clean_pen.p_format, 0.0)
        self.assertGreater(clean_pen.think_words, 5)
        self.assertGreater(clean_pen.prose_words, 130)

        # Unclosed <think> + leaked tag + severe repetition + critical contradiction
        bad_in = _make_clean_chapter_input()
        loop_para = (
            "The silence hung across the empty corridor while cold water dripped from the stone ceiling "
            "and nothing changed in the room as time dragged slowly onward without any decision or event."
        )
        bad_in.generated_chapter = f"<think>Unclosed plan\n{loop_para}\n\n{loop_para}\n</assistant>"
        bad_pen = con_eval.evaluate(
            bad_in,
            contradictions=[{"severity": "critical", "reason": "Dead character alive."}],
        )
        self.assertGreater(bad_pen.p_format, 0.5)
        self.assertGreater(bad_pen.p_contra, 0.5)

    def test_arithmetic_vs_geometric_aggregation(self) -> None:
        # One very low axis (event_realization=0.15) should be penalized MUCH harder by Geometric than Arithmetic
        loc = LocalScores(event_realization=0.15, causal_development=0.95, meaningful_elaboration=0.90)
        glob = GlobalScores(macro_adherence=0.85, chapter_coherence=0.85, character_consistency=0.85, narrative_progression=0.85)

        r_loc_arith = aggregate_local_arithmetic(loc)
        r_loc_geo = aggregate_local_geometric(loc, epsilon=0.05)
        self.assertLess(r_loc_geo, r_loc_arith - 0.08)
        self.assertAlmostEqual(aggregate_global_arithmetic(glob), 0.85, places=3)
        self.assertGreater(aggregate_global_geometric(glob), 0.80)

        pen = ConstraintPenalties(p_length=0.0, p_rep=0.2, p_contra=0.5, p_format=0.0)
        bd_arith = aggregate_phase2_reward("hierarchical", loc, glob, constraints=pen)
        bd_geo = aggregate_phase2_reward("hierarchical_geo", loc, glob, constraints=pen)
        self.assertLess(bd_geo.r_total, bd_arith.r_total)
        # Verify exact penalty formula: 0.10*0.2 + 0.20*0.5 = 0.12
        self.assertAlmostEqual(bd_arith.penalty_total, 0.12, places=4)

    def test_all_six_yaml_configs_and_modes_run_cleanly(self) -> None:
        cfg_dir = REPO_ROOT / "rewards" / "configs"
        expected_files = [
            "exp00_baseline.yaml",
            "exp01_holistic.yaml",
            "exp02_hierarchical.yaml",
            "exp02_rlmr_orthogonal.yaml",
            "exp03_local_only.yaml",
            "exp04_global_only.yaml",
            "exp05_hierarchical_geo.yaml",
        ]
        item = _make_clean_chapter_input()
        for fname in expected_files:
            cfg = load_reward_config(cfg_dir / fname)
            cfg.setdefault("evaluator", {})["backend"] = "mock"
            cfg.setdefault("constraints", {})["use_embedder"] = False
            engine = Phase2RewardEngine(config=cfg)
            bds = engine.evaluate_batch([item])
            self.assertEqual(len(bds), 1, msg=f"Failed on {fname}")
            self.assertGreaterEqual(bds[0].r_total, -0.5)
            self.assertLessEqual(bds[0].r_total, 1.0)

    def test_perturbation_discrimination_and_validation_utilities(self) -> None:
        engine = Phase2RewardEngine(
            config={"mode": "hierarchical", "evaluator": {"backend": "mock", "single_pass": False}, "constraints": {"use_embedder": False}}
        )
        items = [_make_clean_chapter_input()]
        pert = evaluate_perturbation_discrimination(engine, items)
        self.assertTrue(pert["passed"], msg=f"Perturbation discrimination failed: {pert}")
        self.assertEqual(pert["clean_beats_filler_rate"], 1.0)
        self.assertEqual(pert["clean_beats_off_plan_rate"], 1.0)
        self.assertEqual(pert["clean_beats_contradictory_rate"], 1.0)

        # Correlation & human alignment helpers
        flat_logs = [
            {"r_total": 0.85, "r_local": 0.82, "r_global": 0.87, "event_realization": 0.85, "p_rep": 0.05},
            {"r_total": 0.60, "r_local": 0.50, "r_global": 0.66, "event_realization": 0.55, "p_rep": 0.15},
            {"r_total": 0.30, "r_local": 0.25, "r_global": 0.35, "event_realization": 0.20, "p_rep": 0.40},
        ]
        corr = analyze_reward_distributions_and_correlations(flat_logs)
        self.assertIn("r_total", corr["distributions"])
        align = compute_human_alignment_report([0.85, 0.60, 0.30], [5.0, 3.0, 1.0], [(0.85, 0.30, "A"), (0.30, 0.60, "B")])
        self.assertEqual(align["spearman_rho"], 1.0)
        self.assertEqual(align["kendall_tau"], 1.0)
        self.assertEqual(align["pairwise_accuracy"], 1.0)

    def test_rlmr_orthogonal_dynamic_shift_guarantees_negative_advantage(self) -> None:
        from rewards.aggregation import apply_rlmr_dynamic_shift
        from rewards.base import RewardBreakdown

        clean = _make_clean_chapter_input()
        # Group of 4 rollouts on the same prompt: o1 (compliant, 0.85), o2 (too short -> violating, 0.90),
        # o3 (compliant, 0.65), o4 (unclosed think -> violating, 0.60)
        o1 = _make_clean_chapter_input()
        o2 = _make_clean_chapter_input()
        o2.generated_chapter = "<think>short</think>\nOnly ten words here which strictly violates target length constraint."
        o3 = _make_clean_chapter_input()
        o4 = _make_clean_chapter_input()
        o4.generated_chapter = o1.generated_chapter + " </assistant>"

        items = [o1, o2, o3, o4]
        bds = [
            RewardBreakdown(r_total=0.85, r_quality=0.85, r_local=0.85, r_global=0.85, r_holistic=None, penalty_total=0.0, mode="hierarchical"),
            RewardBreakdown(r_total=0.90, r_quality=0.90, r_local=0.90, r_global=0.90, r_holistic=None, penalty_total=0.0, mode="hierarchical"),
            RewardBreakdown(r_total=0.65, r_quality=0.65, r_local=0.65, r_global=0.65, r_holistic=None, penalty_total=0.0, mode="hierarchical"),
            RewardBreakdown(r_total=0.60, r_quality=0.60, r_local=0.60, r_global=0.60, r_holistic=None, penalty_total=0.0, mode="hierarchical"),
        ]
        gamma = 0.05
        apply_rlmr_dynamic_shift(bds, items, gamma=gamma)

        # Exact expected delta = (4*0.90 + 4*0.05 - 3.00) / (4 - 2) = 0.40
        self.assertAlmostEqual(bds[0].r_total, 0.85, places=4)
        self.assertAlmostEqual(bds[2].r_total, 0.65, places=4)
        self.assertAlmostEqual(bds[1].r_total, 0.50, places=4)
        self.assertAlmostEqual(bds[3].r_total, 0.20, places=4)
        mean_adj = sum(b.r_total for b in bds) / 4.0
        # Every violating rollout strictly receives negative GRPO advantage (r'_j <= mean - gamma)
        self.assertLessEqual(bds[1].r_total, mean_adj - gamma + 1e-4)
        self.assertLessEqual(bds[3].r_total, mean_adj - gamma + 1e-4)
        # Compliant ordering preserved
        self.assertGreater(bds[0].r_total, bds[2].r_total)

        # k == n fallback: all violating -> no division by zero, better length compliance wins
        all_vio_items = [o2, o4]
        all_vio_bds = [
            RewardBreakdown(r_total=0.80, r_quality=0.80, r_local=0.80, r_global=0.80, r_holistic=None, penalty_total=0.0, mode="hierarchical"),
            RewardBreakdown(r_total=0.80, r_quality=0.80, r_local=0.80, r_global=0.80, r_holistic=None, penalty_total=0.0, mode="hierarchical"),
        ]
        apply_rlmr_dynamic_shift(all_vio_bds, all_vio_items, gamma=gamma)
        self.assertGreater(all_vio_bds[1].r_total, all_vio_bds[0].r_total)

    def test_cli_offline_audit_mock_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            stories_file = tmp_p / "stories.jsonl"
            item = _make_clean_chapter_input()
            row = {
                "premise_id": item.premise_id,
                "chapter_index": item.chapter_index,
                "target_words": item.target_words,
                "chapter_title": item.chapter_title,
                "chapter_goal": item.chapter_goal,
                "scene_outline": item.scene_outline,
                "characters_active": item.characters_active,
                "context_summary": item.context_summary,
                "prev_chapter_tail": item.prev_chapter_tail,
                "context_state": {"long_term_memory": item.long_term_memory},
                "generated_chapter": item.generated_chapter,
            }
            stories_file.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

            out_dir = tmp_p / "audit_out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "audit_rewards_offline.py"),
                    "--stories",
                    str(stories_file),
                    "--reward-config",
                    str(REPO_ROOT / "rewards" / "configs" / "exp02_hierarchical.yaml"),
                    "--output-dir",
                    str(out_dir),
                    "--mock",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
            self.assertTrue((out_dir / "offline_audit_summary.json").exists())
            self.assertTrue((out_dir / "offline_audit_report.md").exists())
            self.assertTrue((out_dir / "offline_reward_logs.jsonl").exists())

    def test_rlmr_verifier_allows_scene_headings_markdown_endings_and_implicit_think(self):
        from rewards.evaluators.constraint_evaluator import verify_objective_constraints

        # 1. Scene heading + markdown italic ending + implicit leading </think> should PASS V_format
        prose = "Scene 1: Arrival\n\n" + " ".join(
            f"Detailed narrative beat number {i} moves Elena and Marcus steadily toward the sealed harbor vault."
            for i in range(1, 22)
        ) + " *The lantern flickered out...*"
        implicit_text = "Plan three tight scenes within budget.\n</think>\n\n" + prose
        v_ok = verify_objective_constraints(
            EvaluationInput(generated_chapter=implicit_text, target_words=len(prose.split())),
            max_think_words=400,
        )
        self.assertTrue(v_ok["v_format"], msg=f"Unexpected format failure: {v_ok}")
        self.assertEqual(v_ok["think_words"], 6)
        self.assertTrue(v_ok["is_compliant"])

        # 2. Mid-sentence truncated ending MUST fail with explicit diagnostic reason
        trunc_text = "<think>Short plan.</think>\n\n" + " ".join(
            f"Sentence {i} builds suspense along the dock." for i in range(1, 35)
        ) + " And then she reached for the"
        v_trunc = verify_objective_constraints(
            EvaluationInput(generated_chapter=trunc_text, target_words=280),
            max_think_words=400,
        )
        self.assertFalse(v_trunc["v_format"])
        self.assertTrue(any("truncated_end" in f for f in v_trunc["failed_constraints"]), msg=v_trunc)


if __name__ == "__main__":
    unittest.main()
