"""Unit & end-to-end CLI tests for Suite 1 (MacroPlan-Bench) and Suite 2 (FullStory-Bench)."""

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

from experiments.benchmark.evaluator import NLIScorer
from experiments.benchmarks.fullstory_bench.constory_checker import (
    CONSTORY_TAXONOMY,
    ConStoryCheckerEvaluator,
)
from experiments.benchmarks.fullstory_bench.eqbench_evaluator import (
    BlindPairwiseStoryJudge,
    EQBenchLongformEvaluator,
)
from experiments.benchmarks.fullstory_bench.metrics_adherence import (
    compute_intra_chapter_redundancy,
    compute_reasoning_and_length,
    compute_slop_score,
    compute_transition_continuity,
)
from experiments.benchmarks.outline_bench.judge_outline import (
    OutlineJudgeEvaluator,
    compute_spearman_correlation,
    run_outline_perturbation_sanity_check,
)
from experiments.benchmarks.outline_bench.metrics_pacing import evaluate_concreteness_and_pacing


def _make_clean_blueprint() -> dict:
    events = [
        {
            "event_id": "E1",
            "title": "Flooded Vault Discovery",
            "setting": "Submerged Venetian archive beneath San Marco during autumn storm",
            "character": "Archivist Mara Valeri and locksmith Lorenzo",
            "action": "Mara unlocks the rusted iron bulkhead and retrieves a sealed bronze clockwork cylinder",
            "conflict": "Rising canal water threatens to drown the basement while council guards approach",
            "plot_twist": "The cylinder bears Mara's vanished grandfather's cipher mark and a midnight warning",
            "depends_on": [],
        },
        {
            "event_id": "E2",
            "title": "Workshop Decoding Ambush",
            "setting": "Candlelit restoration loft in Dorsoduro overlooking the canal",
            "character": "Mara Valeri, Lorenzo, and Council Inspector Bellini",
            "action": "Mara decodes perforated copper scrolls revealing an ancient tidal sluice override",
            "conflict": "Inspector Bellini raids the workshop demanding the cylinder under maritime law",
            "plot_twist": "Lorenzo secretly palms the second scroll and fakes a fire to buy Mara an escape",
            "depends_on": ["E1"],
        },
        {
            "event_id": "E3",
            "title": "Arsenal Sluice Sabotage",
            "setting": "Stone sluice gallery inside the old Venetian Arsenal at midnight",
            "character": "Mara Valeri and chief engineer Contarini",
            "action": "Mara infiltrates the sluice chamber and discovers Contarini rigging the floodgates",
            "conflict": "Opening the gates prematurely will submerge the lower wards to hide stolen ledgers",
            "plot_twist": "Contarini reveals Mara's grandfather built the mechanism specifically to expose him",
            "depends_on": ["E2"],
        },
        {
            "event_id": "E4",
            "title": "Midnight Tide Resolution",
            "setting": "Lagoon seawall and bell tower above San Marco at dawn",
            "character": "Mara Valeri, Lorenzo, Bellini, and Magistrate Rizzo",
            "action": "Mara engages the mechanical counterweight and broadcasts the ledger chimes across the harbor",
            "conflict": "Storm surge strains the brass gears as Contarini attempts to sever the tension chain",
            "plot_twist": "Bellini arrests Contarini after hearing the confession, restoring civic equilibrium and closure",
            "depends_on": ["E3"],
        },
    ]
    chapters = [
        {
            "chapter_index": i + 1,
            "chapter_title": ev["title"],
            "chapter_goal": ev["action"],
            "required_beats": [ev["conflict"], ev["plot_twist"]],
            "exit_state": ev["plot_twist"],
            "characters_active": [ev["character"]],
            "target_words": 1000,
        }
        for i, ev in enumerate(events)
    ]
    return {
        "event_graph": {"events": events},
        "chapters": chapters,
        "planning_metadata": {"planning_tokens": 420},
    }


class TestSuite1OutlineBench(unittest.TestCase):
    def test_concreteness_and_pacing_detects_sparse_erratic_outline(self) -> None:
        clean_bp = _make_clean_blueprint()
        erratic_bp = _make_clean_blueprint()
        for idx in (1, 2):
            ev = erratic_bp["event_graph"]["events"][idx]
            ev["setting"] = ""
            ev["character"] = ""
            ev["action"] = "Stuff happens."
            ev["conflict"] = ""
            ev["plot_twist"] = ""

        clean_res = evaluate_concreteness_and_pacing(clean_bp)
        erratic_res = evaluate_concreteness_and_pacing(erratic_bp)

        self.assertGreater(clean_res["mean_concreteness"], erratic_res["mean_concreteness"])
        self.assertGreater(clean_res["pacing_balance_score"], erratic_res["pacing_balance_score"])
        self.assertEqual(clean_res["sparse_event_ratio"], 0.0)
        self.assertGreater(erratic_res["sparse_event_ratio"], 0.0)
        self.assertGreater(clean_res["inter_event_diversity"], 0.5)

    def test_outline_judge_pointwise_and_swapped_pairwise(self) -> None:
        bp_a = _make_clean_blueprint()
        bp_b = _make_clean_blueprint()
        bp_b["event_graph"]["events"] = bp_b["event_graph"]["events"][:2]

        # Consistent judge always picks longer outline
        def _consistent_judge(_s: str, u: str) -> str:
            if "[OUTLINE A]:" in u:
                part_a = u.split("[OUTLINE A]:", 1)[1].split("[OUTLINE B]:", 1)[0]
                part_b = u.split("[OUTLINE B]:", 1)[1]
                return json.dumps({"winner": "A" if len(part_a) > len(part_b) else "B", "rationale": "More complete."})
            return json.dumps(
                {
                    "outline_relevance": 4.5,
                    "plot_coherence": 4.7,
                    "surprise_and_twists": 4.2,
                    "engagement_and_complexity": 4.6,
                    "weakest_chapter_index": 3,
                    "evidence": [],
                    "rationale": "Strong arc.",
                }
            )

        judge = OutlineJudgeEvaluator(judge_fn=_consistent_judge)
        pt = judge.evaluate("Venice clockwork mystery", bp_a)
        self.assertEqual(pt["judge_outline_overall"], 4.5)

        pw = judge.compare_pair("Venice clockwork mystery", bp_a, bp_b, label_a="cond_d", label_b="cond_b")
        self.assertEqual(pw["winner"], "cond_d")
        self.assertTrue(pw["position_consistent"])

        # Position-biased judge always picks "A" -> must resolve to "tie"
        biased = OutlineJudgeEvaluator(judge_fn=lambda _s, _u: json.dumps({"winner": "A", "rationale": "First bias"}))
        pw_biased = biased.compare_pair("Venice", bp_a, bp_b, label_a="cond_d", label_b="cond_b")
        self.assertEqual(pw_biased["winner"], "tie")
        self.assertFalse(pw_biased["position_consistent"])

    def test_spearman_and_perturbation_sanity_check(self) -> None:
        self.assertEqual(compute_spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertEqual(compute_spearman_correlation([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

        nli = NLIScorer(use_mock=True)
        sanity = run_outline_perturbation_sanity_check([_make_clean_blueprint()], nli_scorer=nli)
        self.assertTrue(sanity["passed"], msg=f"Sanity check failed: {sanity}")
        self.assertEqual(sanity["causal_drop_rate"], 1.0)
        self.assertEqual(sanity["duplicate_drop_rate"], 1.0)
        self.assertEqual(sanity["sparse_pacing_drop_rate"], 1.0)


class TestSuite2FullStoryBench(unittest.TestCase):
    def test_slop_redundancy_continuity_and_think_budget(self) -> None:
        sloppy_text = (
            "A shiver down her spine made her let out a breath she didn't know she was holding. "
            "It was a testament to the tapestry of fate, and her knuckles turned white in the dim light."
        )
        slop = compute_slop_score(sloppy_text)
        self.assertGreaterEqual(slop["slop_hits"], 4)
        self.assertGreater(slop["slop_score_per_1k"], 50.0)

        # Repeated identical paragraphs -> high redundancy
        para = " ".join(["clockwork", "venice", "archive", "brass", "lantern", "harbor", "cylinder", "cipher"] * 4)
        dup_ch = f"{para}\n\n{para}"
        diverse_ch = (
            " ".join(["clockwork", "venice", "archive", "brass", "lantern", "harbor", "cylinder", "cipher"] * 3)
            + "\n\n"
            + " ".join(["mountain", "glacier", "observatory", "telescope", "spectrum", "asteroid", "orbit", "beacon"] * 3)
        )
        r_dup = compute_intra_chapter_redundancy([dup_ch])["intra_chapter_redundancy"]
        r_div = compute_intra_chapter_redundancy([diverse_ch])["intra_chapter_redundancy"]
        self.assertGreater(r_dup, r_div)

        # Continuity & think budget
        raw_ch1 = "<think>Plan the opening scene carefully in Venice.</think>\nMara stepped into the flooded Venetian vault holding a bronze lamp."
        raw_ch2 = "<think>Continue directly from the bronze lamp discovery.</think>\nWith the bronze lamp steadied, Mara studied the flooded Venetian vault doorway."
        len_res = compute_reasoning_and_length([raw_ch1, raw_ch2], [raw_ch1, raw_ch2], target_words=40)
        self.assertGreater(len_res["think_words"], 10)
        self.assertGreater(len_res["actual_words"], 15)

        transitions = [
            {
                "chapter_index": 2,
                "final_summary": "Mara unlocked the iron gate and spotted a cipher.",
                "is_fallback": True,
            }
        ]
        cont = compute_transition_continuity([raw_ch1, raw_ch2], transitions=transitions)
        self.assertIsNotNone(cont["transition_cosine"])
        self.assertIsNotNone(cont["anti_context_novelty"])
        self.assertEqual(cont["fallback_rate_pct"], 100.0)

    def test_constory_checker_taxonomy_and_weighted_score(self) -> None:
        self.assertEqual(len(CONSTORY_TAXONOMY), 5)
        self.assertEqual(sum(len(v) for v in CONSTORY_TAXONOMY.values()), 19)

        def _fake_checker(_s: str, _u: str) -> str:
            return json.dumps(
                {
                    "errors": [
                        {
                            "category": "characterization",
                            "subtype": "memory_contradiction",
                            "severity": "fatal",
                            "chapter_a": 1,
                            "chapter_b": 6,
                            "quote_a": "Lorenzo died in the collapse.",
                            "quote_b": "Lorenzo greeted her at the pier.",
                            "explanation": "Dead character reappears alive.",
                        },
                        {
                            "category": "world_setting",
                            "subtype": "geography_spatial_contradiction",
                            "severity": "moderate",
                            "chapter_a": 2,
                            "chapter_b": 5,
                            "quote_a": "The tower stands north.",
                            "quote_b": "The tower stands south.",
                            "explanation": "Tower direction inverted.",
                        },
                        {
                            "category": "factual_detail",
                            "subtype": "physical_appearance",
                            "severity": "minor",
                            "chapter_a": 1,
                            "chapter_b": 3,
                            "quote_a": "Her red scarf.",
                            "quote_b": "Her blue scarf.",
                            "explanation": "Scarf color changed.",
                        },
                    ],
                    "summary": "3 errors found.",
                }
            )

        checker = ConStoryCheckerEvaluator(judge_fn=_fake_checker)
        res = checker.evaluate("Premise", ["Chapter one text " * 200, "Chapter two text " * 200], actual_words=1000)
        self.assertEqual(res["constory_total_errors"], 3)
        self.assertEqual(res["constory_errors_per_1k"], 3.0)
        self.assertEqual(res["C_weighted"], 3 * 1 + 2 * 1 + 1 * 1)
        self.assertEqual(res["err_characterization"], 1)
        self.assertEqual(res["err_world_setting"], 1)
        self.assertEqual(res["err_factual_detail"], 1)

    def test_eqbench_evaluator_and_blind_pairwise_story_judge(self) -> None:
        def _fake_eq(_s: str, _u: str) -> str:
            return json.dumps(
                {
                    "nuanced_characters": 8.0,
                    "emotionally_engaging": 8.0,
                    "compelling_plot": 8.0,
                    "coherence": 8.0,
                    "prose_quality": 8.0,
                    "meaningful_elaboration": 8.0,
                    "early_chapters_score": 8.5,
                    "late_chapters_score": 7.0,
                    "rationale": "Solid opening, slight late dip.",
                }
            )

        ev = EQBenchLongformEvaluator(judge_fn=_fake_eq)
        res = ev.evaluate("Premise", ["Ch1", "Ch2"], slop_score_per_1k=2.0)
        self.assertEqual(res["eq_degradation_penalty"], 1.5)
        # 80.0 - (1.5 * 4.0) - (2.0 * 1.5) = 71.0
        self.assertEqual(res["eqbench_score"], 71.0)

        pw = BlindPairwiseStoryJudge(judge_fn=lambda _s, _u: json.dumps({"winner": "A", "winning_axes": ["prose"], "rationale": "bias"}))
        verdict = pw.compare_pair("Premise", ["Story A"], ["Story B"], label_a="arm_a", label_b="arm_b")
        self.assertEqual(verdict["winner"], "tie")
        self.assertFalse(verdict["position_consistent"])


class TestCLIEndToEndSmoke(unittest.TestCase):
    def test_cli_outline_and_fullstory_mock_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bp_a = _make_clean_blueprint()
            bp_b = _make_clean_blueprint()
            bp_b["event_graph"]["events"] = bp_b["event_graph"]["events"][:3]

            # 1. Suite 1 inputs: shared_macro_plans.json + second group JSON
            plans_a = tmp_path / "plans_cond_d.json"
            plans_b = tmp_path / "plans_cond_b.json"
            plans_a.write_text(
                json.dumps([{"item": {"id": "mops_01", "genre": "Mystery", "premise": "Mara unlocks a Venetian clockwork vault."}, "story_plan": bp_a}]),
                encoding="utf-8",
            )
            plans_b.write_text(
                json.dumps([{"item": {"id": "mops_01", "genre": "Mystery", "premise": "Mara unlocks a Venetian clockwork vault."}, "story_plan": bp_b}]),
                encoding="utf-8",
            )
            out_s1 = tmp_path / "out_outline"
            cmd_s1 = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "benchmark_outline.py"),
                "--inputs",
                str(plans_a),
                str(plans_b),
                "--labels",
                "cond_d",
                "cond_b",
                "--output-dir",
                str(out_s1),
                "--mock",
                "--judge-backend",
                "mock",
                "--pairwise-groups",
                "cond_d",
                "cond_b",
                "--run-sanity-check",
            ]
            proc1 = subprocess.run(cmd_s1, capture_output=True, text=True, check=False)
            self.assertEqual(proc1.returncode, 0, msg=proc1.stderr or proc1.stdout)
            self.assertTrue((out_s1 / "outline_benchmark_summary.csv").exists())
            self.assertTrue((out_s1 / "outline_benchmark_report.md").exists())
            self.assertTrue((out_s1 / "outline_pairwise.json").exists())
            self.assertTrue((out_s1 / "outline_sanity_check.json").exists())

            # 2. Suite 2 inputs: stories_7b_outputs.jsonl + coordinator_transitions.jsonl
            stories_file = tmp_path / "stories_7b_outputs.jsonl"
            trans_file = tmp_path / "coordinator_transitions.jsonl"
            ch1 = "Mara stepped into the submerged Venetian archive beneath San Marco holding her lantern high as dark canal water rose."
            ch2 = "Holding her lantern steady above the dark canal water, Mara decoded the perforated copper scrolls inside the workshop."
            rows = [
                {
                    "arm": "validator_0.80",
                    "premise_id": "mops_01",
                    "genre": "Mystery",
                    "premise": "Mara unlocks a Venetian clockwork vault.",
                    "target_words": 200,
                    "chapters": [ch1, ch2],
                    "chapters_raw_with_think": [f"<think>Draft ch1</think>\n{ch1}", f"<think>Draft ch2</think>\n{ch2}"],
                    "story_plan": bp_a,
                },
                {
                    "arm": "no_validator",
                    "premise_id": "mops_01",
                    "genre": "Mystery",
                    "premise": "Mara unlocks a Venetian clockwork vault.",
                    "target_words": 200,
                    "chapters": [ch1, ch2 + " contradiction_trigger"],
                    "chapters_raw_with_think": [ch1, ch2 + " contradiction_trigger"],
                    "story_plan": bp_a,
                },
            ]
            stories_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            trans_file.write_text(
                json.dumps(
                    {
                        "arm": "validator_0.80",
                        "premise_id": "mops_01",
                        "chapter_index": 2,
                        "final_summary": "Mara unlocked the iron vault and retrieved a bronze cylinder.",
                        "is_fallback": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            out_s2 = tmp_path / "out_fullstory"
            cmd_s2 = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "benchmark_fullstory.py"),
                "--stories",
                str(stories_file),
                "--transitions",
                str(trans_file),
                "--output-dir",
                str(out_s2),
                "--mock",
                "--judge-backend",
                "mock",
                "--pairwise-arms",
                "validator_0.80",
                "no_validator",
            ]
            proc2 = subprocess.run(cmd_s2, capture_output=True, text=True, check=False)
            self.assertEqual(proc2.returncode, 0, msg=proc2.stderr or proc2.stdout)
            self.assertTrue((out_s2 / "fullstory_summary.csv").exists())
            self.assertTrue((out_s2 / "fullstory_benchmark_report.md").exists())
            self.assertTrue((out_s2 / "constory_errors.jsonl").exists())
            self.assertTrue((out_s2 / "pairwise_verdicts.json").exists())

            err_lines = [l for l in (out_s2 / "constory_errors.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(err_lines), 1)

            # 3. ClassicNLG-Bench CLI smoke test (+ comparison with Suite 2 fullstory_raw.jsonl)
            ref_file = tmp_path / "gold_refs.jsonl"
            ref_file.write_text(
                json.dumps({"example_id": "mops_01", "targets": f"{ch1} {ch2}"}) + "\n",
                encoding="utf-8",
            )
            out_classic = tmp_path / "out_classic"
            cmd_classic = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "benchmark_classic_nlg.py"),
                "--stories",
                str(stories_file),
                "--reference-file",
                str(ref_file),
                "--compare-with-fullstory",
                str(out_s2 / "fullstory_raw.jsonl"),
                "--output-dir",
                str(out_classic),
                "--mock",
            ]
            proc3 = subprocess.run(cmd_classic, capture_output=True, text=True, check=False)
            self.assertEqual(proc3.returncode, 0, msg=proc3.stderr or proc3.stdout)
            self.assertTrue((out_classic / "classic_nlg_summary.csv").exists())
            self.assertTrue((out_classic / "classic_nlg_report.md").exists())
            self.assertTrue((out_classic / "classic_vs_fullstory_correlation.json").exists())


class TestClassicNLGBenchmark(unittest.TestCase):
    def test_bleu_rouge_distinct_entropy_and_perplexity(self) -> None:
        from experiments.benchmarks.classic_nlg_bench.metrics_classic import (
            SlidingWindowPerplexityScorer,
            compute_bleu_1_to_4,
            compute_distinct_and_entropy,
            compute_rouge_l,
            evaluate_classic_nlg_single,
        )

        ref = "Mara unlocked the ancient Venetian brass vault beneath San Marco before the midnight tide flooded the archive."
        exact_bleu = compute_bleu_1_to_4(ref, ref)
        self.assertGreater(exact_bleu["BLEU-1"], 99.0)
        self.assertGreater(exact_bleu["BLEU-4"], 99.0)
        self.assertEqual(compute_rouge_l(ref, ref), 100.0)

        # Unrelated candidate must score near 0 on BLEU-1 and ROUGE-L
        unrelated = "Quantum satellites orbited distant frozen moons across empty interstellar vacuum."
        bad_bleu = compute_bleu_1_to_4(unrelated, ref)
        self.assertLess(bad_bleu["BLEU-1"], 5.0)
        self.assertEqual(compute_rouge_l(unrelated, ref), 0.0)

        # Distinct-1..3 & Entropy-2 distinguish diverse vs repetitive text
        rep_text = "the dark wind blew cold across the stone " * 12
        div_stats = compute_distinct_and_entropy(ref + " " + unrelated)
        rep_stats = compute_distinct_and_entropy(rep_text)
        self.assertGreater(div_stats["Distinct-1 (%)"], rep_stats["Distinct-1 (%)"])
        self.assertGreater(div_stats["Distinct-2 (%)"], rep_stats["Distinct-2 (%)"])
        self.assertGreater(div_stats["Distinct-3 (%)"], rep_stats["Distinct-3 (%)"])
        self.assertGreater(div_stats["Entropy-2"], rep_stats["Entropy-2"])

        scorer = SlidingWindowPerplexityScorer(use_mock=True)
        full_metrics = evaluate_classic_nlg_single(ref, ref, ppl_scorer=scorer)
        for key in (
            "BLEU-1",
            "BLEU-2",
            "BLEU-3",
            "BLEU-4",
            "ROUGE-L",
            "Distinct-1 (%)",
            "Distinct-2 (%)",
            "Distinct-3 (%)",
            "Entropy-2",
            "Perplexity",
        ):
            self.assertIsNotNone(full_metrics[key], msg=f"Missing metric {key}")


if __name__ == "__main__":
    unittest.main()
