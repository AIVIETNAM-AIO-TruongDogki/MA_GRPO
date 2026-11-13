import copy
import json
import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.planning import (
    OutlineAgent,
    PlanningAgent,
    STORY_ARC_STAGES,
    compute_story_arc_alignment,
)
from src.pipeline import MAGRPOPipeline, VLLMWritingPolicy, resolve_optimal_dtype

from experiments.conditions import (
    PLANNING_CONDITIONS,
    run_condition,
    plan_next_chapter_from_history,
)
from experiments.conditions.condition_a import run_condition_a
from experiments.conditions.condition_b import UnstructuredPlanner, plan_next_chapter_b, run_condition_b
from experiments.conditions.condition_c import LinearChapterPlanner, plan_next_chapter_c, run_condition_c
from experiments.conditions.condition_d import plan_next_chapter_d, run_condition_d
from experiments.conditions.condition_e import plan_next_chapter_e, run_condition_e
from experiments.perturbation import PERTURBATION_TYPES, MacroPlanPerturbator
from experiments.benchmark import (
    strip_length_constraints_from_premise,
    load_tell_me_a_story_premises,
    load_longstory_next_chapter_samples,
    NLIScorer,
    VRCLIPerplexityEvaluator,
    evaluate_macro_plan_group_a,
    evaluate_length_and_termination,
    evaluate_plan_adherence_doc_nli,
    evaluate_lexical_and_burstiness,
    LLMJudge5DimEvaluator,
    compute_perturbation_delta,
    audit_hypotheses_h1_to_h4,
    main as run_benchmark_main,
)


SAMPLE_PREMISE = (
    "In an orbital relay station losing oxygen above Saturn, archivist Mara Voss "
    "discovers that the station's emergency autopilot has been deleting crew survival "
    "logs to protect a corporate black-box cargo."
)

DONOR_PREMISE = (
    "On a flooded spice-trading archipelago, cartographer Ren Kael bargains with "
    "tidal spirits to recover a sunken treaty bell before the harbor freezes."
)


class TestRepositoryPaperBoundaries(unittest.TestCase):
    def test_src_contains_only_core_paper_modules(self):
        src_dir = os.path.join(PROJECT_ROOT, "src")
        py_files = sorted(f for f in os.listdir(src_dir) if f.endswith(".py"))
        expected_files = sorted([
            "__init__.py",
            "coordinator.py",
            "dataset_generator.py",
            "pipeline.py",
            "rewards.py",
        ])
        self.assertEqual(py_files, expected_files)

        sub_dirs = sorted(
            d for d in os.listdir(src_dir)
            if os.path.isdir(os.path.join(src_dir, d)) and not d.startswith("__")
        )
        self.assertEqual(sub_dirs, ["planning"])

        planning_files = sorted(
            f for f in os.listdir(os.path.join(src_dir, "planning")) if f.endswith(".py")
        )
        self.assertEqual(
            planning_files,
            sorted([
                "__init__.py",
                "outline_agent.py",
                "plan_critic.py",
                "planning_agent.py",
                "story_arc.py",
                "sub_event_agent.py",
            ]),
        )

    def test_conditions_abcde_have_separate_folders(self):
        cond_root = os.path.join(PROJECT_ROOT, "experiments", "conditions")
        for sub in ("condition_a", "condition_b", "condition_c", "condition_d", "condition_e"):
            self.assertTrue(os.path.isdir(os.path.join(cond_root, sub)), f"Missing folder {sub}")
            self.assertTrue(os.path.isfile(os.path.join(cond_root, sub, "__init__.py")))


class TestStoryArcAndPlanners(unittest.TestCase):
    def test_story_arc_constants_and_alignment(self):
        self.assertEqual(
            STORY_ARC_STAGES,
            ("Exposition", "Rising Action", "Climax", "Falling Action", "Resolution"),
        )
        self.assertEqual(PLANNING_CONDITIONS, ("A", "B", "C", "D", "E"))

        ordered_events = [
            {"id": "E1", "story_arc": "Exposition"},
            {"id": "E2", "story_arc": "Rising Action"},
            {"id": "E3", "story_arc": "Climax"},
            {"id": "E4", "story_arc": "Falling Action"},
            {"id": "E5", "story_arc": "Resolution"},
        ]
        res_ordered = compute_story_arc_alignment(ordered_events)
        self.assertAlmostEqual(res_ordered["stage_coverage"], 1.0, places=4)
        self.assertAlmostEqual(res_ordered["order_monotonicity"], 1.0, places=4)
        self.assertAlmostEqual(res_ordered["story_arc_score"], 1.0, places=4)
        self.assertEqual(len(res_ordered["sequence_violations"]), 0)
        self.assertEqual(res_ordered["missing_stages"], [])

        inverted_events = [
            {"id": "E1", "story_arc": "Resolution"},
            {"id": "E2", "story_arc": "Exposition"},
            {"id": "E3", "story_arc": "Climax"},
        ]
        res_inverted = compute_story_arc_alignment(inverted_events)
        self.assertLess(res_inverted["stage_coverage"], 1.0)
        self.assertLess(res_inverted["order_monotonicity"], 1.0)
        self.assertGreaterEqual(len(res_inverted["sequence_violations"]), 1)
        self.assertLess(res_inverted["story_arc_score"], res_ordered["story_arc_score"])

    def test_outline_and_ablation_planners_b_c_d_e(self):
        outline_agent = OutlineAgent()
        graph = outline_agent.create_event_graph(SAMPLE_PREMISE, genre="Sci-Fi")
        self.assertIn("events", graph)
        self.assertGreaterEqual(len(graph["events"]), 3)
        self.assertIn("story_arc_alignment", graph)
        for ev in graph["events"]:
            self.assertIn(ev.get("story_arc"), STORY_ARC_STAGES)

        # Condition B: Unstructured (separate condition_b folder)
        plan_b = UnstructuredPlanner().create_story_plan(
            SAMPLE_PREMISE, genre="Sci-Fi", total_target_words=8000, num_chapters=4
        )
        self.assertEqual(plan_b["condition"], "B")
        self.assertEqual(len(plan_b["chapters"]), 4)

        # Condition C: Linear Chapter-by-Chapter (separate condition_c folder)
        plan_c = LinearChapterPlanner().create_story_plan(
            SAMPLE_PREMISE, genre="Sci-Fi", total_target_words=8000, num_chapters=4
        )
        self.assertEqual(plan_c["condition"], "C")
        self.assertEqual(len(plan_c["chapters"]), 4)

        # Condition D (Macro-Only, skip SubEventAgent) vs Condition E (Macro + Micro)
        planner_d = PlanningAgent(enable_micro_subevents=False)
        planner_e = PlanningAgent(enable_micro_subevents=True)

        plan_d = planner_d.partition_into_chapters(
            copy.deepcopy(graph), total_target_words=8000, num_chapters=4
        )
        plan_e = planner_e.partition_into_chapters(
            copy.deepcopy(graph), total_target_words=8000, num_chapters=4
        )

        self.assertEqual(plan_d["condition"], "D")
        self.assertEqual(plan_e["condition"], "E")
        self.assertTrue(all(u.get("is_macro_only") for u in plan_d["sub_events"]))
        self.assertFalse(any(u.get("is_macro_only") for u in plan_e["sub_events"]))
        self.assertEqual(plan_d["planning_metadata"]["sub_event_tokens"], 0)
        self.assertGreater(plan_e["planning_metadata"]["sub_event_tokens"], 0)
        self.assertLess(
            plan_d["planning_metadata"]["planning_tokens"],
            plan_e["planning_metadata"]["planning_tokens"],
        )

    def test_track1_plan_next_chapter_per_condition_folder(self):
        history = (
            "Chapter 1: Mara checks the atmospheric scrubber and finds corrupted logs.\n"
            "Chapter 2: The station AI locks down Deck 3 when Mara queries Cargo Bay 9."
        )
        per_folder = {
            "B": plan_next_chapter_b,
            "C": plan_next_chapter_c,
            "D": plan_next_chapter_d,
            "E": plan_next_chapter_e,
        }
        for cond, fn in per_folder.items():
            res_direct = fn(history, premise=SAMPLE_PREMISE)
            res_disp = plan_next_chapter_from_history(history, condition=cond, premise=SAMPLE_PREMISE)
            self.assertEqual(res_direct["condition"], cond)
            self.assertEqual(res_disp["condition"], cond)
            self.assertTrue(res_disp["plan_text"].strip())
            self.assertIn("planning_tokens", res_disp)


class TestPerturbationProtocol(unittest.TestCase):
    def setUp(self):
        outline_agent = OutlineAgent()
        planner = PlanningAgent(enable_micro_subevents=False)
        self.graph = outline_agent.create_event_graph(SAMPLE_PREMISE, genre="Sci-Fi")
        self.blueprint = planner.partition_into_chapters(
            self.graph, total_target_words=8000, num_chapters=4
        )
        donor_graph = outline_agent.create_event_graph(DONOR_PREMISE, genre="Fantasy")
        self.donor_blueprint = planner.partition_into_chapters(
            donor_graph, total_target_words=8000, num_chapters=4
        )
        self.perturbator = MacroPlanPerturbator()

    def test_all_five_perturbations_without_mutating_clean_plan(self):
        snapshot = copy.deepcopy(self.blueprint)
        self.assertEqual(len(PERTURBATION_TYPES), 5)

        for ptype in PERTURBATION_TYPES:
            perturbed, meta = self.perturbator.apply_to_blueprint(
                self.blueprint,
                perturbation_type=ptype,
                donor_plan=self.donor_blueprint if ptype == "unrelated_plan_swapping" else None,
            )
            self.assertEqual(self.blueprint, snapshot)
            self.assertEqual(meta.get("perturbation_type"), ptype)
            self.assertNotEqual(perturbed, snapshot)

            if ptype == "causal_edge_removal":
                ev_deps = [
                    ev.get("depends_on")
                    for ev in perturbed.get("event_graph", {}).get("events", [])
                    if ev.get("depends_on")
                ]
                sub_deps = [
                    su.get("depends_on")
                    for su in perturbed.get("sub_events", [])
                    if su.get("depends_on")
                ]
                self.assertEqual(len(ev_deps), 0)
                self.assertEqual(len(sub_deps), 0)
            elif ptype == "unrelated_plan_swapping":
                orig_goals = [c.get("chapter_goal") for c in self.blueprint.get("chapters", [])]
                donor_goals = [c.get("chapter_goal") for c in self.donor_blueprint.get("chapters", [])]
                pert_goals = [c.get("chapter_goal") for c in perturbed.get("chapters", [])]
                self.assertEqual(pert_goals, donor_goals)
                self.assertNotEqual(pert_goals, orig_goals)
                pert_ortho, _ = self.perturbator.apply_to_blueprint(
                    self.blueprint, perturbation_type="unrelated_plan_swapping", donor_plan=None
                )
                self.assertNotEqual(
                    [c.get("chapter_title") for c in pert_ortho.get("chapters", [])],
                    [c.get("chapter_title") for c in self.blueprint.get("chapters", [])],
                )

        ctx = {
            "long_term_memory": {"characters": {"Mara": "Alive in Deck 2"}},
            "short_term_memory": {"recent_tail_excerpt": "Mara opened the hatch."},
            "plot_summary": "Mara heads to Deck 3.",
        }
        ch_plan = copy.deepcopy(self.blueprint["chapters"][0])
        p_ch, p_ctx, ch_meta = self.perturbator.apply_to_chapter_input(
            ch_plan, ctx, perturbation_type="context_conflict_injection", chapter_index=1
        )
        self.assertEqual(ch_meta["perturbation_type"], "context_conflict_injection")
        self.assertNotEqual((p_ch, p_ctx), (ch_plan, ctx))


class TestBenchmarkLoadersAndMetrics(unittest.TestCase):
    def test_strip_length_constraints_and_load_tell_me_a_story(self):
        raw = (
            "Write a sci-fi story between 1300-1400 words in third person past tense "
            "about a linguist decoding whale-like signals beneath Europa's ice. "
            "Make sure it is at least 900 words long."
        )
        cleaned = strip_length_constraints_from_premise(raw)
        self.assertNotIn("1300", cleaned)
        self.assertNotIn("900 words", cleaned.lower())
        self.assertIn("third person past tense", cleaned.lower())
        self.assertIn("Europa", cleaned)

        tmas_path = os.path.join(
            PROJECT_ROOT, "..", "Dataset", "tell_me_a_story_data", "tell-me-a-story-test.jsonl"
        )
        self.assertTrue(os.path.exists(tmas_path), f"Missing dataset at {tmas_path}")
        samples = load_tell_me_a_story_premises(
            data_path=tmas_path, max_samples=5, target_words=10000, strip_length_constraints=True
        )
        self.assertEqual(len(samples), 5)
        for s in samples:
            self.assertIn("premise", s)
            self.assertEqual(s["target_words"], 10000)
            self.assertTrue(len(s["premise"]) > 15)

    def test_longstory_loader_strict_vs_demo(self):
        missing_path = os.path.join(PROJECT_ROOT, "nonexistent_longstory.jsonl")
        with self.assertRaises(FileNotFoundError):
            load_longstory_next_chapter_samples(
                data_path=missing_path, max_samples=2, allow_synthetic_demo=False
            )

        demo_samples = load_longstory_next_chapter_samples(
            data_path=missing_path, max_samples=2, allow_synthetic_demo=True
        )
        self.assertEqual(len(demo_samples), 2)
        for item in demo_samples:
            self.assertTrue(item["history_text"])
            self.assertTrue(item["gold_chapter"])

    def test_group_a_vrcli_group_b_judge_and_h1_to_h4_audit(self):
        nli = NLIScorer(use_mock=True)
        outline_agent = OutlineAgent()
        planner = PlanningAgent(enable_micro_subevents=False)
        graph = outline_agent.create_event_graph(SAMPLE_PREMISE, genre="Sci-Fi")
        blueprint = planner.partition_into_chapters(graph, total_target_words=8000, num_chapters=4)

        group_a = evaluate_macro_plan_group_a(SAMPLE_PREMISE, blueprint, nli_scorer=nli)
        for key in (
            "S_cov",
            "S_prog",
            "S_causal",
            "dag_connectivity",
            "S_comp",
            "S_arc",
            "arc_stage_coverage",
            "arc_order_monotonicity",
            "planning_tokens",
            "planning_latency_s",
        ):
            self.assertIn(key, group_a)

        vrcli = VRCLIPerplexityEvaluator(use_mock=True)
        vr_res = vrcli.evaluate_information_gain(
            history_text="Chapter 1 happened on the station.",
            plan_text=blueprint["chapters"][0].get("chapter_goal", "Mara investigates the AI."),
            gold_chapter="Mara stepped into the silent corridor of Deck 3 and inspected the black box.",
        )
        self.assertIn("ppl_base", vr_res)
        self.assertIn("ppl_planned", vr_res)
        self.assertIn("I_pi", vr_res)

        chapter_texts = [
            (ch.get("chapter_goal", "") + " ") * 35 + "Mara walked carefully toward Deck 3."
            for ch in blueprint["chapters"]
        ]
        full_text = "\n\n".join(chapter_texts)

        len_res = evaluate_length_and_termination(full_text, target_words=800)
        self.assertIn("actual_words", len_res)
        self.assertIn("length_score_sl", len_res)
        self.assertIn("early_terminated", len_res)

        adh_res = evaluate_plan_adherence_doc_nli(
            chapter_texts, blueprint, nli_scorer=nli, premise=SAMPLE_PREMISE
        )
        self.assertIn("S_adh", adh_res)
        self.assertGreaterEqual(adh_res["total_goals"], 1)

        lex_res = evaluate_lexical_and_burstiness(full_text)
        self.assertIn("Distinct-3 (%)", lex_res)
        self.assertIn("Distinct-4 (%)", lex_res)
        self.assertIn("shannon_entropy", lex_res)
        self.assertIn("burstiness_std", lex_res)

        judge_disabled = LLMJudge5DimEvaluator(judge_fn=None)
        disabled_res = judge_disabled.evaluate(SAMPLE_PREMISE, "plan", full_text)
        self.assertIsNone(disabled_res["S_coh"])
        self.assertIsNone(disabled_res["C_weighted"])

        def mock_judge_fn(sys_p, user_p):
            return json.dumps({
                "story_coherence": 4.2,
                "plan_adherence": 4.5,
                "completeness": 4.0,
                "writing_quality": 3.9,
                "global_consistency": 4.1,
                "n_fatal_inconsistencies": 0,
                "n_moderate_inconsistencies": 1,
                "n_minor_inconsistencies": 1,
                "perturbation_behavior": "Clean",
                "rationale": "Strong structural progression.",
            })

        judge_enabled = LLMJudge5DimEvaluator(judge_fn=mock_judge_fn)
        j_res = judge_enabled.evaluate(SAMPLE_PREMISE, "plan", full_text)
        self.assertAlmostEqual(j_res["S_coh"], 4.2, places=3)
        self.assertAlmostEqual(j_res["judge_plan_adherence"], 4.5, places=3)
        self.assertEqual(j_res["C_weighted"], 3)

        clean_rec = {
            "S_adh": 0.85,
            "S_coh": 4.2,
            "S_arc": 0.90,
            "S_causal": 0.88,
            "C_weighted": 1,
        }
        pert_rec = {
            "perturbation_type": "causal_edge_removal",
            "S_adh": 0.62,
            "S_coh": 3.0,
            "S_arc": 0.50,
            "S_causal": 0.30,
            "C_weighted": 6,
        }
        delta = compute_perturbation_delta(clean_rec, pert_rec)
        self.assertGreater(delta["delta_S_adh"], 0.15)
        self.assertGreater(delta["delta_S_coh"], 0.5)
        self.assertLess(delta["delta_C_weighted"], 0.0)
        self.assertIn("Plan-Sensitive", delta["diagnosis"])

        summary_by_cond = {
            "A": {"actual_words": 2100, "S_causal": 0.0, "S_arc": 0.0, "S_adh": 0.0, "planning_tokens": 0, "Distinct-3 (%)": 52.0},
            "B": {"actual_words": 7200, "S_causal": 0.35, "S_arc": 0.30, "S_adh": 0.60, "planning_tokens": 180, "Distinct-3 (%)": 46.0},
            "C": {"actual_words": 8100, "S_causal": 0.50, "S_arc": 0.45, "S_adh": 0.72, "planning_tokens": 310, "Distinct-3 (%)": 44.0},
            "D": {"actual_words": 9500, "S_causal": 0.85, "S_arc": 0.90, "S_adh": 0.86, "planning_tokens": 380, "Distinct-3 (%)": 43.0},
            "E": {"actual_words": 9600, "S_causal": 0.84, "S_arc": 0.90, "S_adh": 0.85, "planning_tokens": 920, "Distinct-3 (%)": 42.5},
        }
        h_audit = audit_hypotheses_h1_to_h4(summary_by_cond)
        self.assertTrue(h_audit["H1_length_ceiling_broken"]["supported"])
        self.assertTrue(h_audit["H2_macro_graph_advantage"]["supported"])
        self.assertTrue(h_audit["H3_minimum_necessary_planning"]["supported"])
        self.assertTrue(h_audit["H4_diversity_drop_justifying_rl"]["observed_drop"])


class TestPipelineEndToEndConditions(unittest.TestCase):
    def test_resolve_optimal_dtype_and_all_condition_runners(self):
        dtype = resolve_optimal_dtype("auto")
        self.assertIn(dtype, ("bfloat16", "float16"))

        policy = VLLMWritingPolicy(use_mock=True)
        pipeline = MAGRPOPipeline(writing_policy=policy)

        direct_map = {
            "A": run_condition_a,
            "B": run_condition_b,
            "C": run_condition_c,
            "D": run_condition_d,
            "E": run_condition_e,
        }
        for cond, runner_fn in direct_map.items():
            out_direct = runner_fn(
                pipeline=pipeline,
                premise=SAMPLE_PREMISE,
                genre="Sci-Fi",
                total_target_words=2400,
                num_chapters=4,
            )
            out_disp = run_condition(
                pipeline=pipeline,
                condition=cond,
                premise=SAMPLE_PREMISE,
                genre="Sci-Fi",
                total_target_words=2400,
                num_chapters=4,
            )
            self.assertEqual(out_direct["condition"], cond)
            self.assertEqual(out_disp["condition"], cond)
            self.assertTrue(out_disp["full_text"].strip())
            self.assertGreater(out_disp["actual_total_words"], 0)
            self.assertIsNone(out_disp["perturbation_metadata"])

        # Condition D with perturbation
        out_pert = run_condition(
            pipeline=pipeline,
            condition="D",
            premise=SAMPLE_PREMISE,
            genre="Sci-Fi",
            total_target_words=2400,
            num_chapters=4,
            perturbation_type="causal_edge_removal",
        )
        self.assertEqual(out_pert["condition"], "D")
        self.assertIsNotNone(out_pert["perturbation_metadata"])
        self.assertEqual(
            out_pert["perturbation_metadata"]["perturbation_type"], "causal_edge_removal"
        )

    def test_end_to_end_benchmark_runner_and_notebook_valid(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_benchmark_main([
                "--mode", "all",
                "--backend", "mock",
                "--track1-samples", "2",
                "--track2-samples", "2",
                "--perturb-samples", "2",
                "--target-words", "2400",
                "--num-chapters", "4",
                "--output-dir", tmp_dir,
            ])
            for fname in (
                "track1_vrcli_results.csv",
                "track2_ablation_5conditions.csv",
                "perturbation_delta_results.csv",
                "benchmark_summary.json",
                "macro_planning_benchmark_report.md",
            ):
                fpath = os.path.join(tmp_dir, fname)
                self.assertTrue(os.path.exists(fpath), f"Missing output artifact: {fpath}")
                self.assertGreater(os.path.getsize(fpath), 20)

        nb_path = os.path.join(PROJECT_ROOT, "notebooks", "macro_planning_benchmark.ipynb")
        self.assertTrue(os.path.exists(nb_path))
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)
        self.assertEqual(nb_data.get("nbformat"), 4)
        self.assertGreaterEqual(len(nb_data.get("cells", [])), 8)

    def test_vllm_clamp_max_tokens_respects_max_model_len(self):
        from src.pipeline import VLLMWritingPolicy

        class _StubTokenizer:
            @staticmethod
            def encode(text: str, add_special_tokens: bool = False):
                return text.split()

        class _StubLLM:
            @staticmethod
            def get_tokenizer():
                return _StubTokenizer()

        policy = VLLMWritingPolicy(max_model_len=8192, use_mock=True)
        policy.llm = _StubLLM()
        prompt_200 = " ".join(["word"] * 200)
        clamped = policy._clamp_max_tokens(prompt_200, 8192)
        self.assertEqual(clamped, 8192 - 200 - 32)
        self.assertLessEqual(200 + clamped, 8192)

        # Max-ping uncapped default (32768) still stays strictly inside max_model_len
        policy_32k = VLLMWritingPolicy(use_mock=True)
        self.assertEqual(policy_32k.max_model_len, 32768)
        policy_32k.llm = _StubLLM()
        self.assertEqual(policy_32k._clamp_max_tokens(prompt_200, 32768), 32768 - 200 - 32)

    def test_local_judge_resolution_and_json_resilience(self):
        from experiments.benchmark.evaluator import LLMJudge5DimEvaluator
        from experiments.benchmark.runner import parse_args, resolve_judge_runtime

        # Auto-select local_vllm with Qwen2.5-32B-Instruct-AWQ when OPENAI_API_KEY is absent
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            args = parse_args(["--backend", "vllm", "--enable-judge"])
            jb, jm = resolve_judge_runtime(args)
            self.assertEqual(jb, "local_vllm")
            self.assertEqual(jm, "Qwen/Qwen2.5-32B-Instruct-AWQ")
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key

        # Tolerate markdown fences and recover cleanly from invalid JSON
        fenced_judge = LLMJudge5DimEvaluator(
            judge_fn=lambda _s, _u: "```json\n{\"story_coherence\": 4.5, \"plan_adherence\": 4, \"completeness\": 4, \"writing_quality\": 4, \"global_consistency\": 5, \"n_fatal_inconsistencies\": 0, \"n_moderate_inconsistencies\": 1, \"n_minor_inconsistencies\": 2, \"perturbation_behavior\": \"Clean\"}\n```"
        )
        res_ok = fenced_judge.evaluate("premise", "plan", "story text")
        self.assertEqual(res_ok["S_coh"], 4.5)
        self.assertEqual(res_ok["C_weighted"], 4)

        broken_judge = LLMJudge5DimEvaluator(judge_fn=lambda _s, _u: "not valid json")
        res_bad = broken_judge.evaluate("premise", "plan", "story text")
        self.assertIsNone(res_bad["S_coh"])

        # End-to-end mock run with judge enabled produces populated S_coh & C_weighted
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_benchmark_main([
                "--mode", "track2",
                "--backend", "mock",
                "--track2-samples", "1",
                "--conditions", "A", "D",
                "--target-words", "1200",
                "--num-chapters", "2",
                "--enable-judge",
                "--judge-backend", "mock",
                "--output-dir", tmp_dir,
            ])
            with open(os.path.join(tmp_dir, "benchmark_summary.json"), encoding="utf-8") as f:
                summary = json.load(f)
            self.assertTrue(summary["config"]["judge_enabled"])
            self.assertEqual(summary["config"]["judge_backend"], "mock")
            self.assertIsNotNone(summary["track2_summary"]["D"]["S_coh"])
            self.assertIsNotNone(summary["track2_summary"]["D"]["C_weighted"])


if __name__ == "__main__":
    unittest.main()
