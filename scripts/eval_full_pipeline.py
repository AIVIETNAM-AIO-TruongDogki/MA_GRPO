"""Standalone Linux/Server End-to-End Pipeline Evaluator: Base 7B vs Trained GRPO LoRA Checkpoint.

Zero Google Colab dependencies. Runs all 5 phases with automatic resume:
  Phase 1: Generate N shared Condition D Macro Plans (HF 4-bit 32B + partition_into_chapters + Suite 1 audit)
  Phase 2: Generate full 8-chapter (8,000w) stories for `base_7b` vs `<ckpt_arm>` (native bf16 + PEFT LoRA)
  Phase 3: Run Suite 2 `FullStory-Bench` (DOC-NLI + Continuity + 32B ConStory-Checker + EQ-Bench + Blind Pairwise)
  Phase 4: Run Suite 2A `ClassicNLG-Bench` (BLEU-1..4, ROUGE-L, Distinct-1..3, Entropy-2, Sliding PPL + Spearman)
  Phase 5: Export unified side-by-side comparison table (`comparison_table.csv` & `unified_comparison_summary.json`)

Example:
    python3 -u scripts/eval_full_pipeline.py \
      --checkpoint-dir checkpoints/exp02_rlmr_orthogonal_32b_v2 \
      --num-prompts 5 \
      --output-dir eval_outputs/exp02_rlmr_v2_vs_base
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.coordinator import CoordinatorAgent
from src.pipeline import (
    VLLMWritingPolicy,
    build_writer_chat_messages,
    deduplicate_repeated_paragraphs,
    extract_cot_and_prose,
)
from src.planning import OutlineAgent, PlanningAgent
from src.rewards import count_words
from experiments.benchmark.evaluator import (
    compute_causal_coherence,
    compute_conflict_completeness,
    compute_premise_coverage,
    strip_length_constraints_from_premise,
)
from experiments.benchmarks.outline_bench.metrics_pacing import evaluate_concreteness_and_pacing


def resolve_checkpoint_dir(raw_path: Path) -> Path:
    """Resolve either an adapter folder directly or the highest `checkpoint-*` inside a run directory."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {raw_path}")
    if (raw_path / "adapter_config.json").exists():
        return raw_path
    ckpts = sorted(
        [d for d in raw_path.glob("checkpoint-*") if d.is_dir() and (d / "adapter_config.json").exists()],
        key=lambda d: int(d.name.split("-")[-1]),
    )
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(
        f"No adapter_config.json found directly in {raw_path} or inside any checkpoint-* subfolder."
    )


def load_done_ids(path: Path, num_chapters: int) -> set:
    if not path.exists():
        return set()
    valid_rows: List[Dict[str, Any]] = []
    done_ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if int(rec.get("actual_words", 0) or 0) > 0 and len(rec.get("chapters") or []) >= num_chapters:
            valid_rows.append(rec)
            done_ids.add(rec["premise_id"])
    if not valid_rows:
        path.unlink(missing_ok=True)
    else:
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in valid_rows) + "\n", encoding="utf-8")
    return done_ids


def run_phase1_plans(
    plans_path: Path,
    test_data_path: Path,
    planner_model_id: str,
    num_prompts: int,
    num_chapters: int,
    total_target_words: int,
) -> List[Dict[str, Any]]:
    planning_agent = PlanningAgent(llm_generate_fn=None)

    if plans_path.exists():
        shared_plans = json.loads(plans_path.read_text(encoding="utf-8"))
        repaired = False
        for item in shared_plans:
            if len(item.get("story_plan", {}).get("chapters") or []) < num_chapters:
                item["story_plan"] = planning_agent.partition_into_chapters(
                    event_graph=item["event_graph"],
                    total_target_words=int(item.get("target_words", total_target_words)),
                    num_chapters=num_chapters,
                    enable_micro_subevents=False,
                )
                repaired = True
        if repaired:
            plans_path.write_text(json.dumps(shared_plans, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[Phase 1 Auto-Repair] Partitioned {len(shared_plans)} plans into {num_chapters} chapters.", flush=True)
        if len(shared_plans) >= num_prompts:
            print(f"[Phase 1 Resume] Loaded {num_prompts} shared Condition D plans from {plans_path}", flush=True)
            return shared_plans[:num_prompts]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw_rows = [
        json.loads(line) for line in test_data_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ][:num_prompts]
    print(f"[Phase 1] Loading {planner_model_id} (HF 4-bit SDPA) for {len(raw_rows)} Condition D plans...", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    tok = AutoTokenizer.from_pretrained(planner_model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        planner_model_id,
        torch_dtype=compute_dtype,
        device_map="cuda:0" if torch.cuda.is_available() else "auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()

    @torch.inference_mode()
    def planner_generate_fn(system_prompt: str, user_prompt: str) -> str:
        prompt_str = tok.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tok(prompt_str, return_tensors="pt").to(model.device)
        in_len = enc["input_ids"].shape[1]
        out = model.generate(
            **enc,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )
        return tok.decode(out[0][in_len:], skip_special_tokens=True).strip()

    outline_agent = OutlineAgent(llm_generate_fn=planner_generate_fn)
    shared_plans: List[Dict[str, Any]] = []
    for idx, row in enumerate(raw_rows):
        pid = str(row.get("example_id") or f"test_{idx:03d}")
        clean_premise = strip_length_constraints_from_premise(str(row.get("inputs", "")))
        t0 = time.time()
        event_graph = outline_agent.create_event_graph(
            premise=clean_premise,
            genre="Creative Fiction",
            num_events=num_chapters,
        )
        blueprint = planning_agent.partition_into_chapters(
            event_graph=event_graph,
            total_target_words=total_target_words,
            num_chapters=num_chapters,
            enable_micro_subevents=False,
        )
        s_cov = compute_premise_coverage(clean_premise, blueprint)
        causal_info = compute_causal_coherence(blueprint, nli_scorer=None)
        s_comp = compute_conflict_completeness(blueprint)
        pace_info = evaluate_concreteness_and_pacing(blueprint)
        audit = {
            "s_cov": round(float(s_cov), 4),
            "s_causal": round(float(causal_info.get("S_causal", 0.0)), 4),
            "s_comp": round(float(s_comp), 4),
            "concreteness": round(float(pace_info.get("mean_concreteness", 0.0)), 4),
            "pacing_balance": round(float(pace_info.get("pacing_balance_score", 0.0)), 4),
        }
        shared_plans.append(
            {
                "premise_id": pid,
                "genre": "Creative Fiction",
                "premise": clean_premise,
                "target_words": total_target_words,
                "event_graph": event_graph,
                "story_plan": blueprint,
                "outline_audit": audit,
            }
        )
        print(
            f"  [{idx + 1}/{len(raw_rows)}] {pid} planned in {time.time() - t0:.1f}s | "
            f"S_cov={audit['s_cov']:.3f} S_causal={audit['s_causal']:.3f} S_comp={audit['s_comp']:.3f}",
            flush=True,
        )

    plans_path.write_text(json.dumps(shared_plans, indent=2, ensure_ascii=False), encoding="utf-8")
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return shared_plans


def run_phase2_generation(
    shared_plans: List[Dict[str, Any]],
    writer_base_model: str,
    checkpoint_dir: Path,
    arm_base: str,
    arm_ckpt: str,
    stories_base_path: Path,
    stories_ckpt_path: Path,
    transitions_path: Path,
    num_chapters: int,
    total_target_words: int,
    max_chapter_tokens: int,
) -> None:
    base_done = load_done_ids(stories_base_path, num_chapters)
    ckpt_done = load_done_ids(stories_ckpt_path, num_chapters)
    if len(base_done) >= len(shared_plans) and len(ckpt_done) >= len(shared_plans):
        print(f"[Phase 2 Resume] Both {arm_base} and {arm_ckpt} already generated all {len(shared_plans)} stories.", flush=True)
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = True
    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"[Phase 2] Loading {writer_base_model} in native {compute_dtype} + SDPA...", flush=True)
    tok = AutoTokenizer.from_pretrained(writer_base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        writer_base_model,
        torch_dtype=compute_dtype,
        device_map="cuda:0" if torch.cuda.is_available() else "auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    base_model.eval()

    @torch.inference_mode()
    def hf_chat_generate(model, messages, max_new_tokens: int, temperature: float, top_p: float) -> str:
        prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt_str, return_tensors="pt").to(model.device)
        in_len = enc["input_ids"].shape[1]
        do_sample = temperature > 0.05
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )
        return tok.decode(gen[0][in_len:], skip_special_tokens=True).strip()

    def _run_arm(arm_name: str, model, out_path: Path, is_peft: bool) -> None:
        done = load_done_ids(out_path, num_chapters)
        policy_helper = VLLMWritingPolicy(use_mock=True)

        def coord_llm_fn(sys_p: str, usr_p: str) -> str:
            msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
            if is_peft:
                with model.disable_adapter():
                    return hf_chat_generate(model, msgs, 640, 0.25, 0.9)
            return hf_chat_generate(model, msgs, 640, 0.25, 0.9)

        coordinator = CoordinatorAgent(llm_generate_fn=coord_llm_fn)

        for item in shared_plans:
            pid = item["premise_id"]
            if pid in done:
                print(f"  [{arm_name}] Skipping already completed {pid}", flush=True)
                continue
            t_story = time.time()
            blueprint = item["story_plan"]
            assert len(blueprint.get("chapters", [])) == num_chapters, f"Plan {pid} missing chapters!"
            events = item["event_graph"].get("events", [])
            all_chars = list(dict.fromkeys(ev.get("character", "Protagonist") for ev in events)) or ["Protagonist"]

            chapters_clean: List[str] = []
            chapters_raw: List[str] = []
            past_plans: List[Dict[str, Any]] = []
            prev_ctx: Optional[Dict[str, Any]] = None

            for ch_plan in blueprint["chapters"]:
                ch_idx = ch_plan["chapter_index"]
                target_w = int(ch_plan.get("target_words", total_target_words // num_chapters))
                ctx_state = coordinator.condense_context(
                    next_chapter_index=ch_idx,
                    past_chapters=chapters_clean,
                    past_plans=past_plans,
                    characters=all_chars,
                    previous_context=prev_ctx,
                    next_chapter_plan=ch_plan,
                )
                prev_ctx = ctx_state

                prompt = policy_helper.construct_prompt(ch_plan, ctx_state, enable_cot=True)
                chat_messages = build_writer_chat_messages(prompt, target_words=target_w, enable_cot=True)
                raw_out = hf_chat_generate(model, chat_messages, max_chapter_tokens, 0.75, 0.92)
                cot_txt, prose_txt = extract_cot_and_prose(raw_out)
                prose_dedup, removed = deduplicate_repeated_paragraphs(prose_txt, min_words=30, min_sentence_words=12)

                if removed > 0 and count_words(prose_dedup) < int(0.75 * target_w):
                    raw_retry = hf_chat_generate(
                        model, chat_messages, max_chapter_tokens, 0.85, 0.92
                    )
                    cot_r, prose_r = extract_cot_and_prose(raw_retry)
                    prose_r_dedup, removed_r = deduplicate_repeated_paragraphs(
                        prose_r, min_words=30, min_sentence_words=12
                    )
                    if count_words(prose_r_dedup) > count_words(prose_dedup):
                        raw_out, cot_txt, prose_dedup, removed = raw_retry, cot_r, prose_r_dedup, removed_r

                chapters_raw.append(raw_out)
                chapters_clean.append(prose_dedup)
                past_plans.append(ch_plan)

                with open(transitions_path, "a", encoding="utf-8") as tf:
                    tf.write(
                        json.dumps(
                            {
                                "arm": arm_name,
                                "premise_id": pid,
                                "chapter_index": ch_idx,
                                "context_summary": ctx_state.get("plot_summary", ""),
                                "prev_chapter_tail": ctx_state.get("recent_tail_excerpt", ""),
                                "is_fallback": False,
                                "dedup_removed": removed,
                                "think_words": len(cot_txt.split()),
                                "prose_words": count_words(prose_dedup),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                print(
                    f"    [{arm_name}] {pid} Ch {ch_idx}/{num_chapters}: "
                    f"prose={count_words(prose_dedup)}w (tgt={target_w}w), think={len(cot_txt.split())}w, dedup={removed}",
                    flush=True,
                )

            total_w = sum(count_words(c) for c in chapters_clean)
            rec = {
                "arm": arm_name,
                "premise_id": pid,
                "genre": item["genre"],
                "premise": item["premise"],
                "target_words": total_target_words,
                "actual_words": total_w,
                "chapters": chapters_clean,
                "chapters_raw_with_think": chapters_raw,
                "story_plan": blueprint,
                "generation_latency_s": round(time.time() - t_story, 2),
            }
            with open(out_path, "a", encoding="utf-8") as sf:
                sf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  [{arm_name}] Saved {pid}: {total_w}w in {rec['generation_latency_s']}s", flush=True)

    print(f"\n=== Phase 2 Arm 1: {arm_base} (Untrained Base 7B) ===", flush=True)
    _run_arm(arm_base, base_model, stories_base_path, is_peft=False)

    print(f"\n=== Phase 2 Arm 2: {arm_ckpt} (LoRA: {checkpoint_dir}) ===", flush=True)
    peft_model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
    peft_model.eval()
    _run_arm(arm_ckpt, peft_model, stories_ckpt_path, is_peft=True)

    del peft_model, base_model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_subprocess_checked(cmd: List[str], label: str) -> None:
    print(f"\n[{label}] Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"{label} exited with code {ret}")


def build_unified_summary(
    output_dir: Path,
    fullstory_out: Path,
    classic_out: Path,
    checkpoint_dir: Path,
    arm_base: str,
    arm_ckpt: str,
    num_prompts: int,
    total_target_words: int,
) -> None:
    def _read_csv_by_arm(csv_path: Path) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[row["arm"]] = dict(row)
        return out

    def _to_float(v: Any) -> Optional[float]:
        try:
            if v is None or str(v).strip() == "" or str(v).lower() == "nan":
                return None
            return float(v)
        except Exception:
            return None

    fs_by_arm = _read_csv_by_arm(fullstory_out / "fullstory_summary.csv")
    cl_by_arm = _read_csv_by_arm(classic_out / "classic_nlg_summary.csv")
    base_fs, ckpt_fs = fs_by_arm.get(arm_base, {}), fs_by_arm.get(arm_ckpt, {})
    base_cl, ckpt_cl = cl_by_arm.get(arm_base, {}), cl_by_arm.get(arm_ckpt, {})

    metric_specs = [
        ("FullStory", "Mean Actual Words (Target=8000)", "actual_words", "fs"),
        ("FullStory", "Length Compliance S_l (%) ↑", "length_score_sl", "fs"),
        ("FullStory", "DOC-NLI Plan Adherence (S_adh) ↑", "S_adh", "fs"),
        ("FullStory", "Inter-Chapter Continuity ↑", "transition_cosine", "fs"),
        ("FullStory", "Anti-Context Novelty ↑", "anti_context_novelty", "fs"),
        ("FullStory", "EQ-Bench Longform (0-100) ↑", "eqbench_score", "fs"),
        ("FullStory", "EQ Meaningful Elaboration ↑", "eq_meaningful_elaboration", "fs"),
        ("FullStory", "ConStory C_weighted ↓", "C_weighted", "fs"),
        ("FullStory", "ConStory Errors / 1k Words ↓", "constory_errors_per_1k", "fs"),
        ("FullStory", "Slop Score / 1k Words ↓", "slop_score_per_1k", "fs"),
        ("FullStory", "Intra-Chapter Redundancy ↓", "intra_chapter_redundancy", "fs"),
        ("FullStory", "Mean <think> Words", "think_words", "fs"),
        ("ClassicNLG", "BLEU-1 ↑", "BLEU-1", "cl"),
        ("ClassicNLG", "BLEU-4 ↑", "BLEU-4", "cl"),
        ("ClassicNLG", "ROUGE-L ↑", "ROUGE-L", "cl"),
        ("ClassicNLG", "Distinct-3 (%) ↑", "Distinct-3 (%)", "cl"),
        ("ClassicNLG", "Entropy-2 ↑", "Entropy-2", "cl"),
        ("ClassicNLG", "Sliding-Window Perplexity ↓", "Perplexity", "cl"),
    ]

    rows_tbl: List[Dict[str, Any]] = []
    print(f"\n================ {arm_base} vs {arm_ckpt} ({num_prompts} HELD-OUT PROMPTS) ================")
    print(f"{'Suite':<11} | {'Metric':<36} | {arm_base:>10} | {arm_ckpt:>12} | {'Delta':>10}")
    print("-" * 92)
    for suite, label, key, src in metric_specs:
        b_val = _to_float((base_fs if src == "fs" else base_cl).get(key))
        c_val = _to_float((ckpt_fs if src == "fs" else ckpt_cl).get(key))
        delta = (c_val - b_val) if (b_val is not None and c_val is not None) else None
        rows_tbl.append(
            {
                "Suite": suite,
                "Metric": label,
                arm_base: round(b_val, 4) if b_val is not None else None,
                arm_ckpt: round(c_val, 4) if c_val is not None else None,
                "Delta": round(delta, 4) if delta is not None else None,
            }
        )
        fmt = lambda x: "N/A" if x is None else f"{x:.4f}"
        print(f"{suite:<11} | {label:<36} | {fmt(b_val):>10} | {fmt(c_val):>12} | {fmt(delta):>10}")

    pw_path = fullstory_out / "pairwise_verdicts.json"
    pairwise = json.loads(pw_path.read_text(encoding="utf-8")) if pw_path.exists() else []
    if pairwise:
        w_base = sum(1 for v in pairwise if v.get("winner") == arm_base)
        w_ckpt = sum(1 for v in pairwise if v.get("winner") == arm_ckpt)
        ties = sum(1 for v in pairwise if v.get("winner") == "tie")
        print(f"\n[Blind Pairwise 32B Judge] {arm_base} wins: {w_base} | {arm_ckpt} wins: {w_ckpt} | Ties: {ties}")

    corr_path = classic_out / "classic_vs_fullstory_correlation.json"
    corr_data = json.loads(corr_path.read_text(encoding="utf-8")) if corr_path.exists() else {}

    unified = {
        "checkpoint": str(checkpoint_dir),
        "num_prompts": num_prompts,
        "target_words": total_target_words,
        "table": rows_tbl,
        "fullstory_by_arm": {arm_base: base_fs, arm_ckpt: ckpt_fs},
        "classic_nlg_by_arm": {arm_base: base_cl, arm_ckpt: ckpt_cl},
        "pairwise_verdicts": pairwise,
        "classic_vs_fullstory_correlation": corr_data,
    }
    (output_dir / "unified_comparison_summary.json").write_text(
        json.dumps(unified, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(output_dir / "comparison_table.csv", "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=["Suite", "Metric", arm_base, arm_ckpt, "Delta"])
        writer.writeheader()
        writer.writerows(rows_tbl)
    print(f"\n[Done] Unified comparison saved -> {output_dir / 'unified_comparison_summary.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-pipeline Base 7B vs GRPO Checkpoint evaluator (Suite 2 + 2A).")
    parser.add_argument("--checkpoint-dir", required=True, help="LoRA adapter dir or run dir containing checkpoint-*.")
    parser.add_argument("--output-dir", required=True, help="Directory to store stories, logs, and benchmark reports.")
    parser.add_argument("--test-data", default="data/tell-me-a-story-test.jsonl")
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--num-chapters", type=int, default=8)
    parser.add_argument("--target-words", type=int, default=8000)
    parser.add_argument("--max-chapter-tokens", type=int, default=2300)
    parser.add_argument("--planner-model", default="unsloth/Qwen2.5-32B-Instruct-bnb-4bit")
    parser.add_argument("--writer-base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--judge-model", default="unsloth/Qwen2.5-32B-Instruct-bnb-4bit")
    parser.add_argument("--ppl-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--arm-base", default="base_7b")
    parser.add_argument("--arm-ckpt", default=None, help="Label for checkpoint arm (defaults to checkpoint folder name).")
    args = parser.parse_args()

    ckpt_dir = resolve_checkpoint_dir(Path(args.checkpoint_dir))
    arm_ckpt = args.arm_ckpt or f"rlmr_{ckpt_dir.name.replace('-', '')}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plans_path = out_dir / "shared_macro_plans.json"
    stories_base_path = out_dir / f"stories_{args.arm_base}.jsonl"
    stories_ckpt_path = out_dir / f"stories_{arm_ckpt}.jsonl"
    transitions_path = out_dir / "coordinator_transitions.jsonl"
    fullstory_out = out_dir / "suite2_fullstory_bench"
    classic_out = out_dir / "suite2a_classic_nlg_bench"

    print(f"[Full Eval] Resolved Checkpoint : {ckpt_dir}", flush=True)
    print(f"[Full Eval] Arms                : {args.arm_base} vs {arm_ckpt}", flush=True)
    print(f"[Full Eval] Prompts / Target    : {args.num_prompts} x {args.target_words}w ({args.num_chapters} ch)", flush=True)

    shared_plans = run_phase1_plans(
        plans_path=plans_path,
        test_data_path=Path(args.test_data),
        planner_model_id=args.planner_model,
        num_prompts=args.num_prompts,
        num_chapters=args.num_chapters,
        total_target_words=args.target_words,
    )

    run_phase2_generation(
        shared_plans=shared_plans,
        writer_base_model=args.writer_base_model,
        checkpoint_dir=ckpt_dir,
        arm_base=args.arm_base,
        arm_ckpt=arm_ckpt,
        stories_base_path=stories_base_path,
        stories_ckpt_path=stories_ckpt_path,
        transitions_path=transitions_path,
        num_chapters=args.num_chapters,
        total_target_words=args.target_words,
        max_chapter_tokens=args.max_chapter_tokens,
    )

    cmd_fs = [
        sys.executable, "-u", str(REPO_ROOT / "scripts" / "benchmark_fullstory.py"),
        "--stories", str(stories_base_path), str(stories_ckpt_path),
        "--labels", args.arm_base, arm_ckpt,
        "--judge-backend", "local_hf",
        "--judge-model", args.judge_model,
        "--judge-max-len", "24576",
        "--pairwise-arms", args.arm_base, arm_ckpt,
        "--output-dir", str(fullstory_out),
    ]
    if transitions_path.exists():
        cmd_fs.extend(["--transitions", str(transitions_path)])
    run_subprocess_checked(cmd_fs, "Phase 3: FullStory-Bench")

    cmd_cnlg = [
        sys.executable, "-u", str(REPO_ROOT / "scripts" / "benchmark_classic_nlg.py"),
        "--stories", str(stories_base_path), str(stories_ckpt_path),
        "--labels", args.arm_base, arm_ckpt,
        "--reference-file", str(args.test_data),
        "--reference-fallback", "premise_and_plan",
        "--ppl-model", args.ppl_model,
        "--compare-with-fullstory", str(fullstory_out / "fullstory_raw.jsonl"),
        "--output-dir", str(classic_out),
    ]
    run_subprocess_checked(cmd_cnlg, "Phase 4: ClassicNLG-Bench")

    build_unified_summary(
        output_dir=out_dir,
        fullstory_out=fullstory_out,
        classic_out=classic_out,
        checkpoint_dir=ckpt_dir,
        arm_base=args.arm_base,
        arm_ckpt=arm_ckpt,
        num_prompts=args.num_prompts,
        total_target_words=args.target_words,
    )


if __name__ == "__main__":
    main()
