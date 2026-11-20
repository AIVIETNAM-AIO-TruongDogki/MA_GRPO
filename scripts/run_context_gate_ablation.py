#!/usr/bin/env python3
"""
Process-Isolated Runner for ContextGateValidator Ablation:
  - Outline + Coordinator + LLM Validator : Qwen/Qwen2.5-32B-Instruct-AWQ (vLLM float16)
  - Untrained Writing Agent               : Qwen/Qwen2.5-7B-Instruct (vLLM bfloat16/float16)

Why subprocess isolation per phase?
  1. Eliminates IPython/Colab CUDA fork RuntimeError (VLLM_WORKER_MULTIPROC_METHOD=spawn).
  2. Guarantees 100% OS-level GPU VRAM release between 32B-AWQ and 7B loads on A100 40GB.
  3. Saves checkpoint state (`ablation_state.json`) to Google Drive after every chapter.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Must be set BEFORE importing torch or vllm
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.coordinator import (
    COORDINATOR_TWO_TIER_PROMPT,
    VALIDATOR_MULTI_SIGNAL_PROMPT,
    ContextGateValidator,
    CoordinatorAgent,
    clean_json_response,
    detect_heuristic_negation_contradictions,
)
from src.dataset_generator import load_premises_pool
from src.pipeline import VLLMWritingPolicy, extract_cot_and_prose, resolve_optimal_dtype
from src.planning import OutlineAgent, PlanCriticAgent, PlanningAgent


def _load_vllm(model_id: str, max_model_len: int, gpu_mem_util: float):
    from vllm import LLM

    lower_m = model_id.lower()
    dtype = "float16" if ("awq" in lower_m or "gptq" in lower_m) else resolve_optimal_dtype()
    print(
        f"[vLLM] Loading {model_id} (dtype={dtype}, max_len={max_model_len}, gpu_mem={gpu_mem_util})...",
        flush=True,
    )
    eng = LLM(
        model=model_id,
        dtype=dtype,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    return eng, eng.get_tokenizer()


def _batch_chat(
    eng: Any,
    tok: Any,
    max_model_len: int,
    requests: List[Tuple[str, str]],
    max_tokens: int,
    temperature: float,
    seed: int = 42,
) -> List[str]:
    from vllm import SamplingParams

    if not requests:
        return []
    formatted: List[str] = []
    max_p_tok = 0
    for sys_p, usr_p in requests:
        txt = tok.apply_chat_template(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted.append(txt)
        max_p_tok = max(max_p_tok, len(tok.encode(txt, add_special_tokens=False)))

    safe_max = max(64, min(max_tokens, max_model_len - max_p_tok - 32))
    params = SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=safe_max,
        presence_penalty=0.1,
        seed=seed,
    )
    outputs = eng.generate(formatted, params, use_tqdm=False)
    return [o.outputs[0].text.strip() if o.outputs else "" for o in outputs]


def run_phase_outline(args: argparse.Namespace, output_dir: Path) -> None:
    plans_file = output_dir / "shared_macro_plans.json"
    state_file = output_dir / "ablation_state.json"

    if args.resume and plans_file.exists():
        shared_plans = json.loads(plans_file.read_text(encoding="utf-8"))
        print(f"[Outline] Reusing {len(shared_plans)} saved Macro Plans from {plans_file}", flush=True)
    else:
        premises, excluded = load_premises_pool(
            premises_files=[args.premises_file],
            include_canonical=False,
            exclude_files=[args.exclude_test_file] if args.exclude_test_file else [],
            num_premises=args.num_stories,
            target_words_choices=args.target_words_choices,
            num_chapters=args.num_chapters,
        )
        print(
            f"[Outline] Selected {len(premises)} MoPS premises (excluded {excluded} test overlaps).",
            flush=True,
        )

        if args.backend == "vllm":
            eng32, tok32 = _load_vllm(args.planner_model, args.coord_max_len, args.planner_gpu_mem)
            outline_fn = lambda s, u: _batch_chat(
                eng32, tok32, args.coord_max_len, [(s, u)], max_tokens=2048, temperature=0.7, seed=args.seed
            )[0]
        else:
            outline_fn = None

        outline_agent = OutlineAgent(llm_generate_fn=outline_fn)
        planning_agent = PlanningAgent(
            sub_event_agent=None, critic=PlanCriticAgent(), enable_micro_subevents=False
        )

        shared_plans = []
        for idx, item in enumerate(premises, 1):
            t0 = time.time()
            print(
                f"  -> [{idx}/{len(premises)}] Outline + Planning: {item['id']} ({item['genre']}, {item['target_words']}w)",
                flush=True,
            )
            eg = outline_agent.create_event_graph(
                item["premise"], genre=item["genre"], num_events=args.num_chapters
            )
            sp = planning_agent.partition_into_chapters(
                event_graph=eg,
                total_target_words=item["target_words"],
                num_chapters=args.num_chapters,
                enable_micro_subevents=False,
            )
            shared_plans.append({"item": item, "event_graph": eg, "story_plan": sp})
            print(
                f"     Done in {time.time() - t0:.1f}s | planning_tokens={eg.get('planning_tokens')}",
                flush=True,
            )

        plans_file.write_text(json.dumps(shared_plans, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Outline] Saved shared Macro Plans -> {plans_file}", flush=True)

    # Initialize state file if missing (or not resuming)
    if not (args.resume and state_file.exists()):
        helper_coord = CoordinatorAgent(llm_generate_fn=None)
        states: List[Dict[str, Any]] = []
        for arm in args.arms:
            for bundle in shared_plans:
                item = bundle["item"]
                sp = bundle["story_plan"]
                init_chars = sp["chapters"][0].get("characters_active", ["Protagonist"])
                init_ctx = helper_coord.condense_context(1, [], [], init_chars)
                states.append(
                    {
                        "arm": arm,
                        "item": item,
                        "story_plan": sp,
                        "ctx": init_ctx,
                        "past_chapters": [],
                        "past_raw_outputs": [],
                        "past_plans": [],
                        "transitions": [],
                        "fallback_count": 0,
                        "coord_latency_s": 0.0,
                        "writer_latency_s": 0.0,
                    }
                )
        state_file.write_text(json.dumps(states, ensure_ascii=False), encoding="utf-8")
        print(f"[Outline] Initialized {len(states)} arm x story states -> {state_file}", flush=True)


def run_phase_write(args: argparse.Namespace, output_dir: Path, ch_idx: int) -> None:
    state_file = output_dir / "ablation_state.json"
    states = json.loads(state_file.read_text(encoding="utf-8"))

    # Skip if already written in a previous resumed run
    if all(len(st["past_chapters"]) >= ch_idx for st in states):
        print(f"[Write Ch {ch_idx}] Already completed for all {len(states)} runs. Skipping.", flush=True)
        return

    prompt_builder = VLLMWritingPolicy(use_mock=True)
    t0 = time.time()

    if args.backend == "vllm":
        eng7, tok7 = _load_vllm(args.writer_model, args.writer_max_len, args.writer_gpu_mem)
        reqs: List[Tuple[str, str]] = []
        todo_indices: List[int] = []
        for i, st in enumerate(states):
            if len(st["past_chapters"]) >= ch_idx:
                continue
            ch_plan = st["story_plan"]["chapters"][ch_idx - 1]
            w_prompt = prompt_builder.construct_prompt(ch_plan, st["ctx"], enable_cot=True)
            reqs.append(("You are an elite literary novelist.", w_prompt))
            todo_indices.append(i)

        raw_outputs = _batch_chat(
            eng7,
            tok7,
            args.writer_max_len,
            reqs,
            max_tokens=args.writer_max_len,
            temperature=0.7,
            seed=args.seed + ch_idx,
        )
    else:
        todo_indices = [i for i, st in enumerate(states) if len(st["past_chapters"]) < ch_idx]
        raw_outputs = [
            prompt_builder._mock_prose(
                states[i]["story_plan"]["chapters"][ch_idx - 1], states[i]["ctx"]
            )
            for i in todo_indices
        ]

    elapsed = time.time() - t0
    for i, raw_out in zip(todo_indices, raw_outputs):
        st = states[i]
        ch_plan = st["story_plan"]["chapters"][ch_idx - 1]
        _, prose = extract_cot_and_prose(raw_out)
        prose = prose if prose else raw_out
        st["past_raw_outputs"].append(raw_out)
        st["past_chapters"].append(prose)
        st["past_plans"].append(ch_plan)
        st["writer_latency_s"] = round(st["writer_latency_s"] + elapsed / max(1, len(todo_indices)), 2)

    state_file.write_text(json.dumps(states, ensure_ascii=False), encoding="utf-8")
    avg_w = sum(len(st["past_chapters"][ch_idx - 1].split()) for st in states) / max(1, len(states))
    print(
        f"[Write Ch {ch_idx}/{args.num_chapters}] Done {len(todo_indices)} runs in {elapsed:.1f}s "
        f"(avg {avg_w:.0f} words/ch)",
        flush=True,
    )


def run_phase_coord(args: argparse.Namespace, output_dir: Path, ch_idx: int) -> None:
    state_file = output_dir / "ablation_state.json"
    states = json.loads(state_file.read_text(encoding="utf-8"))

    # Transition index for ch_idx -> ch_idx + 1 is count == ch_idx
    if all(len(st["transitions"]) >= ch_idx for st in states):
        print(f"[Coord Ch {ch_idx}->{ch_idx+1}] Already completed. Skipping.", flush=True)
        return

    helper_coord = CoordinatorAgent(llm_generate_fn=None)
    heuristic_validator = ContextGateValidator(llm_generate_fn=None, threshold=args.validator_threshold)
    todo_indices = [i for i, st in enumerate(states) if len(st["transitions"]) < ch_idx]
    t0 = time.time()

    coord_reqs: List[Tuple[str, str]] = []
    tails: Dict[int, str] = {}
    for i in todo_indices:
        st = states[i]
        prev_text = st["past_chapters"][ch_idx - 1]
        tails[i] = " ".join(prev_text.strip().split()[-180:])
        last_p = st["past_plans"][ch_idx - 1]
        u_prompt = (
            f"Total chapters completed: {len(st['past_chapters'])}.\n"
            f"Upcoming chapter index: {ch_idx + 1}.\n\n"
            f"[LAST CHAPTER PROSE (TAIL)]:\n{prev_text[-2500:]}\n\n"
            f"[LAST CHAPTER PLAN GOAL]:\n{last_p.get('chapter_goal', '')}\n"
            f"[LAST CHAPTER EXIT STATE]:\n{last_p.get('exit_state', '')}\n\n"
            "Synthesize Two-Tier Context State matching the JSON schema.\n"
        )
        coord_reqs.append((COORDINATOR_TWO_TIER_PROMPT, u_prompt))

    eng32 = tok32 = None
    if args.backend == "vllm":
        eng32, tok32 = _load_vllm(args.planner_model, args.coord_max_len, args.planner_gpu_mem)
        coord_raws = _batch_chat(
            eng32,
            tok32,
            args.coord_max_len,
            coord_reqs,
            max_tokens=768,
            temperature=0.3,
            seed=args.seed + 100 + ch_idx,
        )
    else:
        coord_raws = [None] * len(todo_indices)

    candidates: Dict[int, Dict[str, Any]] = {}
    json_fb: Dict[int, bool] = {}
    for idx_pos, i in enumerate(todo_indices):
        st = states[i]
        tail = tails[i]
        next_p = st["story_plan"]["chapters"][ch_idx]
        chars = next_p.get("characters_active", ["Protagonist"])
        raw_c = coord_raws[idx_pos]
        cand = None
        if raw_c is not None:
            try:
                data = json.loads(clean_json_response(raw_c))
                long_m = data.get("long_term_memory", {})
                short_m = data.get("short_term_memory", {})
                if not isinstance(long_m, dict) or not isinstance(short_m, dict):
                    raise ValueError("Invalid JSON shape")
                short_m["recent_chapter_index"] = len(st["past_chapters"])
                short_m["recent_tail_excerpt"] = tail
                plot_sum = str(short_m.get("recent_chapter_summary", "")).strip()
                if not plot_sum:
                    raise ValueError("Empty summary")
                c_map = long_m.get("character_states", {})
                c_list = (
                    [f"{k}: {v}" for k, v in c_map.items()]
                    if isinstance(c_map, dict) and c_map
                    else [f"{c}: Active" for c in chars]
                )
                cand = {
                    "chapter_index": ch_idx + 1,
                    "completed_chapters": len(st["past_chapters"]),
                    "long_term_memory": long_m,
                    "short_term_memory": short_m,
                    "plot_summary": plot_sum,
                    "character_statuses": c_list,
                    "pending_threads": long_m.get("unresolved_plot_threads", []),
                    "recent_tail_excerpt": tail,
                }
                json_fb[i] = False
            except Exception:
                cand = None
        if cand is None:
            cand = helper_coord._structured_state_extraction(
                ch_idx + 1, st["past_chapters"], st["past_plans"], chars, tail, st["ctx"], next_p
            )
            json_fb[i] = raw_c is not None
        candidates[i] = cand

    # Batch LLM Validator calls (same 32B-AWQ engine, temp=0.0)
    llm_val_indices = [i for i in todo_indices if states[i]["arm"] == "with_validator_llm"]
    llm_val_raw_map: Dict[int, Optional[str]] = {}
    if llm_val_indices and args.backend == "vllm":
        val_reqs: List[Tuple[str, str]] = []
        for i in llm_val_indices:
            st = states[i]
            cand = candidates[i]
            prev_text = st["past_chapters"][ch_idx - 1]
            prev_goal = st["past_plans"][ch_idx - 1].get("chapter_goal", "")
            next_goal = st["story_plan"]["chapters"][ch_idx].get("chapter_goal", "")
            lm = cand.get("long_term_memory", {})
            v_user = (
                f"[PRECEDING CHAPTER PROSE EXCERPTS (Source of Truth)]:\n"
                f"--- BEGINNING OF CHAPTER ---\n{prev_text[:1200]}\n...\n"
                f"--- CONCLUSION OF CHAPTER ---\n{prev_text[-1800:]}\n\n"
                f"[PRECEDING CHAPTER GOAL]: {prev_goal}\n"
                f"[UPCOMING CHAPTER GOAL]: {next_goal}\n\n"
                f"[PROPOSED CANDIDATE CONTEXT]:\n"
                f"Short-Term Recent Summary: {cand.get('plot_summary', '')}\n"
                f"Character States: {lm.get('character_states', cand.get('character_statuses', []))}\n"
                f"Unresolved Threads: {lm.get('unresolved_plot_threads', cand.get('pending_threads', []))}\n"
                f"Major Past Events: {lm.get('major_events', [])}\n\n"
                "Audit faithfulness (0.0-1.0) and completeness (0.0-1.0).\n"
            )
            val_reqs.append((VALIDATOR_MULTI_SIGNAL_PROMPT, v_user))
        val_outs = _batch_chat(
            eng32,
            tok32,
            args.coord_max_len,
            val_reqs,
            max_tokens=512,
            temperature=0.0,
            seed=args.seed + 200 + ch_idx,
        )
        for i, v_raw in zip(llm_val_indices, val_outs):
            llm_val_raw_map[i] = v_raw

    elapsed = time.time() - t0
    for i in todo_indices:
        st = states[i]
        arm = st["arm"]
        cand = candidates[i]
        tail = tails[i]
        prev_text = st["past_chapters"][ch_idx - 1]
        next_p = st["story_plan"]["chapters"][ch_idx]
        chars = next_p.get("characters_active", ["Protagonist"])

        if arm == "no_validator":
            final_ctx = cand
            is_fb = json_fb[i]
            val_meta = {
                "bypassed": True,
                "is_valid": True,
                "is_fallback": is_fb,
                "composite_score": None,
                "issues": [],
            }
        elif arm == "with_validator_heuristic":
            val_meta = heuristic_validator.validate_summary(
                summary=cand,
                prev_chapter_text=prev_text,
                prev_chapter_plan=st["past_plans"][ch_idx - 1],
                previous_context=st["ctx"],
                next_chapter_plan=next_p,
            )
            val_meta["bypassed"] = False
            if val_meta.get("is_valid", False):
                final_ctx = cand
                is_fb = json_fb[i]
            else:
                final_ctx = helper_coord._structured_state_extraction(
                    ch_idx + 1, st["past_chapters"], st["past_plans"], chars, tail, st["ctx"], next_p
                )
                is_fb = True
        elif arm == "with_validator_llm":
            neg_clashes = detect_heuristic_negation_contradictions(cand.get("plot_summary", ""), prev_text)
            issues = list(neg_clashes)
            v_raw = llm_val_raw_map.get(i)
            if v_raw is not None:
                try:
                    vdata = json.loads(clean_json_response(v_raw))
                    faith = float(vdata.get("faithfulness_score", 0.85))
                    comp = float(vdata.get("completeness_score", 0.85))
                    llm_contra = vdata.get("negation_contradictions", [])
                    feedback = str(vdata.get("feedback", "LLM 32B validation processed."))
                    if llm_contra:
                        issues.extend(llm_contra)
                    if neg_clashes:
                        faith = max(
                            0.0,
                            faith - (0.25 * len(set(neg_clashes) | set(llm_contra)) if llm_contra else 0.05 * len(neg_clashes)),
                        )
                except Exception:
                    faith, comp = heuristic_validator._heuristic_evaluation(cand, prev_text, next_p)
                    feedback = "Fallback heuristic scoring applied."
            else:
                faith, comp = heuristic_validator._heuristic_evaluation(cand, prev_text, next_p)
                feedback = "Mock LLM validator heuristic."
            comp_score = round(0.5 * faith + 0.5 * comp, 3)
            is_valid = comp_score >= args.validator_threshold
            val_meta = {
                "bypassed": False,
                "faithfulness_score": round(faith, 3),
                "completeness_score": round(comp, 3),
                "composite_score": comp_score,
                "is_valid": is_valid,
                "is_fallback": not is_valid,
                "feedback": feedback,
                "issues": issues,
            }
            if is_valid:
                final_ctx = cand
                is_fb = json_fb[i]
            else:
                final_ctx = helper_coord._structured_state_extraction(
                    ch_idx + 1, st["past_chapters"], st["past_plans"], chars, tail, st["ctx"], next_p
                )
                is_fb = True
        else:
            raise ValueError(f"Unknown arm: {arm}")

        final_ctx["validation_metadata"] = val_meta
        if is_fb:
            st["fallback_count"] += 1
        st["coord_latency_s"] = round(st["coord_latency_s"] + elapsed / max(1, len(todo_indices)), 2)
        st["transitions"].append(
            {
                "arm": arm,
                "premise_id": st["item"]["id"],
                "chapter_index": ch_idx + 1,
                "candidate_summary": cand.get("plot_summary", ""),
                "final_summary": final_ctx.get("plot_summary", ""),
                "prev_chapter_tail": tail,
                "context_state": final_ctx,
                "validation_metadata": val_meta,
                "is_fallback": is_fb,
            }
        )
        st["ctx"] = final_ctx

    state_file.write_text(json.dumps(states, ensure_ascii=False), encoding="utf-8")
    fb_str = ", ".join(
        f"{arm}={sum(1 for s in states if s['arm'] == arm and s['transitions'][ch_idx - 1]['is_fallback'])}"
        for arm in args.arms
    )
    print(
        f"[Coord Ch {ch_idx}->{ch_idx+1}] Done in {elapsed:.1f}s | fallbacks: {fb_str}",
        flush=True,
    )


def run_phase_export(args: argparse.Namespace, output_dir: Path) -> None:
    state_file = output_dir / "ablation_state.json"
    states = json.loads(state_file.read_text(encoding="utf-8"))

    stories_jsonl = output_dir / "stories_7b_outputs.jsonl"
    transitions_jsonl = output_dir / "coordinator_transitions.jsonl"
    stories_dir = output_dir / "stories"
    stories_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    with open(stories_jsonl, "w", encoding="utf-8") as fs, open(transitions_jsonl, "w", encoding="utf-8") as ft:
        for st in states:
            item = st["item"]
            sp = st["story_plan"]
            full_story = "\n\n".join(st["past_chapters"])
            actual_words = len(full_story.split())
            rec = {
                "arm": st["arm"],
                "planner_model": args.planner_model,
                "coordinator_model": args.planner_model,
                "validator_model": (
                    args.planner_model
                    if st["arm"] == "with_validator_llm"
                    else ("heuristic" if st["arm"] == "with_validator_heuristic" else None)
                ),
                "writer_model": args.writer_model,
                "premise_id": item["id"],
                "genre": item["genre"],
                "premise": item["premise"],
                "target_words": item["target_words"],
                "num_chapters": len(st["past_chapters"]),
                "actual_words": actual_words,
                "chapter_word_counts": [len(c.split()) for c in st["past_chapters"]],
                "fallback_count": st["fallback_count"],
                "n_transitions": len(st["transitions"]),
                "coord_latency_s": round(st["coord_latency_s"], 2),
                "writer_latency_s": round(st["writer_latency_s"], 2),
                "chapters": st["past_chapters"],
                "chapters_raw_with_think": st["past_raw_outputs"],
                "full_story": full_story,
                "story_plan": sp,
            }
            fs.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for tr in st["transitions"]:
                ft.write(json.dumps(tr, ensure_ascii=False) + "\n")

            arm_dir = stories_dir / st["arm"]
            arm_dir.mkdir(parents=True, exist_ok=True)
            md_parts = [
                f"# [{st['arm']}] Story `{item['id']}` ({item['genre']})",
                f"- **Outline + Coordinator:** `{args.planner_model}` | **Writer:** `{args.writer_model}` (untrained)",
                f"- **Premise:** {item['premise']}",
                f"- **Target Words:** {item['target_words']} | **Actual Words (7B):** {actual_words} | **Validator Fallbacks:** {st['fallback_count']}/{len(st['transitions'])}\n",
            ]
            for ch_p, ch_txt in zip(sp["chapters"], st["past_chapters"]):
                md_parts.append(f"## {ch_p['chapter_title']} ({len(ch_txt.split())} words)\n")
                md_parts.append(ch_txt + "\n")
            (arm_dir / f"{item['id']}.md").write_text("\n".join(md_parts), encoding="utf-8")

            summary_rows.append(
                {
                    "arm": st["arm"],
                    "premise_id": item["id"][:14] + "...",
                    "genre": item["genre"],
                    "target_w": item["target_words"],
                    "actual_w_7b": actual_words,
                    "fallbacks": f"{st['fallback_count']}/{len(st['transitions'])}",
                    "writer_s": rec["writer_latency_s"],
                    "coord_s": rec["coord_latency_s"],
                }
            )

    summary_file = output_dir / "ablation_summary.json"
    summary_file.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Export] Saved {len(summary_rows)} full stories -> {stories_jsonl}", flush=True)
    print(f"[Export] Saved transitions -> {transitions_jsonl}", flush=True)
    print(f"[Export] Saved readable .md files -> {stories_dir}", flush=True)


def _spawn_step(base_argv: List[str], phase: str, chapter: Optional[int] = None) -> None:
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--phase", phase]
    if chapter is not None:
        cmd.extend(["--chapter", str(chapter)])
    cmd.extend(base_argv)
    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"Subprocess phase='{phase}' chapter={chapter} failed with exit code {res.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process-isolated ContextGateValidator Ablation Runner.")
    parser.add_argument("--phase", choices=["all", "outline", "write", "coord", "export"], default="all")
    parser.add_argument("--chapter", type=int, default=1)
    parser.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    parser.add_argument("--planner-model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    parser.add_argument("--writer-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--planner-gpu-mem", type=float, default=0.85)
    parser.add_argument("--writer-gpu-mem", type=float, default=0.80)
    parser.add_argument("--coord-max-len", type=int, default=12288)
    parser.add_argument("--writer-max-len", type=int, default=16384)
    parser.add_argument("--premises-file", default=str(PROJECT_ROOT / "data" / "mops_premises.jsonl"))
    parser.add_argument("--exclude-test-file", default=str(PROJECT_ROOT / "data" / "tell-me-a-story-test.jsonl"))
    parser.add_argument("--num-stories", type=int, default=3)
    parser.add_argument("--num-chapters", type=int, default=8)
    parser.add_argument("--target-words-choices", nargs="+", type=int, default=[8000, 9000, 10000])
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["no_validator", "with_validator_heuristic", "with_validator_llm"],
    )
    parser.add_argument("--validator-threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "ablation_context_gate_outputs"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "all":
        t_all = time.time()
        shared_flags = [
            "--backend", args.backend,
            "--planner-model", args.planner_model,
            "--writer-model", args.writer_model,
            "--planner-gpu-mem", str(args.planner_gpu_mem),
            "--writer-gpu-mem", str(args.writer_gpu_mem),
            "--coord-max-len", str(args.coord_max_len),
            "--writer-max-len", str(args.writer_max_len),
            "--premises-file", str(args.premises_file),
            "--exclude-test-file", str(args.exclude_test_file),
            "--num-stories", str(args.num_stories),
            "--num-chapters", str(args.num_chapters),
            "--target-words-choices", *[str(x) for x in args.target_words_choices],
            "--arms", *args.arms,
            "--validator-threshold", str(args.validator_threshold),
            "--seed", str(args.seed),
            "--output-dir", str(output_dir),
        ]
        if args.resume:
            shared_flags.append("--resume")

        print("=== PHASE 1: OUTLINE & MACRO PLANNING (32B-AWQ) ===", flush=True)
        _spawn_step(shared_flags, phase="outline")

        for ch in range(1, args.num_chapters + 1):
            print(f"\n=== CHAPTER {ch}/{args.num_chapters} — 7B WRITER ===", flush=True)
            _spawn_step(shared_flags, phase="write", chapter=ch)
            if ch < args.num_chapters:
                print(f"=== CHAPTER {ch}->{ch+1} — 32B COORDINATOR + VALIDATOR ===", flush=True)
                _spawn_step(shared_flags, phase="coord", chapter=ch)

        print("\n=== FINAL EXPORT ===", flush=True)
        _spawn_step(shared_flags, phase="export")
        print(f"\n✅ ALL PHASES COMPLETE in {(time.time() - t_all) / 60.0:.1f} min!", flush=True)
    elif args.phase == "outline":
        run_phase_outline(args, output_dir)
    elif args.phase == "write":
        run_phase_write(args, output_dir, args.chapter)
    elif args.phase == "coord":
        run_phase_coord(args, output_dir, args.chapter)
    elif args.phase == "export":
        run_phase_export(args, output_dir)


if __name__ == "__main__":
    main()
