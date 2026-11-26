"""Fast Base-Model Length & Format Diagnostic for MA-GRPO (Pre-Training Smoke Test).

Tests whether Qwen2.5-7B-Instruct (untrained base) can naturally hit the 1,000-word
chapter target ([850, 1150] words) and pass all 3 RLMR objective constraints
(`V_length`, `V_format`, `V_no_loop`) across 4 prompt/template variants BEFORE
launching a multi-hour GRPO training run:

  A_v2_raw_no_chat          : Raw plain-string prompt without apply_chat_template (reproduces v2 train.py bug)
  B_v2_chat_baseline        : Existing v2 prompt wrapped with apply_chat_template (reproduces v2 eval baseline)
  C_chat_budget_short_think : Chat template + explicit per-scene paragraph budgets (3+5+4=12 paras ~1,000w)
                              + strict bullet-only <think> (<120w, forbids prose inside <think>)
  D_chat_budget_no_think    : Chat template + same per-scene paragraph budgets (~1,000w), NO <think> block

Example (on 48GB Blackwell server — uses cached HF weights, finishes in ~2 minutes):
    HF_HOME=/user-data/.hf_cache .venv/bin/python -u scripts/test_base_length.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rewards.base import EvaluationInput
from rewards.evaluators.constraint_evaluator import verify_objective_constraints


SYSTEM_NOVELIST_SHORT_THINK = (
    "You are an acclaimed literary novelist writing full-length chapters of approximately 1,000 words. "
    "You NEVER summarize or rush scenes; you fully dramatize every beat with concrete sensory detail, "
    "physical blocking, multi-turn character dialogue, and psychological depth so the chapter genuinely "
    "reaches 950-1,050 words of narrative prose (about 11-13 rich paragraphs of 80-95 words each). "
    "You strictly obey formatting rules: your <think> block contains ONLY 4-6 short bullet points "
    "(under 100 words total) and NEVER contains narrative prose or dialogue."
)

SYSTEM_NOVELIST_NO_THINK = (
    "You are an acclaimed literary novelist writing full-length chapters of approximately 1,000 words. "
    "You NEVER summarize or rush scenes; you fully dramatize every beat with concrete sensory detail, "
    "physical blocking, multi-turn character dialogue, and psychological depth so the chapter genuinely "
    "reaches 950-1,050 words of narrative prose (about 11-13 rich paragraphs of 80-95 words each). "
    "Output ONLY clean narrative prose with zero headings, scene labels, <think> tags, or author notes."
)


def _extract_core_body(v2_prompt: str) -> str:
    """Strip old CRITICAL WRITING INSTRUCTIONS tail from an existing Condition D prompt."""
    idx = v2_prompt.find("CRITICAL WRITING INSTRUCTIONS:")
    return v2_prompt[:idx].rstrip() if idx != -1 else v2_prompt.rstrip()


def _annotate_scene_paragraphs(body: str, target_words: int = 1000) -> str:
    """Annotate each Scene line's word budget with explicit paragraph counts (~85w/paragraph)."""
    def _repl(match: re.Match) -> str:
        prefix = match.group(1)
        words = int(match.group(2))
        paras = max(2, int(round(words / 85.0)))
        return f"{prefix}(~{words} words — write {paras} full, detailed paragraphs)"

    annotated = re.sub(
        r"(Scene\s+\d+\s*)\(~(\d+)\s*words\)",
        _repl,
        body,
        flags=re.IGNORECASE,
    )
    return annotated


def build_variant_payload(
    row: Dict[str, Any],
    variant: str,
) -> Tuple[Optional[List[Dict[str, str]]], Optional[str]]:
    """Build either chat messages (for apply_chat_template) or raw prompt string for a variant."""
    raw_prompt = str(row.get("prompt") or row.get("full_training_prompt") or "").strip()
    target_words = int(row.get("target_words") or 1000)
    l_low = int(round(target_words * 0.85))
    l_high = int(round(target_words * 1.15))
    total_paras = max(8, int(round(target_words / 85.0)))

    if variant == "A_v2_raw_no_chat":
        return None, raw_prompt

    if variant == "B_v2_chat_baseline":
        return [{"role": "user", "content": raw_prompt}], None

    core_body = _annotate_scene_paragraphs(_extract_core_body(raw_prompt), target_words=target_words)

    if variant == "C_chat_budget_short_think":
        user_msg = (
            f"{core_body}\n\n"
            f"STRICT LENGTH & FORMAT REQUIREMENTS:\n"
            f"1. Brief <think> Plan (MAX 80-100 words, bullet points ONLY):\n"
            f"   - Start with <think> and write ONLY 4-6 concise bullet points mapping paragraph counts and transitions.\n"
            f"   - DO NOT write any story sentences, descriptions, or dialogue inside <think>.\n"
            f"   - Close </think> immediately after the bullets.\n"
            f"2. Full Chapter Prose ({l_low}-{l_high} words, target ~{target_words} words):\n"
            f"   - Immediately after </think>, write approximately {total_paras} rich paragraphs (~80-95 words each).\n"
            f"   - Fully dramatize every scene with real-time action, sensory atmosphere, internal tension, and extended back-and-forth dialogue.\n"
            f"   - NEVER summarize events in a few sentences, NEVER skip ahead, and NEVER stop early around 400-600 words.\n"
            f"   - Do NOT include chapter/scene labels, markdown headings, or meta-commentary after </think>.\n"
            f"   - End cleanly on a complete final sentence.\n"
            f"Output <think>...</think> followed immediately by the complete ~{target_words}-word chapter:"
        )
        return [
            {"role": "system", "content": SYSTEM_NOVELIST_SHORT_THINK},
            {"role": "user", "content": user_msg},
        ], None

    if variant == "D_chat_budget_no_think":
        user_msg = (
            f"{core_body}\n\n"
            f"STRICT LENGTH & FORMAT REQUIREMENTS:\n"
            f"- Target Prose Length: strictly between {l_low} and {l_high} words (aim right around {target_words} words).\n"
            f"- Structure: write approximately {total_paras} substantial paragraphs (~80-95 words each) following the scene budgets above.\n"
            f"- Dramatization: expand every scene in real time with concrete sensory texture, physical gestures, internal conflict, and multi-turn dialogue (at least 6-10 lines of dialogue across the chapter).\n"
            f"- Anti-Brevity Rule: DO NOT summarize events, DO NOT wrap up after 400-600 words, and DO NOT repeat earlier paragraphs.\n"
            f"- Clean Prose Only: DO NOT output <think> tags, scene labels, section titles, or author notes. End cleanly on a complete final sentence.\n"
            f"Begin the complete ~{target_words}-word narrative prose immediately:"
        )
        return [
            {"role": "system", "content": SYSTEM_NOVELIST_NO_THINK},
            {"role": "user", "content": user_msg},
        ], None

    raise ValueError(f"Unknown variant: {variant}")


def select_diverse_samples(dataset_path: Path, num_samples: int) -> List[Dict[str, Any]]:
    """Pick diverse chapter indices (Ch1, Ch3, Ch5, Ch7...) across distinct premises."""
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    desired_chapters = [1, 3, 5, 7, 2, 4, 6, 8]
    selected: List[Dict[str, Any]] = []
    used_premises = set()

    for ch in desired_chapters:
        if len(selected) >= num_samples:
            break
        for r in rows:
            pid = str(r.get("premise_id", ""))
            if int(r.get("chapter_index", 0)) == ch and pid not in used_premises:
                selected.append(r)
                used_premises.add(pid)
                break

    for r in rows:
        if len(selected) >= num_samples:
            break
        if r not in selected:
            selected.append(r)
    return selected[:num_samples]


def evaluate_generated_output(
    row: Dict[str, Any],
    raw_text: str,
    gen_tokens: int,
    max_new_tokens: int,
    variant: str,
) -> Dict[str, Any]:
    """Run the exact Phase-2 RLMR Python verifier on a generated chapter."""
    target_words = int(row.get("target_words") or 1000)
    ev_in = EvaluationInput(
        generated_chapter=raw_text,
        target_words=target_words,
        chapter_title=str(row.get("chapter_title", "")),
        chapter_goal=str(row.get("chapter_goal", "")),
        context_summary=str(row.get("context_summary", "")),
        prev_chapter_tail=str(row.get("prev_chapter_tail", "")),
        chapter_index=int(row.get("chapter_index", 1)),
        premise_id=str(row.get("premise_id", "")),
    )
    v = verify_objective_constraints(ev_in, min_len_ratio=0.85, max_len_ratio=1.15, max_think_words=400)
    clipped = gen_tokens >= max_new_tokens
    return {
        "variant": variant,
        "sample_id": row.get("sample_id", f"{row.get('premise_id')}_ch{row.get('chapter_index')}"),
        "chapter_index": int(row.get("chapter_index", 1)),
        "target_words": target_words,
        "gen_tokens": int(gen_tokens),
        "clipped": bool(clipped),
        "prose_words": int(v["prose_words"]),
        "think_words": int(v["think_words"]),
        "v_length": bool(v["v_length"]),
        "v_format": bool(v["v_format"]),
        "v_no_loop": bool(v["v_no_loop"]),
        "is_compliant": bool(v["is_compliant"]),
        "length_compliance": float(v["length_compliance"]),
        "failed_constraints": list(v["failed_constraints"]),
        "raw_preview_head": raw_text[:240].replace("\n", " "),
        "raw_preview_tail": raw_text[-200:].replace("\n", " "),
    }


def summarize_variant(variant: str, records: List[Dict[str, Any]], elapsed_s: float) -> Dict[str, Any]:
    n = max(1, len(records))
    pws = [r["prose_words"] for r in records]
    tws = [r["think_words"] for r in records]
    return {
        "variant": variant,
        "n": len(records),
        "elapsed_s": round(elapsed_s, 1),
        "prose_min": min(pws),
        "prose_median": int(round(statistics.median(pws))),
        "prose_mean": int(round(statistics.mean(pws))),
        "prose_max": max(pws),
        "think_mean": int(round(statistics.mean(tws))),
        "v_length_pct": round(sum(r["v_length"] for r in records) / n * 100.0, 1),
        "v_format_pct": round(sum(r["v_format"] for r in records) / n * 100.0, 1),
        "v_no_loop_pct": round(sum(r["v_no_loop"] for r in records) / n * 100.0, 1),
        "compliant_pct": round(sum(r["is_compliant"] for r in records) / n * 100.0, 1),
        "clipped_pct": round(sum(r["clipped"] for r in records) / n * 100.0, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-training Base 7B 1000-word length & format diagnostic.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", default="data/grpo_train_condition_d.jsonl")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of diverse prompts to test per variant.")
    parser.add_argument("--max-new-tokens", type=int, default=2300)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--use-4bit", action="store_true", help="Load base 7B in 4-bit instead of native bf16.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "A_v2_raw_no_chat",
            "B_v2_chat_baseline",
            "C_chat_budget_short_think",
            "D_chat_budget_no_think",
        ],
    )
    parser.add_argument("--output-dir", default="eval_outputs/base_length_diag")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("HF_HOME") and Path("/user-data").is_dir():
        os.environ["HF_HOME"] = "/user-data/.hf_cache"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    samples = select_diverse_samples(Path(args.dataset), args.num_samples)
    print(f"[Base Length Diag] Model   : {args.model} ({'4-bit NF4' if args.use_4bit else 'native bf16'})", flush=True)
    print(f"[Base Length Diag] Samples : {len(samples)} chapters ({[s.get('sample_id') for s in samples]})", flush=True)
    print(f"[Base Length Diag] Variants: {args.variants}", flush=True)
    print(f"[Base Length Diag] Decoding: max_new_tokens={args.max_new_tokens}, temp={args.temperature}, top_p={args.top_p}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": compute_dtype,
        "device_map": "cuda:0" if torch.cuda.is_available() else "auto",
        "attn_implementation": "sdpa",
        "trust_remote_code": True,
    }
    if args.use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    print(f"[Base Length Diag] Loaded model in {time.time() - t_load:.1f}s", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for variant in args.variants:
        t0 = time.time()
        formatted_prompts: List[str] = []
        for row in samples:
            messages, raw_prompt = build_variant_payload(row, variant)
            if messages is not None:
                prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_str = raw_prompt or ""
            formatted_prompts.append(prompt_str)

        enc = tok(formatted_prompts, return_tensors="pt", padding=True).to(model.device)
        in_len = enc["input_ids"].shape[1]
        print(f"\n--- Running {variant} (batch={len(samples)}, max_prompt_tok={in_len}) ---", flush=True)

        with torch.inference_mode():
            out_ids = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
            )

        elapsed = time.time() - t0
        var_records: List[Dict[str, Any]] = []
        for i, row in enumerate(samples):
            gen_slice = out_ids[i][in_len:]
            # Count non-pad generated tokens accurately
            non_pad = [int(t) for t in gen_slice.tolist() if int(t) != tok.pad_token_id]
            gen_text = tok.decode(gen_slice, skip_special_tokens=True).strip()
            rec = evaluate_generated_output(
                row=row,
                raw_text=gen_text,
                gen_tokens=len(non_pad),
                max_new_tokens=args.max_new_tokens,
                variant=variant,
            )
            var_records.append(rec)
            all_records.append(rec)
            print(
                f"  Ch{rec['chapter_index']} ({rec['sample_id']}): "
                f"prose={rec['prose_words']:4d}w (tgt={rec['target_words']}w) | "
                f"think={rec['think_words']:3d}w | tok={rec['gen_tokens']:4d} "
                f"{'[CLIP]' if rec['clipped'] else '[EOS] '} | "
                f"RLMR={'PASS' if rec['is_compliant'] else 'FAIL'} {rec['failed_constraints']}",
                flush=True,
            )

        s = summarize_variant(variant, var_records, elapsed)
        summaries.append(s)

    print("\n" + "=" * 112)
    print(f"{'Variant':<26} | {'Min':>4} {'Med':>4} {'Mean':>4} {'Max':>4} | {'Think':>5} | {'V_len':>6} {'V_fmt':>6} {'V_loop':>6} | {'RLMR':>6} | {'Clip':>5} | {'Time':>5}")
    print("-" * 112)
    for s in summaries:
        print(
            f"{s['variant']:<26} | {s['prose_min']:4d} {s['prose_median']:4d} {s['prose_mean']:4d} {s['prose_max']:4d} | "
            f"{s['think_mean']:4d}w | {s['v_length_pct']:5.1f}% {s['v_format_pct']:5.1f}% {s['v_no_loop_pct']:5.1f}% | "
            f"{s['compliant_pct']:5.1f}% | {s['clipped_pct']:4.1f}% | {s['elapsed_s']:4.0f}s"
        )
    print("=" * 112)

    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "samples.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_records) + "\n",
        encoding="utf-8",
    )
    print(f"[Saved] {out_dir / 'summary.json'} & {out_dir / 'samples.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
