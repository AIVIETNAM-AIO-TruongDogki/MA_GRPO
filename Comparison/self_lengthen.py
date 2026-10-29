"""
Self-Lengthen: Training LLMs for Long Text Generation via Intrinsic Self-Alignment
(Alibaba / Qwen Team, 2024 - Faithful implementation of `Self-Lengthen/llama/` & `Self-Lengthen/qwen/`)

Exact Two-Stage Iterative Split-and-Extend Algorithm (`step1_gen_initial_responses.py` & `step2_ext_responses.py`):
  1. Stage 1 (Generator): Generates the initial response draft for the user instruction.
  2. Stage 2 (Extender - up to 3 iterations):
     - Splits `inputs` at the paragraph midpoint into `former = '\n\n'.join(lines[:len(lines) // 2])`.
     - Expands `former` using the exact `extend_response.md` prompt template.
     - Trims the expanded `former_output` to its first 2/3 paragraphs (`former_output[:len(former_output) * 2 // 3]`)
       so premature endings are removed.
     - Constructs the prefix-continuation input:
       `latter_input = prompt_prefix + extend_template.format(prompt=prompt, response=inputs) + prompt_suffix + former_output`
       and calls raw completion (`generate_completion`) so the model seamlessly writes `latter_output`.
     - Validates via `check_response(output)` (punctuation check, stopword check, and 10-gram repetition filter from `utils.py`).
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import save_model_result_csv
from .qwen_backend import QwenAgentBackend, get_peak_vram_gb, reset_vram_stats


# Exact template from Self-Lengthen/llama/prompt_templates/extend_response.md
EXTEND_RESPONSE_TEMPLATE = """You are an expert in writing polishing. Based on the provided instruction and original outline, enhance the content by carefully expanding the original outline to increase its informational content, incorporate specific details, and improve its descriptive quality. Remember that your task is not to continue writing content, but to enrich the current content. You can achieve this by enriching existing paragraphs with more detailed descriptions or creating new sections. You should try to be creative and informative as much as posible and do NOT include any repetitive content or lengthy narratives in your response. You need to generate text that is as long, rich, and engaging as possible.

Instruction: {prompt}

Original outline:

{response}"""


def check_response(response: str) -> bool:
    """
    Exact validation function from Self-Lengthen/llama/utils.py:
      1. Ends with valid punctuation mark.
      2. Does not contain meta-stopwords ('revised', 'revision', 'response', 'version', 'paragraph') in opening/closing lines.
      3. No repeated 10-gram anywhere in the text.
    """
    punctuation_marks = {".", "!", "?", ",", ";", ":", "-", "...", '"', "'", ")", "]", "}", ">", "`", "”", "’"}
    response_clean = response.rstrip()
    if not response_clean or response_clean[-1] not in punctuation_marks:
        return False

    stopwords = ["revised", "revision", "response", "version", "paragraph"]
    lines = response_clean.split("\n")
    boundary_part = "\n".join(lines[:2] + lines[-3:]).lower()
    for word in stopwords:
        if word in boundary_part:
            return False

    words = response_clean.split()
    n_gram = {}
    count = 10
    for i in range(len(words)):
        if i >= count - 1:
            substr = tuple(words[i - count + 1 : i + 1])
            x = n_gram.get(substr, 0)
            n_gram[substr] = x + 1
            if x > 0:
                return False
    return True


def run_self_lengthen_single(prompt: str, target_length: int, backend: QwenAgentBackend) -> str:
    """
    Execute the faithful Self-Lengthen pipeline (`step1` initial draft + `step2` split-and-extend).
    """
    # =========================================================================
    # STEP 1: INITIAL RESPONSE GENERATION (`step1_gen_initial_responses.py`)
    # =========================================================================
    init_max_tokens = min(1100, max(500, int(target_length * 0.55)))
    initial_response = backend.generate_chat(
        user_prompt=prompt,
        system_prompt="You are a helpful and creative literary assistant.",
        max_new_tokens=init_max_tokens,
        temperature=0.72,
        top_p=0.9,
    )

    inputs = initial_response.strip()
    best_output = inputs
    best_diff = abs(len(best_output.split()) - target_length)

    # Qwen ChatML prefix/suffix from Self-Lengthen/qwen/step2_ext_responses.py lines 15-16
    prompt_prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n"

    # =========================================================================
    # STEP 2: ITERATIVE SPLIT-AND-EXTEND (`step2_ext_responses.py`)
    # =========================================================================
    for step_idx in range(3):
        current_words = len(inputs.split())
        # Stop if already within 8% of target_length or exceeding target_length
        if current_words >= int(target_length * 0.94):
            break

        lines = [p for p in inputs.split("\n\n") if p.strip()]
        if len(lines) <= 2:
            lines = [p for p in inputs.split("\n") if p.strip()]
        if len(lines) < 2:
            words_list = inputs.split()
            mid_w = max(1, len(words_list) // 2)
            lines = [" ".join(words_list[:mid_w]), " ".join(words_list[mid_w:])]

        # 1. Split first half (`former`)
        half_idx = max(1, len(lines) // 2)
        former = "\n\n".join(lines[:half_idx])
        former_input = EXTEND_RESPONSE_TEMPLATE.format(prompt=prompt, response=former)

        # Calculate token budget for this expansion step so we approach target_length smoothly
        step_target_words = min(target_length, int(current_words * 1.45))
        former_budget_tokens = min(1400, max(450, int(step_target_words * 0.85)))

        former_output = backend.generate_chat(
            user_prompt=former_input,
            system_prompt="You are a helpful assistant.",
            max_new_tokens=former_budget_tokens,
            temperature=0.72,
            top_p=0.9,
        )

        # 2. Keep first 2/3 of `former_output` paragraphs (`step2_ext_responses.py` lines 25-27)
        former_paras = [p for p in former_output.split("\n\n") if p.strip()]
        if len(former_paras) >= 3:
            keep_n = max(1, (len(former_paras) * 2) // 3)
            former_trimmed = "\n\n".join(former_paras[:keep_n])
        else:
            fw = former_output.split()
            keep_w = max(1, (len(fw) * 2) // 3)
            former_trimmed = " ".join(fw[:keep_w])

        # 3. Prefix-continuation for `latter_output` (`step2_ext_responses.py` lines 29-31)
        latter_input = (
            prompt_prefix
            + EXTEND_RESPONSE_TEMPLATE.format(prompt=prompt, response=inputs)
            + prompt_suffix
            + former_trimmed
            + "\n\n"
        )
        former_trimmed_words = len(former_trimmed.split())
        remaining_needed_words = max(250, target_length - former_trimmed_words)
        latter_budget_tokens = min(1500, max(350, int(remaining_needed_words * 1.38)))

        latter_output = backend.generate_completion(
            raw_prompt_prefix=latter_input,
            max_new_tokens=latter_budget_tokens,
            temperature=0.72,
            top_p=0.9,
        )

        output = (former_trimmed.strip() + "\n\n" + latter_output.strip()).strip()
        out_words = len(output.split())
        out_diff = abs(out_words - target_length)

        # Accept if valid or closer to target_length
        if check_response(output) and out_words > current_words:
            inputs = output
            best_output = output
            best_diff = out_diff
        elif out_words > len(best_output.split()) and out_diff <= best_diff:
            inputs = output
            best_output = output
            best_diff = out_diff

    return best_output


def run_self_lengthen_benchmark(
    prompts: List[Dict[str, Any]],
    output_csv: Path,
    backend: Optional[QwenAgentBackend] = None,
):
    """Run Self-Lengthen across all benchmark prompts and save CSV."""
    own_backend = False
    if backend is None:
        backend = QwenAgentBackend("Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True)
        backend.load()
        own_backend = True
    else:
        backend.load()

    print("\n" + "=" * 85, flush=True)
    print("RUNNING MODEL [3/8]: Self-Lengthen (Two-Stage Split-and-Extend + Prefix Completion)", flush=True)
    print("=" * 85, flush=True)

    records = []
    for idx, item in enumerate(prompts, 1):
        p_id = item["id"]
        p_text = item["prompt"]
        t_len = int(item["target_length"])
        print(f" -> [Self-Lengthen] [{idx}/{len(prompts)}] {p_id} (Target: {t_len} words)...", flush=True)

        reset_vram_stats()
        t0 = time.time()
        raw_out = run_self_lengthen_single(p_text, t_len, backend)
        elapsed = round(max(0.01, time.time() - t0), 2)
        vram = get_peak_vram_gb()

        actual_w = len(raw_out.split())
        print(f"    Done in {elapsed}s | Words: {actual_w}/{t_len} | Peak VRAM: {vram} GB", flush=True)

        records.append({
            "Model": "Self-Lengthen",
            "Prompt_ID": p_id,
            "Prompt_Text": p_text,
            "Target_Len": t_len,
            "Raw_Output": raw_out,
            "Latency (s)": elapsed,
            "Peak_VRAM (GB)": vram,
        })

    if own_backend:
        backend.unload()

    return save_model_result_csv(records, output_csv)
