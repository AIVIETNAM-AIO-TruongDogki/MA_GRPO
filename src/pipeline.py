"""
End-to-End Orchestrator Pipeline for MA-GRPO with vLLM / HF / Mock Integration.
Core Paper Architecture:
    1. Hardware-aware precision (bfloat16 on Ampere/Hopper A100/H100/4090, float16 on T4/V100).
    2. Single-chapter inference and parallel GRPO rollouts (n=G) via vLLM PagedAttention.
    3. Clean Stage-1 Macro Planning -> Stage-2 Two-Tier Coordinator & Writing Policy execution.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .coordinator import CoordinatorAgent, format_long_term_memory_block
from .planning import (
    OutlineAgent,
    PlanningAgent,
    estimate_token_count,
)
from .rewards import count_words, strip_think_tags


_ARTIFACT_TAG_RE = re.compile(
    r"(?:<think>[\s\S]*?</think>|"
    r"</?think>|"
    r"</?assistant>|"
    r"<\|im_start\|>(?:assistant|user|system)?|"
    r"<\|im_end\|>|"
    r"<\|endoftext\|>|"
    r"\[/?INST\]|"
    r"</?s>)",
    flags=re.IGNORECASE,
)

WRITER_SYSTEM_PROMPT_COT = (
    "You are an acclaimed literary novelist writing full-length chapters of approximately 1,000 words. "
    "You NEVER summarize or rush scenes; you fully dramatize every beat with concrete sensory detail, "
    "physical blocking, multi-turn character dialogue, and psychological depth so the chapter genuinely "
    "reaches the target narrative length (about 11-13 rich paragraphs of 80-95 words each for a 1,000-word chapter). "
    "You strictly obey formatting rules: your <think> block contains ONLY 4-6 short bullet points "
    "(under 100 words total) and NEVER contains narrative prose or dialogue."
)

WRITER_SYSTEM_PROMPT_DIRECT = (
    "You are an acclaimed literary novelist writing full-length chapters of approximately 1,000 words. "
    "You NEVER summarize or rush scenes; you fully dramatize every beat with concrete sensory detail, "
    "physical blocking, multi-turn character dialogue, and psychological depth so the chapter genuinely "
    "reaches the target narrative length. Output ONLY clean narrative prose with zero headings, "
    "scene labels, <think> tags, or author notes."
)


def annotate_scene_paragraph_budgets(text: str) -> str:
    """Idempotently annotate `Scene N (~X words)` lines with explicit paragraph counts (~85w/para)."""
    if not text or "full, detailed paragraphs)" in text:
        return text

    def _repl(match: re.Match) -> str:
        prefix = match.group(1)
        words = int(match.group(2))
        paras = max(2, int(round(words / 85.0)))
        return f"{prefix}(~{words} words — write {paras} full, detailed paragraphs)"

    return re.sub(
        r"(Scene\s+\d+\s*)\(~(\d+)\s*words\)",
        _repl,
        text,
        flags=re.IGNORECASE,
    )


def build_writer_chat_messages(
    raw_prompt: str,
    target_words: int = 1000,
    enable_cot: bool = True,
) -> List[Dict[str, str]]:
    """Build synchronized [system, user] chat messages used identically across Training and Eval.

    Proven by Base 7B diagnostic (`C_chat_budget_short_think`):
      - 100% V_format & V_no_loop compliance (zero prose leaked inside `<think>`)
      - Concise ~55-word bullet-only `<think>` plan
      - Explicit per-scene paragraph budgets (~85w/paragraph) to counter Instruct brevity bias.
    """
    text = (raw_prompt or "").strip()
    for marker in ("STRICT LENGTH & FORMAT REQUIREMENTS:", "CRITICAL WRITING INSTRUCTIONS:"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].rstrip()
            break

    target = max(200, int(target_words or 1000))
    l_low = int(round(target * 0.85))
    l_high = int(round(target * 1.15))
    total_paras = max(6, int(round(target / 85.0)))
    core_body = annotate_scene_paragraph_budgets(text)

    if enable_cot:
        user_msg = (
            f"{core_body}\n\n"
            f"STRICT LENGTH & FORMAT REQUIREMENTS:\n"
            f"1. Brief <think> Plan (MAX 80-100 words, bullet points ONLY):\n"
            f"   - Start with <think> and write ONLY 4-6 concise bullet points mapping paragraph counts and transitions.\n"
            f"   - DO NOT write any story sentences, descriptions, or dialogue inside <think>.\n"
            f"   - Close </think> immediately after the bullets.\n"
            f"2. Full Chapter Prose ({l_low}-{l_high} words, target ~{target} words):\n"
            f"   - Immediately after </think>, write approximately {total_paras} rich paragraphs (~80-95 words each).\n"
            f"   - Fully dramatize every scene with real-time action, sensory atmosphere, internal tension, and extended back-and-forth dialogue.\n"
            f"   - NEVER summarize events in a few sentences, NEVER skip ahead, and NEVER stop early around 400-600 words.\n"
            f"   - Do NOT include chapter/scene labels, markdown headings, or meta-commentary after </think>.\n"
            f"   - End cleanly on a complete final sentence.\n"
            f"Output <think>...</think> followed immediately by the complete ~{target}-word chapter:"
        )
        return [
            {"role": "system", "content": WRITER_SYSTEM_PROMPT_COT},
            {"role": "user", "content": user_msg},
        ]

    user_msg = (
        f"{core_body}\n\n"
        f"STRICT LENGTH & FORMAT REQUIREMENTS:\n"
        f"- Target Prose Length: strictly between {l_low} and {l_high} words (aim right around {target} words).\n"
        f"- Structure: write approximately {total_paras} substantial paragraphs (~80-95 words each) following the scene budgets above.\n"
        f"- Dramatization: expand every scene in real time with concrete sensory texture, physical gestures, internal conflict, and multi-turn dialogue.\n"
        f"- Anti-Brevity Rule: DO NOT summarize events, DO NOT wrap up early, and DO NOT repeat earlier paragraphs.\n"
        f"- Clean Prose Only: DO NOT output <think> tags, scene labels, section titles, or author notes. End cleanly on a complete final sentence.\n"
        f"Begin the complete ~{target}-word narrative prose immediately:"
    )
    return [
        {"role": "system", "content": WRITER_SYSTEM_PROMPT_DIRECT},
        {"role": "user", "content": user_msg},
    ]


def clean_generation_artifacts(text: str) -> str:
    """
    Lightweight 0-token Artifact Cleaner:
    Strips stray chat/reasoning special tokens (`<think>`, `</think>`, `</assistant>`,
    `<|im_start|>`, `<|im_end|>`, `[INST]`, `[/INST]`, `<s>`, `</s>`) and normalizes whitespace.
    """
    if not text:
        return ""
    cleaned = _ARTIFACT_TAG_RE.sub("", text)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _normalize_para_for_dedup(paragraph: str) -> str:
    return re.sub(r"\W+", " ", paragraph.lower()).strip()


def _deduplicate_sentences_in_paragraph(
    paragraph: str,
    min_sentence_words: int = 12,
    similarity_threshold: float = 0.88,
) -> Tuple[str, int]:
    """Removes repeated long sentences (> `min_sentence_words`) looping inside the same paragraph."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph.strip()) if s.strip()]
    if len(sentences) < 2:
        return paragraph, 0

    kept: List[str] = []
    seen_norm: List[str] = []
    seen_toks: List[set] = []
    removed = 0
    for sent in sentences:
        if len(sent.split()) <= min_sentence_words:
            kept.append(sent)
            continue
        norm = _normalize_para_for_dedup(sent)
        toks = set(norm.split())
        is_dup = False
        for p_norm, p_toks in zip(seen_norm, seen_toks):
            if norm == p_norm or (toks and p_toks and (len(toks & p_toks) / float(len(toks | p_toks))) >= similarity_threshold):
                is_dup = True
                break
        if is_dup:
            removed += 1
        else:
            kept.append(sent)
            seen_norm.append(norm)
            seen_toks.append(toks)
    return " ".join(kept), removed


def deduplicate_repeated_paragraphs(
    text: str,
    min_words: int = 30,
    similarity_threshold: float = 0.88,
    min_sentence_words: int = 12,
) -> Tuple[str, int]:
    """
    Lightweight 0-token Deduplication Gate:
      1. Removes repeated long paragraphs (> `min_words`, default 30 words) appearing a second
         time within the same chapter (exact normalized match or token Jaccard >= `similarity_threshold`).
      2. Removes repeated long sentences (> `min_sentence_words`, default 12 words) looping
         inside a single continuous paragraph.
    Returns `(deduped_text, removed_count)`.
    """
    if not text or not text.strip():
        return "", 0

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    kept: List[str] = []
    seen_norm: List[str] = []
    seen_token_sets: List[set] = []
    removed = 0

    for para in paragraphs:
        para, sent_removed = _deduplicate_sentences_in_paragraph(
            para, min_sentence_words=min_sentence_words, similarity_threshold=similarity_threshold
        )
        removed += sent_removed
        words = para.split()
        if len(words) <= min_words:
            if para:
                kept.append(para)
            continue

        norm = _normalize_para_for_dedup(para)
        toks = set(norm.split())
        is_dup = False
        for prev_norm, prev_toks in zip(seen_norm, seen_token_sets):
            if norm == prev_norm:
                is_dup = True
                break
            if toks and prev_toks:
                jaccard = len(toks & prev_toks) / float(len(toks | prev_toks))
                if jaccard >= similarity_threshold:
                    is_dup = True
                    break

        if is_dup:
            removed += 1
        else:
            kept.append(para)
            seen_norm.append(norm)
            seen_token_sets.append(toks)

    return "\n\n".join(kept), removed


def extract_cot_and_prose(text: str) -> Tuple[str, str]:
    """
    Extracts the <think>...</think> Test-Time reasoning chain and the actual narrative prose.
    Applies Artifact Cleaner to prose so no orphan tags leak into stored chapters.
    Returns:
        (cot_text, prose_text)
    """
    if not text:
        return "", ""
    think_match = re.search(r"<think>([\s\S]*?)</think>", text, flags=re.IGNORECASE)
    cot_text = think_match.group(1).strip() if think_match else ""
    if re.search(r"</think>", text, flags=re.IGNORECASE):
        prose_raw = re.split(r"</think>", text, maxsplit=1, flags=re.IGNORECASE)[1]
    else:
        prose_raw = re.sub(r"<think>[\s\S]*", "", text, flags=re.IGNORECASE) if re.search(r"<think>", text, flags=re.IGNORECASE) else text
    return cot_text, clean_generation_artifacts(prose_raw)


def resolve_optimal_dtype(preferred: str = "auto") -> str:
    """Resolve bfloat16 on Ampere+ GPUs (A100/H100/4090) and float16 on Turing (T4)."""
    if preferred in {"bfloat16", "float16", "float32"}:
        return preferred
    try:
        import torch
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            return "bfloat16" if major >= 8 else "float16"
    except Exception:
        pass
    return "float16"


class VLLMWritingPolicy:
    """
    Writing Policy supporting vLLM, custom HuggingFace/API callback, or mock generation.
    Supports Condition A (Direct), Condition B (Unstructured), and Conditions C/D/E (Structured).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.65,
        max_model_len: int = 32768,
        dtype: str = "auto",
        use_mock: bool = False,
        custom_generate_fn: Optional[Callable[[str, int, float, float], str]] = None,
    ):
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.use_mock = use_mock
        self.custom_generate_fn = custom_generate_fn
        self.llm = None
        self.dtype = resolve_optimal_dtype(dtype)

        if not self.use_mock and self.custom_generate_fn is None:
            try:
                from vllm import LLM
                print(
                    f"[vLLM] Initializing engine with model: {model_name} "
                    f"({self.dtype}, VRAM limit: {gpu_memory_utilization * 100}%)..."
                )
                self.llm = LLM(
                    model=model_name,
                    tensor_parallel_size=tensor_parallel_size,
                    dtype=self.dtype,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=max_model_len,
                    trust_remote_code=True,
                )
            except (ImportError, Exception) as e:
                print(f"[vLLM] Warning: vLLM engine unavailable ({e}). Falling back to mock generator.")
                self.use_mock = True

    def _format_vllm_prompt(self, prompt: str, enable_cot: bool = True) -> str:
        """Apply model chat template with synchronized system+user messages when supported by vLLM tokenizer."""
        if self.llm is None:
            return prompt
        try:
            tok = self.llm.get_tokenizer()
            if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
                sys_msg = WRITER_SYSTEM_PROMPT_COT if enable_cot else WRITER_SYSTEM_PROMPT_DIRECT
                return tok.apply_chat_template(
                    [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception:
            pass
        return prompt

    def _clamp_max_tokens(self, formatted_prompt: str, requested_max_tokens: int) -> int:
        """Clamp max_tokens so prompt_tokens + max_tokens fits inside vLLM max_model_len."""
        if self.llm is None:
            return max(1, requested_max_tokens)
        try:
            tok = self.llm.get_tokenizer()
            prompt_tokens = len(tok.encode(formatted_prompt, add_special_tokens=False))
        except Exception:
            prompt_tokens = max(1, int(len(formatted_prompt.split()) * 1.4))
        available = max(1, self.max_model_len - prompt_tokens - 32)
        return max(1, min(requested_max_tokens, available))

    def generate_raw(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        mock_fn: Optional[Callable[[], str]] = None,
    ) -> str:
        """Execute single-prompt generation via custom callback, vLLM, or mock fallback."""
        if self.custom_generate_fn is not None:
            return self.custom_generate_fn(prompt, max_tokens, temperature, top_p)
        if self.use_mock or self.llm is None:
            return mock_fn() if mock_fn is not None else ""
        from vllm import SamplingParams

        formatted = self._format_vllm_prompt(prompt)
        safe_max_tokens = self._clamp_max_tokens(formatted, max_tokens)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=safe_max_tokens,
            presence_penalty=0.1,
        )
        outputs = self.llm.generate([formatted], sampling_params)
        return outputs[0].outputs[0].text

    def generate_chapter_prose(
        self,
        chapter_plan: Dict[str, Any],
        context_summary: Dict[str, Any],
        temperature: float = 0.85,
        top_p: float = 0.95,
        max_tokens: int = 2200,
        enable_cot: bool = True,
        enable_artifact_cleaner: bool = True,
        enable_dedup_gate: bool = True,
        dedup_retry_on_drop: bool = True,
    ) -> str:
        """
        Generates prose for a single structured chapter plan.
        Applies lightweight 0-token Artifact Cleaner and Paragraph Deduplication Gate
        (with optional higher-temperature retry when heavy duplication truncates the chapter).
        Returns strictly clean narrative prose (post-</think>).
        """
        prompt = self.construct_prompt(chapter_plan, context_summary, enable_cot=enable_cot)
        raw_output = self.generate_raw(
            prompt,
            max_tokens,
            temperature,
            top_p,
            lambda: self._mock_prose(chapter_plan, context_summary),
        )
        _, prose_text = extract_cot_and_prose(raw_output)
        prose_text = prose_text if prose_text else (
            clean_generation_artifacts(raw_output) if enable_artifact_cleaner else raw_output.strip()
        )
        if enable_artifact_cleaner:
            prose_text = clean_generation_artifacts(prose_text)

        if enable_dedup_gate and not (self.use_mock and self.custom_generate_fn is None):
            deduped, removed = deduplicate_repeated_paragraphs(prose_text, min_words=30)
            target_words = int(chapter_plan.get("target_words", 0) or 0)
            if (
                removed > 0
                and dedup_retry_on_drop
                and not self.use_mock
                and target_words > 0
                and count_words(deduped) < int(0.75 * target_words)
            ):
                retry_temp = min(1.05, round(temperature + 0.12, 2))
                raw_retry = self.generate_raw(
                    prompt,
                    max_tokens,
                    retry_temp,
                    top_p,
                    lambda: self._mock_prose(chapter_plan, context_summary),
                )
                _, retry_prose = extract_cot_and_prose(raw_retry)
                if enable_artifact_cleaner:
                    retry_prose = clean_generation_artifacts(retry_prose or raw_retry)
                retry_deduped, retry_removed = deduplicate_repeated_paragraphs(retry_prose, min_words=30)
                if retry_removed < removed or count_words(retry_deduped) > count_words(deduped):
                    deduped = retry_deduped
            prose_text = deduped

        return prose_text

    def generate_rollouts(
        self,
        chapter_plan: Dict[str, Any],
        context_summary: Dict[str, Any],
        group_size: int = 8,
        temperature: float = 0.85,
        top_p: float = 0.95,
        max_tokens: int = 2200,
        return_cot: bool = False,
    ) -> List[str]:
        """Generates G candidate rollouts in parallel for GRPO training."""
        prompt = self.construct_prompt(chapter_plan, context_summary)

        if self.custom_generate_fn is not None:
            raw_outputs = [self.custom_generate_fn(prompt, max_tokens, temperature, top_p) for _ in range(group_size)]
        elif self.use_mock or self.llm is None:
            raw_outputs = [self._mock_prose(chapter_plan, context_summary) for _ in range(group_size)]
        else:
            from vllm import SamplingParams

            formatted = self._format_vllm_prompt(prompt)
            safe_max_tokens = self._clamp_max_tokens(formatted, max_tokens)
            sampling_params = SamplingParams(
                n=group_size,
                temperature=temperature,
                top_p=top_p,
                max_tokens=safe_max_tokens,
                presence_penalty=0.1,
            )
            outputs = self.llm.generate([formatted], sampling_params)
            raw_outputs = [o.text.strip() for o in outputs[0].outputs]

        if return_cot:
            return raw_outputs

        cleaned = []
        for raw in raw_outputs:
            _, prose = extract_cot_and_prose(raw)
            cleaned.append(prose if prose else raw)
        return cleaned

    def construct_prompt(
        self,
        chapter_plan: Dict[str, Any],
        context_summary: Dict[str, Any],
        enable_cot: bool = True,
    ) -> str:
        recent_tail = context_summary.get("recent_tail_excerpt", "")
        tail_ref = (
            f"\n[PRECEDING SCENE CONTINUATION (Pick up immediately from this scene)]:\n{recent_tail}\n"
            if recent_tail
            else ""
        )
        active_chars = chapter_plan.get("characters_active", ["Protagonist"])
        scene_outline = chapter_plan.get("scene_outline", [])
        outline_str = "\n- ".join(scene_outline) if scene_outline else chapter_plan.get("chapter_goal", "")
        target_words = int(chapter_plan.get("target_words", 1200) or 1200)
        conflict_directive = (
            f"\n[DIRECTIVE WARNING]: {chapter_plan['injected_contradiction']}"
            if chapter_plan.get("injected_contradiction")
            else ""
        )

        ltm_block = format_long_term_memory_block(context_summary, max_total_words=220)

        core_prompt = f"""You are an elite literary novelist. Write the complete narrative prose for {chapter_plan.get('chapter_title', 'Chapter')}.
Total Target Length: Approximately {target_words} words.

{ltm_block}[CONTEXT SUMMARY]:
{context_summary.get('plot_summary', '')}
Active Characters: {', '.join(active_chars)}
{tail_ref}{conflict_directive}
[CHAPTER GOAL]:
{chapter_plan.get('chapter_goal', '')}

[SCENE BREAKDOWN & WORD BUDGETS]:
- {outline_str}"""

        return build_writer_chat_messages(core_prompt, target_words=target_words, enable_cot=enable_cot)[1]["content"]

    def _mock_prose(self, chapter_plan: Dict[str, Any], context_summary: Dict[str, Any]) -> str:
        ch_title = chapter_plan.get("chapter_title", "Chapter")
        ch_goal = chapter_plan.get("chapter_goal", "The journey continues.")
        plot_sum = context_summary.get("plot_summary", "")
        active_chars = chapter_plan.get("characters_active", ["Protagonist"])
        target = chapter_plan.get("target_words", 1200)
        scenes = chapter_plan.get("scenes", [])
        injected = chapter_plan.get("injected_contradiction", "")

        scene_plans = [
            f"- Scene {s.get('scene_index', i+1)} (~{s.get('target_words', 400)} words): Focus on {s.get('beat_goal', 'advancing plot')}."
            for i, s in enumerate(scenes)
        ] or [f"- Distribute ~{target} words evenly across dramatic arc."]

        cot_block = (
            "<think>\n"
            f"Planning narrative pacing for {ch_title} (Target: {target} words):\n"
            + "\n".join(scene_plans) + "\n"
            "- Establish physical continuity from preceding excerpt without repeating phrases.\n"
            "- Modulate sentence length variation to maintain high burstiness and engagement.\n"
            "</think>\n\n"
        )

        injected_clause = f" {injected[:120]}." if injected else ""
        paragraphs = [
            f"{ch_title.upper()}\n\n"
            f"The morning light pierced the dense gloom, casting long, fractured shadows across the threshold. "
            f"{active_chars[0] if active_chars else 'Elena'} stepped forward calmly, determined to execute the goal: {ch_goal}.{injected_clause} "
            f"Prior context weighed heavily: {plot_sum[:110]}...",
            f"\"We cannot afford to falter now,\" a quiet voice echoed through the silence. "
            f"The figures involved—{', '.join(active_chars)}—faced shifting complications while pursuing the exit objective: "
            f"{chapter_plan.get('exit_state', 'tension builds further')}.",
        ]

        # Add distinct prose per scene to accurately simulate Plan Adherence & Lexical Diversity in offline mock runs
        for s in scenes:
            beat_txt = s.get("beat_goal", "")
            if beat_txt:
                paragraphs.append(f"During this unfolding moment, the focus shifted decisively: {beat_txt}.")

        base_prose = "\n\n".join(paragraphs)
        current_words = count_words(base_prose)

        filler_sentences = [
            "Shadows lengthened as the quiet stillness stretched across every arched corridor around them.",
            "Across the distant ridge, a storm front gathered weight, staining the horizon with violet haze.",
            "Uncertain memories surfaced unexpectedly beneath the weight of unspoken vows and old suspicions.",
            "Ancient stone walls hummed faintly with a low vibration that echoed every measured footstep.",
            "Time slipped steadily past as watchful eyes scanned the shifting perimeter for movement.",
            "Cold air carried the scent of wet iron, cedar smoke, and distant rain.",
            "Every small gesture revealed unspoken calculations hidden behind composed expressions.",
        ]

        extra_text = []
        idx = chapter_plan.get("chapter_index", 1) * 3
        while current_words < target:
            sentence = filler_sentences[idx % len(filler_sentences)]
            extra_text.append(sentence)
            current_words += count_words(sentence)
            idx += 1

        if extra_text:
            base_prose += "\n\n" + " ".join(extra_text)

        return cot_block + base_prose


class MAGRPOPipeline:
    """
    Complete Two-Stage Local-to-Global Generation Pipeline for MA-GRPO.
    Coordinates Stage-1 Macro Planning (OutlineAgent + PlanningAgent),
    Stage-2 Two-Tier Memory Compression (CoordinatorAgent), and Writing Policy generation.
    """

    def __init__(
        self,
        outline_agent: Optional[OutlineAgent] = None,
        planning_agent: Optional[PlanningAgent] = None,
        coordinator_agent: Optional[CoordinatorAgent] = None,
        writing_policy: Optional[VLLMWritingPolicy] = None,
        planner_llm_fn: Optional[Callable[[str, str], str]] = None,
    ):
        self.planner_llm_fn = planner_llm_fn
        self.outline_agent = outline_agent or OutlineAgent(llm_generate_fn=planner_llm_fn)
        self.planning_agent = planning_agent or PlanningAgent(llm_generate_fn=planner_llm_fn)
        self.coordinator_agent = coordinator_agent or CoordinatorAgent(llm_generate_fn=planner_llm_fn)
        self.writing_policy = writing_policy or VLLMWritingPolicy(use_mock=True)

    def execute_chapter_loop(
        self,
        blueprint: Dict[str, Any],
        temperature: float = 0.7,
        top_p: float = 0.9,
        enable_cot: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute rolling multi-chapter generation over a partitioned blueprint
        using Stage-2 CoordinatorAgent and VLLMWritingPolicy.
        """
        t_gen0 = time.time()
        generated_chapters: List[str] = []
        past_plans: List[Dict[str, Any]] = []
        context_summaries: List[Dict[str, Any]] = []

        events = blueprint.get("event_graph", {}).get("events", [])
        all_characters = list(dict.fromkeys(
            ev.get("character", "Protagonist") for ev in events
        )) or ["Protagonist"]

        for ch_plan in blueprint["chapters"]:
            ch_idx = ch_plan["chapter_index"]
            context_summary = self.coordinator_agent.condense_context(
                next_chapter_index=ch_idx,
                past_chapters=generated_chapters,
                past_plans=past_plans,
                characters=all_characters,
            )
            context_summaries.append(context_summary)

            chapter_prose = self.writing_policy.generate_chapter_prose(
                chapter_plan=ch_plan,
                context_summary=context_summary,
                temperature=temperature,
                top_p=top_p,
                max_tokens=self.writing_policy.max_model_len,
                enable_cot=enable_cot,
            )
            generated_chapters.append(chapter_prose)
            past_plans.append(ch_plan)

        divider = "\n\n" + ("=" * 80) + "\n\n"
        full_text = divider.join(generated_chapters)
        return {
            "chapters": generated_chapters,
            "full_text": full_text,
            "actual_total_words": count_words(full_text),
            "generation_latency_s": round(time.time() - t_gen0, 2),
            "context_summaries": context_summaries,
        }

    def generate_story(
        self,
        premise: str,
        genre: str = "Science Fiction",
        total_target_words: int = 12000,
        num_chapters: int = 10,
        enable_micro_subevents: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end rolling multi-agent generation (backward-compatible default: E / configurable D).
        """
        start_time = time.time()

        event_graph = self.outline_agent.create_event_graph(
            premise=premise,
            genre=genre,
            num_events=max(4, num_chapters // 2),
        )

        blueprint = self.planning_agent.create_story_plan(
            event_graph=event_graph,
            total_target_words=total_target_words,
            num_chapters=num_chapters,
            enable_micro_subevents=enable_micro_subevents,
        )

        generated_chapters: List[str] = []
        past_plans: List[Dict[str, Any]] = []
        context_summaries: List[Dict[str, Any]] = []

        all_characters = list(dict.fromkeys(
            event.get("character", "Protagonist") for event in event_graph.get("events", [])
        )) or ["Protagonist"]

        for ch_idx in range(1, num_chapters + 1):
            context_summary = self.coordinator_agent.condense_context(
                next_chapter_index=ch_idx,
                past_chapters=generated_chapters,
                past_plans=past_plans,
                characters=all_characters,
            )
            context_summaries.append(context_summary)

            ch_plan = self.planning_agent.plan_next_chapter(
                story_plan=blueprint,
                context_summary=context_summary,
            )

            chapter_prose = self.writing_policy.generate_chapter_prose(
                chapter_plan=ch_plan,
                context_summary=context_summary,
                max_tokens=self.writing_policy.max_model_len,
            )

            generated_chapters.append(chapter_prose)
            past_plans.append(ch_plan)
            self.planning_agent.mark_chapter_complete(blueprint, ch_plan)

        divider = "\n\n" + ("=" * 80) + "\n\n"
        full_text = divider.join(generated_chapters)
        actual_total_words = count_words(full_text)
        elapsed_time = round(time.time() - start_time, 2)

        return {
            "title": f"Story: {blueprint.get('genre', 'Novel')}",
            "blueprint": blueprint,
            "chapters": generated_chapters,
            "full_text": full_text,
            "target_words": total_target_words,
            "actual_total_words": actual_total_words,
            "num_chapters": num_chapters,
            "elapsed_seconds": elapsed_time,
            "context_summaries": context_summaries,
        }
