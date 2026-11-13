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

from .coordinator import CoordinatorAgent
from .planning import (
    OutlineAgent,
    PlanningAgent,
    estimate_token_count,
)
from .rewards import count_words, strip_think_tags


def extract_cot_and_prose(text: str) -> Tuple[str, str]:
    """
    Extracts the <think>...</think> Test-Time reasoning chain and the actual narrative prose.
    Returns:
        (cot_text, prose_text)
    """
    if not text:
        return "", ""
    think_match = re.search(r"<think>([\s\S]*?)</think>", text)
    cot_text = think_match.group(1).strip() if think_match else ""
    if "</think>" in text:
        prose_text = text.split("</think>", 1)[1].strip()
    else:
        prose_text = re.sub(r"<think>[\s\S]*", "", text).strip() if "<think>" in text else text.strip()
    return cot_text, prose_text


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
        max_model_len: int = 8192,
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

    def _format_vllm_prompt(self, prompt: str) -> str:
        """Apply model chat template when supported by vLLM tokenizer."""
        if self.llm is None:
            return prompt
        try:
            tok = self.llm.get_tokenizer()
            if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
                return tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
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
    ) -> str:
        """
        Generates prose for a single structured chapter plan.
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
        return prose_text if prose_text else raw_output.strip()

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
        target_words = chapter_plan.get("target_words", 1200)
        story_arc = chapter_plan.get("story_arc")
        arc_line = f"\nNarrative Arc Stage: {story_arc}" if story_arc else ""
        conflict_directive = (
            f"\n[DIRECTIVE WARNING]: {chapter_plan['injected_contradiction']}"
            if chapter_plan.get("injected_contradiction")
            else ""
        )

        cot_instructions = (
            "CRITICAL WRITING INSTRUCTIONS:\n"
            "1. Test-Time Reasoning (Chain-of-Thought):\n"
            "   You MUST begin your response with a <think>...</think> block containing your step-by-step narrative plan:\n"
            "   - Scene-by-scene pacing: Plan word distribution to strictly match each scene's target words.\n"
            "   - Narrative transition: Hook directly into the preceding scene continuation without verbatim sentence repetition.\n"
            "   - Tone & Sensory palette: Outline sensory imagery and avoid literal phrasing from the context summary.\n"
            "2. Chapter Prose:\n"
            "   Immediately after </think>, output ONLY the final narrative prose.\n"
            "   - Seamlessly flow between scenes according to the budgets.\n"
            "   - DO NOT include scene labels, section titles, author notes, or meta-commentary outside <think>.\n"
            "Begin with <think> now:\n"
            if enable_cot
            else "CRITICAL WRITING INSTRUCTIONS:\n"
            "Output ONLY the final narrative prose seamlessly executing every scene goal and target word budget.\n"
            "DO NOT include scene labels, section titles, or meta-commentary.\n"
        )

        return f"""You are an elite literary novelist. Write the complete narrative prose for {chapter_plan.get('chapter_title', 'Chapter')}.{arc_line}
Total Target Length: Approximately {target_words} words.

[CONTEXT SUMMARY]:
{context_summary.get('plot_summary', '')}
Active Characters: {', '.join(active_chars)}
{tail_ref}{conflict_directive}
[CHAPTER GOAL]:
{chapter_plan.get('chapter_goal', '')}

[SCENE BREAKDOWN & WORD BUDGETS]:
- {outline_str}

{cot_instructions}"""

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
                max_tokens=max(1600, int(ch_plan.get("target_words", 1200) * 1.45)),
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
                max_tokens=int(ch_plan.get("target_words", 1200) * 1.3),
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
