"""
End-to-End Orchestrator Pipeline for MA-GRPO with vLLM Integration.
Optimized for high-throughput generation on Server 40GB (A100 / A6000 / L40).
Supports:
    1. Single chapter inference via vLLM PagedAttention.
    2. Parallel rollout generation (n=G) for GRPO training.
    3. Seamless mock fallback for offline local unit testing.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import time

from .coordinator import CoordinatorAgent
from .planning import OutlineAgent, PlanningAgent
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


class VLLMWritingPolicy:
    """
    vLLM-backed Writing Policy for ultra-fast chapter generation and GRPO rollouts.
    Takes full advantage of 40GB VRAM via PagedAttention and FlashAttention-2.
    Integrates Test-Time Scaling (<think>...</think> reasoning) and scene-level word budgets.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.65,
        max_model_len: int = 8192,
        use_mock: bool = False,
    ):
        self.model_name = model_name
        self.use_mock = use_mock
        self.llm = None

        if not self.use_mock:
            try:
                from vllm import LLM
                print(f"[vLLM] Initializing engine with model: {model_name} (bfloat16, VRAM limit: {gpu_memory_utilization*100}%)...")
                self.llm = LLM(
                    model=model_name,
                    tensor_parallel_size=tensor_parallel_size,
                    dtype="bfloat16",
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=max_model_len,
                    trust_remote_code=True,
                )
            except (ImportError, Exception) as e:
                print(f"[vLLM] Warning: vLLM engine unavailable ({e}). Falling back to mock generator.")
                self.use_mock = True

    def generate_chapter_prose(
        self,
        chapter_plan: Dict[str, Any],
        context_summary: Dict[str, Any],
        temperature: float = 0.85,
        top_p: float = 0.95,
        max_tokens: int = 2200,
    ) -> str:
        """
        Generates prose for a single chapter using vLLM with Test-Time CoT reasoning.
        Returns strictly the clean narrative prose (post-</think>).
        """
        prompt = self.construct_prompt(chapter_plan, context_summary)

        if self.use_mock or self.llm is None:
            raw_output = self._mock_prose(chapter_plan, context_summary)
        else:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                presence_penalty=0.1,
            )

            outputs = self.llm.generate([prompt], sampling_params)
            raw_output = outputs[0].outputs[0].text

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
        """
        Generates G candidate rollouts in parallel for GRPO training.
        Uses vLLM's native parallel sampling (n=group_size) for extreme throughput.
        """
        prompt = self.construct_prompt(chapter_plan, context_summary)

        if self.use_mock or self.llm is None:
            raw_outputs = [self._mock_prose(chapter_plan, context_summary) for _ in range(group_size)]
        else:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                n=group_size,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                presence_penalty=0.1,
            )

            outputs = self.llm.generate([prompt], sampling_params)
            raw_outputs = [o.text.strip() for o in outputs[0].outputs]

        if return_cot:
            return raw_outputs

        cleaned = []
        for raw in raw_outputs:
            _, prose = extract_cot_and_prose(raw)
            cleaned.append(prose if prose else raw)
        return cleaned

    def construct_prompt(self, chapter_plan: Dict[str, Any], context_summary: Dict[str, Any]) -> str:
        recent_tail = context_summary.get("recent_tail_excerpt", "")
        tail_ref = (
            f"\n[PRECEDING SCENE CONTINUATION (Pick up immediately from this scene)]:\n{recent_tail}\n"
            if recent_tail
            else ""
        )
        active_chars = chapter_plan.get("characters_active", ["Protagonist"])
        scene_outline = chapter_plan.get("scene_outline", [])
        outline_str = "\n- ".join(scene_outline)
        target_words = chapter_plan.get("target_words", 1200)

        return f"""You are an elite literary novelist. Write the complete narrative prose for {chapter_plan.get('chapter_title', 'Chapter')}.
Total Target Length: Approximately {target_words} words.

[CONTEXT SUMMARY]:
{context_summary.get('plot_summary', '')}
Active Characters: {', '.join(active_chars)}
{tail_ref}
[CHAPTER GOAL]:
{chapter_plan.get('chapter_goal', '')}

[SCENE BREAKDOWN & WORD BUDGETS]:
- {outline_str}

CRITICAL WRITING INSTRUCTIONS:
1. Test-Time Reasoning (Chain-of-Thought):
   You MUST begin your response with a <think>...</think> block containing your step-by-step narrative plan:
   - Scene-by-scene pacing: Plan word distribution to strictly match each scene's target words.
   - Narrative transition: Hook directly into the preceding scene continuation without verbatim sentence repetition.
   - Tone & Sensory palette: Outline sensory imagery and avoid literal phrasing from the context summary.
2. Chapter Prose:
   Immediately after </think>, output ONLY the final narrative prose.
   - Seamlessly flow between scenes according to the budgets.
   - DO NOT include scene labels, section titles, author notes, or meta-commentary outside <think>.
Begin with <think> now:
"""

    def _mock_prose(self, chapter_plan: Dict[str, Any], context_summary: Dict[str, Any]) -> str:
        ch_title = chapter_plan.get("chapter_title", "Chapter")
        ch_goal = chapter_plan.get("chapter_goal", "The journey continues.")
        plot_sum = context_summary.get("plot_summary", "")
        active_chars = chapter_plan.get("characters_active", ["Protagonist"])
        target = chapter_plan.get("target_words", 1200)
        scenes = chapter_plan.get("scenes", [])

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

        paragraphs = [
            f"{ch_title.upper()}\n\n"
            f"The morning light pierced the dense gloom, casting long, fractured shadows across the stone floor. "
            f"Elena stepped forward, her heart pounding with a steady, fierce rhythm. "
            f"{ch_goal} Every step carried the weight of what had already transpired: {plot_sum[:120]}...",
            f"\"We cannot afford to falter now,\" she murmured, turning to face the lingering silence. "
            f"The air smelled of ozone, rust, and old parchment. The active figures—{', '.join(active_chars)}—stood at the precipice of a momentous discovery. "
            f"Every thread of intention wound tightly around the central mystery, pulling them deeper into the labyrinth.",
            f"Outside, the wind howled like an omen. The trials they faced were not merely physical; they probed the very boundaries of loyalty and purpose. "
            f"With resolute clarity, the decision was made, cementing the transition into the next perilous chapter of their journey.",
        ]

        base_prose = "\n\n".join(paragraphs)
        current_words = count_words(base_prose)

        filler_sentences = [
            "Shadows lengthened as the silence stretched between them, heavy with unspoken truths.",
            "Across the horizon, the tempest gathered strength, painting the sky in violent hues of violet and charcoal.",
            "Memory was an unreliable compass in a territory governed by shifting loyalties.",
            "The ancient machinery hummed beneath their feet, vibrating with dormant energy awaiting ignition.",
            "She checked the perimeter once more, knowing that time was slipping through their fingers like dry sand.",
        ]

        extra_text = []
        idx = 0
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
    """

    def __init__(
        self,
        outline_agent: Optional[OutlineAgent] = None,
        planning_agent: Optional[PlanningAgent] = None,
        coordinator_agent: Optional[CoordinatorAgent] = None,
        writing_policy: Optional[VLLMWritingPolicy] = None,
    ):
        self.outline_agent = outline_agent or OutlineAgent()
        self.planning_agent = planning_agent or PlanningAgent()
        self.coordinator_agent = coordinator_agent or CoordinatorAgent()
        self.writing_policy = writing_policy or VLLMWritingPolicy(use_mock=True)

    def generate_story(
        self,
        premise: str,
        genre: str = "Science Fiction",
        total_target_words: int = 12000,
        num_chapters: int = 10,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multi-agent generation using vLLM on Server 40GB.
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
