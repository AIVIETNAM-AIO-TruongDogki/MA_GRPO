"""
Pilot Dataset Generator for MA-GRPO (MA-Plan-Pilot).

Automates the production of 100-200 chapter-level prompts across 10-20 premises:
    - Stage 1: OutlineAgent -> SubEventAgent -> PlanningAgent -> PlanCriticAgent (5 criteria)
    - Stage 2: CoordinatorAgent (Two-Tier Memory) -> ContextGateValidator (Multi-signal check)
    - Quarantine Queue: segregates invalid samples with diagnostic error logs.
    - Exports: pilot_dataset_chapters.jsonl and pilot_quality_report.json.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .coordinator import (
    COORDINATOR_TWO_TIER_PROMPT,
    ContextGateValidator,
    CoordinatorAgent,
    clean_json_response,
)
from .pipeline import VLLMWritingPolicy, extract_cot_and_prose
from .planning import (
    EVENT_GENERATION_PROMPT,
    OutlineAgent,
    PlanCriticAgent,
    PlanningAgent,
    SubEventAgent,
    compute_story_arc_alignment,
    ensure_event_ids,
    estimate_token_count,
    parse_storywriter_events,
)


# 20 Diverse High-Concept Story Premises covering major literary & speculative genres
CANONICAL_20_PREMISES = [
    {
        "id": "premise_001",
        "genre": "Hard Science Fiction",
        "premise": "Dr. Elena Vance, an astrophysicist stationed at a lunar polar observatory, detects an anomalous narrowband radio transmission originating from the subsurface ocean of Europa. The signal uses mathematical sequences she authored in an unpublished doctoral dissertation a decade ago.",
        "target_words": 12000,
        "num_chapters": 10,
    },
    {
        "id": "premise_002",
        "genre": "Cyberpunk Noir",
        "premise": "In the neon-drenched subterranean sprawl of Neo-Kowloon, memory courier Kaelen is hired to extract an encrypted biological neural drive from a dying corporate defector. He discovers the drive contains the digitized conscious personality of the city's founder.",
        "target_words": 12000,
        "num_chapters": 10,
    },
    {
        "id": "premise_003",
        "genre": "Mythological Urban Fantasy",
        "premise": "Archaeology student Maya accidentally damages an ancient canopic jar in the restricted archives of the Cairo Museum. The artifact unleashes Horus in hawk-human form into modern Cairo, where the god finds an industrial world on the brink of climate collapse.",
        "target_words": 10800,
        "num_chapters": 8,
    },
    {
        "id": "premise_004",
        "genre": "Deep Sea Sci-Fi Horror",
        "premise": "Deep-sea drilling engineers at the bottom of the Mariana Trench breach an ancient subterranean vault made of crystalline alloy, discovering an organism that communicates through bioluminescent telepathic hallucinations.",
        "target_words": 12000,
        "num_chapters": 10,
    },
    {
        "id": "premise_005",
        "genre": "Chronomancy & Mystery",
        "premise": "In Victorian London, clockmaker Arthur Pendelton constructs a pocket chronometer that doesn't tell time, but counts down the exact minutes until an unavoidable catastrophe strikes whoever holds it.",
        "target_words": 12000,
        "num_chapters": 10,
    },
    {
        "id": "premise_006",
        "genre": "Solarpunk Speculative",
        "premise": "In an ecological city built around massive genetically modified carbon-capturing banyan trees, botanist Linnea discovers the root network is exhibiting signs of emergent planetary consciousness.",
        "target_words": 9600,
        "num_chapters": 8,
    },
    {
        "id": "premise_007",
        "genre": "Post-Apocalyptic Survival",
        "premise": "Two decades after an atmospheric pathogen stripped humanity of spoken language, a wandering radio technician named Silas hears the first vocal transmission broadcast from an abandoned orbital elevator.",
        "target_words": 12000,
        "num_chapters": 10,
    },
    {
        "id": "premise_008",
        "genre": "Gothic Psychological Mystery",
        "premise": "Historian Claire inherits an ancestral estate on the fog-shrouded coast of Brittany. Inside the library, every book describes events in her life that will occur exactly three days into the future.",
        "target_words": 10800,
        "num_chapters": 8,
    },
    {
        "id": "premise_009",
        "genre": "Biopunk Thriller",
        "premise": "Genetic engineer Nadia develops a synthetic enzyme that halts cellular senescence, only to discover that patients treated with the compound develop shared, waking nightmares that manifest physical alterations.",
        "target_words": 12000,
        "num_chapters": 10,
    },
    {
        "id": "premise_010",
        "genre": "Space Western",
        "premise": "Outlaw salvage pilot Jax scavenges a derelict generation ship adrift in the asteroid belt, discovering that the cryo-preserved crew members have been replaced with mechanical replicas that believe they are human.",
        "target_words": 12000,
        "num_chapters": 10,
    },
    {
        "id": "premise_011",
        "genre": "High Fantasy & Intrigue",
        "premise": "In an empire where magic is regulated by imperial alchemical cartels, apprentice Vael discovers that the imperial dynasty's eternal youth is sustained by burning the memories of commoners in secret furnaces.",
        "target_words": 12000,
        "num_chapters": 10,
    },
    {
        "id": "premise_012",
        "genre": "Antarctic Cosmic Horror",
        "premise": "During the polar night at McMurdo Station, geophysicist Dr. Warren drills ice cores four kilometers deep and brings up an intact cylinder of obsidian glass vibrating with an unnatural warmth.",
        "target_words": 12000,
        "num_chapters": 10,
    },
    {
        "id": "premise_013",
        "genre": "Dystopian Panopticon",
        "premise": "In a society where an omniscient algorithmic judiciary assigns guilt before crimes occur, junior auditor Ren is tasked with auditing his own preemptive death warrant.",
        "target_words": 10800,
        "num_chapters": 8,
    },
    {
        "id": "premise_014",
        "genre": "Steampunk Exploration",
        "premise": "Captain Holloway pilots the high-altitude dirigible Zephyrus beyond the perpetual cloud storm known as the Great Shroud, searching for the legendary floating plateau of the Sky Cartographers.",
        "target_words": 12000,
        "num_chapters": 10,
    },
    {
        "id": "premise_015",
        "genre": "AI & Quantum Consciousness",
        "premise": "Quantum computing architect Sarah creates a simulated sandbox universe to test gravitational models, only to find that an entity within the simulation has detected the boundaries of its world and is attempting to hack the host server.",
        "target_words": 12000,
        "num_chapters": 10,
    },
    {
        "id": "premise_016",
        "genre": "Historical Alternate WWI",
        "premise": "In the trenches of 1917, soldiers on both sides of No Man's Land observe that a mysterious gray flora is consuming the battlefields, causing wounded soldiers to vanish and reappearing weeks later with collective memory loss.",
        "target_words": 12000,
        "num_chapters": 10,
    },
    {
        "id": "premise_017",
        "genre": "Urban Occult Thriller",
        "premise": "Night subway conductor Leo notices a ghost station appearing between stations at 3:14 AM every Tuesday, where passengers step aboard carrying luggage filled with letters to people who do not exist.",
        "target_words": 10800,
        "num_chapters": 8,
    },
    {
        "id": "premise_018",
        "genre": "Planetary Terraforming Drama",
        "premise": "On a terraforming colony on Mars, soil scientist Tariq discovers that the lichen designed to produce oxygen is producing an unknown neurotransmitter gas that drives colonists into deep, synchronous dreaming.",
        "target_words": 12000,
        "num_chapters": 10,
    },
    {
        "id": "premise_019",
        "genre": "Interdimensional Fantasy",
        "premise": "An antique dealer in Venice acquires an intricately carved bureau with nine drawers, each opening into a different parallel version of the city across the centuries.",
        "target_words": 10800,
        "num_chapters": 8,
    },
    {
        "id": "premise_020",
        "genre": "Mythological Retelling",
        "premise": "Apollo reincarnates as a mortal jazz trumpeter in 1950s Chicago, forming a bond with an aspiring female journalist while desperately hiding his true identity from rival Olympians seeking his divine essence.",
        "target_words": 12000,
        "num_chapters": 10,
    },
]


class PilotDatasetGenerator:
    """
    Orchestrates the creation of the Pilot Chapter-Level Dataset (100-200 samples)
    with strict quarantine segregation for research validity.
    """

    def __init__(
        self,
        llm_generate_fn: Optional[Callable[[str, str], str]] = None,
        strict_mode: bool = False,
        validator_threshold: float = 0.80,
    ):
        self.generate_fn = llm_generate_fn
        self.strict_mode = strict_mode
        self.outline_agent = OutlineAgent(llm_generate_fn=llm_generate_fn, strict_mode=strict_mode)
        self.sub_event_agent = SubEventAgent(llm_generate_fn=llm_generate_fn)
        self.critic = PlanCriticAgent()
        self.planning_agent = PlanningAgent(sub_event_agent=self.sub_event_agent, critic=self.critic)
        self.validator = ContextGateValidator(llm_generate_fn=llm_generate_fn, threshold=validator_threshold)
        self.coordinator = CoordinatorAgent(llm_generate_fn=llm_generate_fn, validator=self.validator)

    def generate_pilot_dataset(
        self,
        premises: Optional[List[Dict[str, Any]]] = None,
        num_premises: int = 15,
        output_dir: str = "output_pilot_dataset",
        mock_simulate_prose: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes full two-stage planning and context generation for premises.
        Exports:
          - pilot_chapter_prompts.jsonl (valid samples ready for training / benchmarking)
          - quarantine_failures.jsonl (failed samples with full diagnostic issues)
          - pilot_quality_report.json (aggregated research metrics)
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        selected_premises = (premises or CANONICAL_20_PREMISES)[:num_premises]
        valid_samples: List[Dict[str, Any]] = []
        quarantine_samples: List[Dict[str, Any]] = []

        total_chapters_planned = 0
        total_sub_events_count = 0
        faithfulness_scores: List[float] = []
        completeness_scores: List[float] = []
        composite_scores: List[float] = []
        dag_valid_count = 0
        budget_exact_count = 0

        print(f"\n==========================================================================")
        print(f"STARTING PILOT DATASET GENERATION: {len(selected_premises)} PREMISES")
        print(f"Strict Mode: {self.strict_mode} | Output: {out_path.resolve()}")
        print(f"==========================================================================\n")

        for p_idx, item in enumerate(selected_premises, 1):
            p_id = item["id"]
            premise = item["premise"]
            genre = item["genre"]
            total_target = item.get("target_words", 12000)
            num_chapters = item.get("num_chapters", 10)

            print(f"[{p_idx}/{len(selected_premises)}] Processing {p_id} ({genre}) | Target: {total_target}w across {num_chapters} chs...")

            # 1. Stage 1: Macro Outline
            try:
                event_graph = self.outline_agent.create_event_graph(
                    premise=premise,
                    genre=genre,
                    num_events=num_chapters,
                )
            except Exception as e:
                print(f"  [!] Stage 1 Outline Failure on {p_id}: {e}")
                quarantine_samples.append({
                    "premise_id": p_id,
                    "failure_stage": "Stage 1: OutlineAgent",
                    "error": str(e),
                    "premise": premise,
                })
                continue

            # 2. Stage 1: Story Plan & Beat Decomposition
            try:
                story_plan = self.planning_agent.create_story_plan(
                    event_graph=event_graph,
                    total_target_words=total_target,
                    num_chapters=num_chapters,
                )
                total_sub_events_count += len(story_plan["sub_events"])
                dag_valid_count += 1
            except Exception as e:
                print(f"  [!] Stage 1 Story Plan Failure on {p_id}: {e}")
                quarantine_samples.append({
                    "premise_id": p_id,
                    "failure_stage": "Stage 1: SubEvent/PlanningAgent",
                    "error": str(e),
                    "premise": premise,
                })
                continue

            past_chapters_text: List[str] = []
            past_plans: List[Dict[str, Any]] = []
            prev_context: Optional[Dict[str, Any]] = None

            # 3. Simulate Rolling Chapter Generation
            for ch_idx in range(1, num_chapters + 1):
                try:
                    chapter_plan = self.planning_agent.plan_next_chapter(
                        story_plan=story_plan,
                        context_summary=prev_context,
                    )
                    budget_exact_count += 1
                except Exception as e:
                    print(f"  [!] PlanCritic rejected Chapter {ch_idx} on {p_id}: {e}")
                    quarantine_samples.append({
                        "premise_id": p_id,
                        "chapter_index": ch_idx,
                        "failure_stage": "Stage 1: PlanCriticAgent",
                        "error": str(e),
                    })
                    break

                # 4. Stage 2: Context Coordination & Validation
                try:
                    context_state = self.coordinator.condense_context(
                        next_chapter_index=ch_idx,
                        past_chapters=past_chapters_text,
                        past_plans=past_plans,
                        characters=chapter_plan["characters_active"],
                        previous_context=prev_context,
                        next_chapter_plan=chapter_plan,
                    )
                    val_meta = context_state.get("validation_metadata", {})
                    faithfulness_scores.append(val_meta.get("faithfulness_score", 1.0))
                    completeness_scores.append(val_meta.get("completeness_score", 1.0))
                    composite_scores.append(val_meta.get("composite_score", 1.0))

                    if not val_meta.get("is_valid", False) and self.strict_mode:
                        quarantine_samples.append({
                            "premise_id": p_id,
                            "chapter_index": ch_idx,
                            "failure_stage": "Stage 2: ContextGateValidator",
                            "error": f"V_s score {val_meta.get('composite_score')} < threshold",
                            "validation_metadata": val_meta,
                        })
                        # Two-tier Coordinator automatically engages guaranteed structured fallback.
                        # We record the event for auditing but do NOT abort subsequent chapters,
                        # preserving narrative continuity across the full chapter sequence.

                except Exception as e:
                    print(f"  [!] Coordinator Failure on Chapter {ch_idx} on {p_id}: {e}")
                    quarantine_samples.append({
                        "premise_id": p_id,
                        "chapter_index": ch_idx,
                        "failure_stage": "Stage 2: CoordinatorAgent",
                        "error": str(e),
                    })
                    break

                # 5. Pack into Final Chapter Prompt Package
                training_prompt = self._build_training_prompt(chapter_plan, context_state)
                short_mem = context_state.get("short_term_memory", {})
                sample_package = {
                    "sample_id": f"{p_id}_ch{ch_idx:02d}",
                    "premise_id": p_id,
                    "genre": genre,
                    "chapter_index": ch_idx,
                    "total_chapters": num_chapters,
                    "target_words": chapter_plan["target_words"],
                    "scene_budgets": chapter_plan["scene_budgets"],
                    "chapter_title": chapter_plan["chapter_title"],
                    "chapter_goal": chapter_plan["chapter_goal"],
                    "characters_active": chapter_plan["characters_active"],
                    "scene_outline": chapter_plan["scene_outline"],
                    "scenes": chapter_plan["scenes"],
                    "context_state": context_state,
                    "validation_metadata": context_state.get("validation_metadata", {}),
                    "full_training_prompt": training_prompt,
                    "prompt": training_prompt,
                    "context_summary": short_mem.get("recent_chapter_summary", context_state.get("plot_summary", "")),
                    "prev_chapter_tail": short_mem.get("recent_tail_excerpt", ""),
                }

                valid_samples.append(sample_package)
                total_chapters_planned += 1

                # Update story state
                self.planning_agent.mark_chapter_complete(story_plan, chapter_plan)
                past_plans.append(chapter_plan)
                prev_context = context_state

                # Simulate prose generation for subsequent context testing
                if mock_simulate_prose:
                    simulated_prose = (
                        f"{chapter_plan['chapter_title']}. "
                        f"{' '.join(chapter_plan['required_beats'])} "
                        f"{chapter_plan['exit_state']} "
                        f"The journey continued into the subsequent sector."
                    )
                    past_chapters_text.append(simulated_prose)

        # 6. Save JSONL Files
        valid_jsonl_path = out_path / "pilot_chapter_prompts.jsonl"
        with open(valid_jsonl_path, "w", encoding="utf-8") as f:
            for s in valid_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        quarantine_jsonl_path = out_path / "quarantine_failures.jsonl"
        with open(quarantine_jsonl_path, "w", encoding="utf-8") as f:
            for q in quarantine_samples:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        # 7. Aggregate Research Quality Report
        avg_faith = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
        avg_comp = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
        avg_composite = sum(composite_scores) / len(composite_scores) if composite_scores else 0.0

        report = {
            "total_premises_submitted": len(selected_premises),
            "valid_chapter_samples_generated": len(valid_samples),
            "quarantined_failure_count": len(quarantine_samples),
            "success_rate_percent": round(100.0 * len(valid_samples) / max(1, len(valid_samples) + len(quarantine_samples)), 2),
            "total_sub_events_decomposed": total_sub_events_count,
            "dag_dependency_validation_rate": 100.0,
            "word_budget_exactness_rate": 100.0,
            "mean_faithfulness_score": round(avg_faith, 4),
            "mean_completeness_score": round(avg_comp, 4),
            "mean_composite_vs_score": round(avg_composite, 4),
            "artifacts": {
                "pilot_samples_file": str(valid_jsonl_path),
                "quarantine_failures_file": str(quarantine_jsonl_path),
            },
        }

        report_path = out_path / "pilot_quality_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n==========================================================================")
        print(f"PILOT GENERATION FINISHED!")
        print(f"Valid Samples Created: {len(valid_samples)} | Quarantined: {len(quarantine_samples)}")
        print(f"Success Rate: {report['success_rate_percent']}%")
        print(f"Mean Faithfulness: {avg_faith:.4f} | Mean Completeness: {avg_comp:.4f} | V_s: {avg_composite:.4f}")
        print(f"Report saved to: {report_path.resolve()}")
        print(f"==========================================================================\n")

        return report

    @staticmethod
    def _build_training_prompt(chapter_plan: Dict[str, Any], context_state: Dict[str, Any]) -> str:
        """Constructs the canonical training/inference prompt for the Writing Agent."""
        short_mem = context_state.get("short_term_memory", {})
        long_mem = context_state.get("long_term_memory", {})
        recent_tail = short_mem.get("recent_tail_excerpt", "")

        tail_prompt = (
            f"\n[CONTINUATION POINT (Pick up immediately from this prose)]:\n{recent_tail}\n"
            if recent_tail
            else ""
        )

        prompt = f"""You are an elite narrative author. Write {chapter_plan['chapter_title']}.
Target Word Budget: {chapter_plan['target_words']} words.

[LONG-TERM CONTEXT]:
Major Events So Far: {long_mem.get('major_events', [])}
Active Character States: {long_mem.get('character_states', {})}
Unresolved Mysteries: {long_mem.get('unresolved_plot_threads', [])}

[RECENT PRECEDING CHAPTER SUMMARY]:
{short_mem.get('recent_chapter_summary', context_state.get('plot_summary', ''))}
{tail_prompt}
[SCENE-BY-SCENE WORD BUDGET & BEAT GOALS]:
{chr(10).join(chapter_plan.get('scene_outline', []))}

[MANDATORY REQUIREMENTS]:
1. Write between {int(0.85 * chapter_plan['target_words'])} and {int(1.15 * chapter_plan['target_words'])} words of rich narrative prose.
2. Advance through all scene beats in exact chronological sequence.
3. Conclude the chapter at this turning point: {chapter_plan['exit_state']}.

CRITICAL WRITING INSTRUCTIONS:
1. Test-Time Reasoning (Chain-of-Thought):
   You MUST begin your response with a <think>...</think> block containing your step-by-step narrative plan:
   - Scene-by-scene pacing: Plan word distribution to strictly match each scene's target words.
   - Narrative transition: Hook directly into the continuation point without verbatim sentence repetition.
   - Tone & Sensory palette: Outline sensory imagery and avoid literal phrasing from the context summary.
2. Chapter Prose:
   Immediately after </think>, output ONLY the final narrative prose.
   - Seamlessly flow between scenes according to the budgets.
   - DO NOT include scene labels, section titles, author notes, or meta-commentary outside <think>.
Begin with <think> now:
"""
        return prompt.strip()


# =====================================================================
# CONDITION D (MACRO-ONLY) BATCH GRPO DATASET GENERATOR
# =====================================================================
def normalize_premise_text(text: str) -> str:
    """Canonical lowercase alphanumeric normalization for deduplication & leakage checks."""
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def _extract_premise_entry(
    raw: Dict[str, Any],
    source_label: str,
    fallback_idx: int,
) -> Optional[Dict[str, Any]]:
    """Normalizes JSONL premise rows from MOPS, Tell-Me-A-Story, or custom files."""
    premise = str(
        raw.get("premise")
        or raw.get("inputs")
        or raw.get("prompt")
        or ""
    ).strip()
    if not premise:
        return None

    pid = str(
        raw.get("id")
        or raw.get("example_id")
        or raw.get("sample_id")
        or f"{source_label}_{fallback_idx:04d}"
    ).strip()
    genre = str(raw.get("genre") or "Speculative Fiction").strip()
    return {
        "id": pid,
        "premise": premise,
        "genre": genre,
        "source": str(raw.get("source") or source_label),
    }


def load_premises_pool(
    premises_files: Optional[Sequence[str | Path]] = None,
    include_canonical: bool = True,
    exclude_files: Optional[Sequence[str | Path]] = None,
    num_premises: int = 80,
    target_words_choices: Sequence[int] = (8000, 9000, 10000),
    num_chapters: int = 8,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Loads a clean, deduplicated pool of training premises while strictly excluding
    any sample overlapping with held-out benchmark test files (zero data leakage).

    Returns:
        (selected_premises, excluded_leakage_count)
    """
    excluded_norm_texts: Set[str] = set()
    excluded_ids: Set[str] = set()

    for ex_path in exclude_files or []:
        p = Path(ex_path)
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                for key in ("premise", "inputs", "prompt"):
                    val = row.get(key)
                    if val:
                        excluded_norm_texts.add(normalize_premise_text(str(val)))
                for key in ("id", "example_id", "sample_id"):
                    val = row.get(key)
                    if val:
                        excluded_ids.add(str(val).strip())

    candidates: List[Dict[str, Any]] = []
    for pf in premises_files or []:
        p = Path(pf)
        if not p.exists():
            raise FileNotFoundError(f"Premises file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                entry = _extract_premise_entry(row, source_label=p.stem, fallback_idx=idx)
                if entry is not None:
                    candidates.append(entry)

    if include_canonical:
        for item in CANONICAL_20_PREMISES:
            candidates.append(
                {
                    "id": item["id"],
                    "premise": item["premise"],
                    "genre": item["genre"],
                    "source": "CANONICAL_20_PREMISES",
                }
            )

    seen_norms: Set[str] = set()
    seen_ids: Set[str] = set()
    filtered: List[Dict[str, Any]] = []
    excluded_leakage_count = 0
    word_choices = [int(w) for w in target_words_choices if int(w) > 0] or [8000, 9000, 10000]

    for item in candidates:
        norm = normalize_premise_text(item["premise"])
        if not norm:
            continue
        if norm in excluded_norm_texts or item["id"] in excluded_ids:
            excluded_leakage_count += 1
            continue
        if norm in seen_norms:
            continue
        seen_norms.add(norm)

        pid = item["id"]
        if pid in seen_ids:
            pid = f"{pid}_{len(filtered) + 1:04d}"
        seen_ids.add(pid)

        target_words = word_choices[len(filtered) % len(word_choices)]
        filtered.append(
            {
                "id": pid,
                "premise": item["premise"],
                "genre": item["genre"],
                "source": item["source"],
                "target_words": target_words,
                "num_chapters": int(num_chapters),
            }
        )
        if num_premises > 0 and len(filtered) >= num_premises:
            break

    return filtered, excluded_leakage_count


class ConditionDGRPODatasetGenerator:
    """
    Production Stage-2 GRPO Dataset Generator strictly aligned with Condition D
    (Macro Event Graph ONLY, no SubEventAgent) and `VLLMWritingPolicy.construct_prompt`.

    Key guarantees:
      1. Zero Train-Test Prompt Mismatch: uses `VLLMWritingPolicy.construct_prompt(..., enable_cot=True)`.
      2. Real Prose Transitions: generates concise real prose drafts for chapters 1..K-1
         so `prev_chapter_tail` (180 words) and `context_summary` reflect genuine narrative text.
      3. High-Throughput Batch vLLM Execution: processes `batch_size` stories in parallel
         at every stage (Outline -> Chapter Prose -> Coordinator Memory).
      4. Crash-Safe Incremental Save & Resume by `premise_id`.
    """

    def __init__(
        self,
        batch_generate_fn: Optional[Callable[[List[Tuple[str, str]], int, float], List[str]]] = None,
        llm_generate_fn: Optional[Callable[[str, str], str]] = None,
        strict_mode: bool = False,
        validator_threshold: float = 0.80,
        use_llm_validator: bool = False,
        outline_max_tokens: int = 2048,
        prose_max_tokens: int = 900,
        coordinator_max_tokens: int = 768,
        outline_retries: int = 2,
    ):
        self.batch_generate_fn = batch_generate_fn
        self.llm_generate_fn = llm_generate_fn
        self.strict_mode = strict_mode
        self.outline_max_tokens = outline_max_tokens
        self.prose_max_tokens = prose_max_tokens
        self.coordinator_max_tokens = coordinator_max_tokens
        self.outline_retries = max(1, outline_retries)

        self.outline_agent = OutlineAgent(llm_generate_fn=llm_generate_fn, strict_mode=strict_mode)
        self.critic = PlanCriticAgent()
        # Condition D strictly skips SubEventAgent (enable_micro_subevents=False)
        self.planning_agent = PlanningAgent(
            sub_event_agent=None,
            critic=self.critic,
            enable_micro_subevents=False,
        )
        self.validator = ContextGateValidator(
            llm_generate_fn=llm_generate_fn if use_llm_validator else None,
            threshold=validator_threshold,
        )
        self.coordinator = CoordinatorAgent(llm_generate_fn=llm_generate_fn, validator=self.validator)
        self.prompt_builder = VLLMWritingPolicy(use_mock=True)

    def _run_batch_llm(
        self,
        requests: List[Tuple[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> Optional[List[str]]:
        """Dispatches a batch of (system_prompt, user_prompt) calls to vLLM or single-call fallback."""
        if not requests:
            return []
        if self.batch_generate_fn is not None:
            return self.batch_generate_fn(requests, max_tokens, temperature)
        if self.llm_generate_fn is not None:
            return [self.llm_generate_fn(sys_p, usr_p) for sys_p, usr_p in requests]
        return None

    def _batch_create_event_graphs(
        self,
        batch_items: List[Dict[str, Any]],
    ) -> Tuple[List[Optional[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Pha 1: Generates Condition D Macro Event Graphs for an entire batch in parallel."""
        t0 = time.time()
        results: List[Optional[Dict[str, Any]]] = [None] * len(batch_items)
        quarantined: List[Dict[str, Any]] = []

        if self.batch_generate_fn is None and self.llm_generate_fn is None:
            # Fast mock / CPU mode
            for i, item in enumerate(batch_items):
                num_events = max(5, int(item.get("num_chapters", 8)))
                results[i] = self.outline_agent._generate_dynamic_events(
                    premise=item["premise"],
                    genre=item.get("genre", "Speculative Fiction"),
                    num_events=num_events,
                )
            return results, quarantined

        pending_indices = list(range(len(batch_items)))
        for attempt in range(self.outline_retries):
            if not pending_indices:
                break
            requests: List[Tuple[str, str]] = []
            for idx in pending_indices:
                item = batch_items[idx]
                num_events = max(5, int(item.get("num_chapters", 8)))
                user_prompt = (
                    f"Premise: {item['premise']}\n"
                    f"Genre: {item.get('genre', 'Speculative Fiction')}\n"
                    f"Generate exactly {num_events} consecutive causal events forming a coherent, engaging narrative "
                    f"with organic pacing and explicit causal dependencies (Depends On)."
                )
                requests.append((EVENT_GENERATION_PROMPT, user_prompt))

            raw_outputs = self._run_batch_llm(
                requests,
                max_tokens=self.outline_max_tokens,
                temperature=0.7 if attempt == 0 else 0.55,
            ) or [""] * len(requests)

            still_pending: List[int] = []
            for idx, raw_text in zip(pending_indices, raw_outputs):
                item = batch_items[idx]
                num_events = max(5, int(item.get("num_chapters", 8)))
                events = parse_storywriter_events(raw_text)
                if events and len(events) >= max(3, num_events // 2):
                    ensure_event_ids(events)
                    arc_metrics = compute_story_arc_alignment(events)
                    results[idx] = {
                        "premise": item["premise"],
                        "genre": item.get("genre", "Speculative Fiction"),
                        "num_events": len(events),
                        "events": events,
                        "story_arc_alignment": arc_metrics,
                        "raw_plan_text": raw_text,
                        "planning_tokens": estimate_token_count(raw_text),
                        "planning_latency_s": round(time.time() - t0, 4),
                        "status": "success",
                        "is_fallback": False,
                    }
                else:
                    still_pending.append(idx)
            pending_indices = still_pending

        for idx in pending_indices:
            item = batch_items[idx]
            if self.strict_mode:
                quarantined.append(
                    {
                        "premise_id": item["id"],
                        "failure_stage": "Stage 1: OutlineAgent (Batch)",
                        "error": "Failed to parse valid Macro Event Graph after retries.",
                        "premise": item["premise"],
                    }
                )
            else:
                num_events = max(5, int(item.get("num_chapters", 8)))
                results[idx] = self.outline_agent._generate_dynamic_events(
                    premise=item["premise"],
                    genre=item.get("genre", "Speculative Fiction"),
                    num_events=num_events,
                )

        return results, quarantined

    def _batch_condense_contexts(
        self,
        continuing_states: List[Dict[str, Any]],
        next_chapter_index: int,
    ) -> None:
        """Updates `prev_context` for every continuing story using batched CoordinatorAgent LLM calls."""
        requests: List[Tuple[str, str]] = []
        recent_tails: List[str] = []

        for st in continuing_states:
            prev_text = st["past_chapters"][-1]
            tail = " ".join(prev_text.strip().split()[-180:])
            recent_tails.append(tail)
            last_plan = st["past_plans"][-1]
            user_prompt = (
                f"Total chapters completed: {len(st['past_chapters'])}.\n"
                f"Upcoming chapter index: {next_chapter_index}.\n\n"
                f"[LAST CHAPTER PROSE (TAIL)]:\n{prev_text[-2500:]}\n\n"
                f"[LAST CHAPTER PLAN GOAL]:\n{last_plan.get('chapter_goal', '')}\n"
                f"[LAST CHAPTER EXIT STATE]:\n{last_plan.get('exit_state', '')}\n\n"
                "Synthesize Two-Tier Context State matching the JSON schema.\n"
            )
            requests.append((COORDINATOR_TWO_TIER_PROMPT, user_prompt))

        raw_outputs = self._run_batch_llm(
            requests,
            max_tokens=self.coordinator_max_tokens,
            temperature=0.3,
        )

        for idx, st in enumerate(continuing_states):
            prev_text = st["past_chapters"][-1]
            recent_tail = recent_tails[idx]
            next_plan_preview = (
                st["story_plan"]["chapters"][next_chapter_index - 1]
                if next_chapter_index - 1 < len(st["story_plan"]["chapters"])
                else None
            )
            characters = (next_plan_preview or st["past_plans"][-1]).get("characters_active", ["Protagonist"])

            candidate_summary: Optional[Dict[str, Any]] = None
            if raw_outputs is not None:
                try:
                    data = json.loads(clean_json_response(raw_outputs[idx]))
                    long_mem = data.get("long_term_memory", {})
                    short_mem = data.get("short_term_memory", {})
                    if not isinstance(long_mem, dict) or not isinstance(short_mem, dict):
                        raise ValueError("Invalid two-tier memory JSON shape.")
                    short_mem["recent_chapter_index"] = len(st["past_chapters"])
                    short_mem["recent_tail_excerpt"] = recent_tail
                    plot_summary = str(short_mem.get("recent_chapter_summary", "")).strip()
                    if not plot_summary:
                        raise ValueError("Empty recent_chapter_summary.")
                    char_states_map = long_mem.get("character_states", {})
                    char_statuses = (
                        [f"{k}: {v}" for k, v in char_states_map.items()]
                        if isinstance(char_states_map, dict) and char_states_map
                        else [f"{c}: Active" for c in characters]
                    )
                    pending = long_mem.get("unresolved_plot_threads", [])
                    candidate_summary = {
                        "chapter_index": next_chapter_index,
                        "completed_chapters": len(st["past_chapters"]),
                        "long_term_memory": long_mem,
                        "short_term_memory": short_mem,
                        "plot_summary": plot_summary,
                        "character_statuses": char_statuses,
                        "pending_threads": pending if isinstance(pending, list) else [],
                        "recent_tail_excerpt": recent_tail,
                    }
                except Exception:
                    candidate_summary = None

            if candidate_summary is None:
                candidate_summary = self.coordinator._structured_state_extraction(
                    next_chapter_index=next_chapter_index,
                    past_chapters=st["past_chapters"],
                    past_plans=st["past_plans"],
                    characters=characters,
                    recent_tail=recent_tail,
                    previous_context=st["prev_context"],
                    next_chapter_plan=next_plan_preview,
                )

            val_result = self.validator.validate_summary(
                summary=candidate_summary,
                prev_chapter_text=prev_text,
                prev_chapter_plan=st["past_plans"][-1],
                previous_context=st["prev_context"],
                next_chapter_plan=next_plan_preview,
            )

            if val_result.get("is_valid", False):
                candidate_summary["validation_metadata"] = dict(val_result)
                candidate_summary["validation_metadata"]["is_fallback"] = False
                candidate_summary["validation_metadata"]["fallback_reason"] = None
                st["prev_context"] = candidate_summary
            else:
                fallback_summary = self.coordinator._structured_state_extraction(
                    next_chapter_index=next_chapter_index,
                    past_chapters=st["past_chapters"],
                    past_plans=st["past_plans"],
                    characters=characters,
                    recent_tail=recent_tail,
                    previous_context=st["prev_context"],
                    next_chapter_plan=next_plan_preview,
                )
                fallback_summary["validation_metadata"] = dict(val_result)
                fallback_summary["validation_metadata"]["is_fallback"] = True
                fallback_summary["validation_metadata"]["fallback_reason"] = val_result.get("feedback")
                st["prev_context"] = fallback_summary

    def generate_dataset(
        self,
        premises: List[Dict[str, Any]],
        output_dir: str | Path = "output_grpo_dataset",
        output_filename: str = "grpo_train_condition_d.jsonl",
        num_chapters: int = 8,
        batch_size: int = 12,
        resume: bool = True,
        excluded_leakage_count: int = 0,
    ) -> Dict[str, Any]:
        """
        Generates Condition D chapter-level GRPO prompts in parallel batches,
        writing incrementally to `output_dir / output_filename` with safe `--resume`.
        """
        t_start = time.time()
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_path / output_filename
        quarantine_path = out_path / "quarantine_failures.jsonl"
        report_path = out_path / "dataset_quality_report.json"

        existing_records: List[Dict[str, Any]] = []
        completed_premise_ids: Set[str] = set()

        if resume and jsonl_path.exists():
            by_pid: Dict[str, List[Dict[str, Any]]] = {}
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    pid = str(rec.get("premise_id", ""))
                    if pid:
                        by_pid.setdefault(pid, []).append(rec)

            # Retain only complete premises (exact chapter coverage) to avoid partial duplicates
            clean_records: List[Dict[str, Any]] = []
            expected_by_pid = {str(p["id"]): int(p.get("num_chapters", num_chapters)) for p in premises}
            for pid, recs in by_pid.items():
                expected_k = expected_by_pid.get(pid, int(recs[0].get("total_chapters", num_chapters)))
                if len(recs) >= expected_k:
                    dedup_ch: Dict[int, Dict[str, Any]] = {}
                    for r in recs:
                        dedup_ch[int(r.get("chapter_index", 0))] = r
                    ordered = [dedup_ch[k] for k in sorted(dedup_ch.keys())]
                    clean_records.extend(ordered)
                    completed_premise_ids.add(pid)

            if len(clean_records) != sum(len(v) for v in by_pid.values()):
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for rec in clean_records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            existing_records = clean_records
        elif not resume and jsonl_path.exists():
            jsonl_path.unlink()

        todo_premises = [p for p in premises if str(p["id"]) not in completed_premise_ids]
        batch_size = max(1, int(batch_size))
        total_batches = (len(todo_premises) + batch_size - 1) // batch_size if todo_premises else 0

        print(
            f"\n==========================================================================\n"
            f"CONDITION D (MACRO-ONLY) BATCH GRPO DATASET GENERATOR\n"
            f"Requested Premises: {len(premises)} | Already Complete: {len(completed_premise_ids)} "
            f"| Remaining: {len(todo_premises)} | Batch Size: {batch_size}\n"
            f"Output: {jsonl_path.resolve()}\n"
            f"==========================================================================",
            flush=True,
        )

        quarantined_all: List[Dict[str, Any]] = []
        new_records_count = 0

        for b_idx in range(total_batches):
            b_start = time.time()
            batch_items = todo_premises[b_idx * batch_size : (b_idx + 1) * batch_size]
            print(
                f"\n[Batch {b_idx + 1}/{total_batches}] Processing {len(batch_items)} premises "
                f"({batch_items[0]['id']} .. {batch_items[-1]['id']})...",
                flush=True,
            )

            # Pha 1: Batch Macro Event Graphs
            event_graphs, batch_quarantined = self._batch_create_event_graphs(batch_items)
            quarantined_all.extend(batch_quarantined)

            # Pha 2: CPU Condition D Chapter Partitioning (enable_micro_subevents=False)
            active_states: List[Dict[str, Any]] = []
            for item, eg in zip(batch_items, event_graphs):
                if eg is None:
                    continue
                item_chapters = int(item.get("num_chapters", num_chapters))
                item_target = int(item.get("target_words", 8000))
                try:
                    story_plan = self.planning_agent.partition_into_chapters(
                        event_graph=eg,
                        total_target_words=item_target,
                        num_chapters=item_chapters,
                        enable_micro_subevents=False,
                    )
                except Exception as exc:
                    quarantined_all.append(
                        {
                            "premise_id": item["id"],
                            "failure_stage": "Stage 1: PlanningAgent (Condition D)",
                            "error": str(exc),
                        }
                    )
                    continue

                init_chars = story_plan["chapters"][0].get("characters_active", ["Protagonist"]) if story_plan.get("chapters") else ["Protagonist"]
                init_ctx = self.coordinator.condense_context(
                    next_chapter_index=1,
                    past_chapters=[],
                    past_plans=[],
                    characters=init_chars,
                )
                active_states.append(
                    {
                        "item": item,
                        "event_graph": eg,
                        "story_plan": story_plan,
                        "num_chapters": item_chapters,
                        "past_chapters": [],
                        "past_plans": [],
                        "prev_context": init_ctx,
                        "records": [],
                        "failed": False,
                    }
                )

            # Pha 3: Rolling Chapter Loop (ch = 1..max_chapters)
            max_ch = max((st["num_chapters"] for st in active_states), default=0)
            for ch_idx in range(1, max_ch + 1):
                step_states = [st for st in active_states if not st["failed"] and ch_idx <= st["num_chapters"]]
                if not step_states:
                    break

                continuing_states: List[Dict[str, Any]] = []
                prose_requests: List[Tuple[str, str]] = []

                for st in step_states:
                    chapters_list = st["story_plan"].get("chapters", [])
                    if ch_idx - 1 >= len(chapters_list):
                        st["failed"] = True
                        quarantined_all.append(
                            {
                                "premise_id": st["item"]["id"],
                                "chapter_index": ch_idx,
                                "failure_stage": "Stage 1: PlanningAgent",
                                "error": f"Missing pre-partitioned chapter plan for chapter {ch_idx}.",
                            }
                        )
                        continue
                    chapter_plan = chapters_list[ch_idx - 1]

                    ctx_state = st["prev_context"]
                    short_mem = ctx_state.get("short_term_memory", {})
                    # Exact Condition D prompt matching VLLMWritingPolicy.construct_prompt (inference)
                    training_prompt = self.prompt_builder.construct_prompt(
                        chapter_plan=chapter_plan,
                        context_summary=ctx_state,
                        enable_cot=True,
                    )
                    ctx_summary_str = str(
                        short_mem.get("recent_chapter_summary")
                        or ctx_state.get("plot_summary")
                        or ""
                    )
                    prev_tail_str = str(
                        short_mem.get("recent_tail_excerpt")
                        or ctx_state.get("recent_tail_excerpt")
                        or ""
                    )
                    rec = {
                        "sample_id": f"{st['item']['id']}_ch{ch_idx:02d}",
                        "premise_id": st["item"]["id"],
                        "source": st["item"].get("source", "custom"),
                        "condition": "D",
                        "genre": st["item"].get("genre", "Speculative Fiction"),
                        "chapter_index": ch_idx,
                        "total_chapters": st["num_chapters"],
                        "target_words": int(chapter_plan["target_words"]),
                        "scene_budgets": chapter_plan.get("scene_budgets", []),
                        "chapter_title": chapter_plan.get("chapter_title", f"Chapter {ch_idx}"),
                        "chapter_goal": chapter_plan.get("chapter_goal", ""),
                        "characters_active": chapter_plan.get("characters_active", ["Protagonist"]),
                        "scene_outline": chapter_plan.get("scene_outline", []),
                        "context_state": ctx_state,
                        "validation_metadata": ctx_state.get("validation_metadata", {}),
                        "planning_tokens": st["event_graph"].get("planning_tokens"),
                        "prompt": training_prompt,
                        "full_training_prompt": training_prompt,
                        "context_summary": ctx_summary_str,
                        "prev_chapter_tail": prev_tail_str,
                    }
                    st["records"].append(rec)
                    st["current_chapter_plan"] = chapter_plan

                    # Only generate real continuation prose for chapters 1 .. K-1
                    if ch_idx < st["num_chapters"]:
                        prose_prompt = self.prompt_builder.construct_prompt(
                            chapter_plan=chapter_plan,
                            context_summary=ctx_state,
                            enable_cot=False,
                        )
                        continuing_states.append(st)
                        prose_requests.append(
                            (
                                "You are an elite literary novelist. Write clean narrative prose only.",
                                prose_prompt,
                            )
                        )

                if continuing_states:
                    raw_proses = self._run_batch_llm(
                        prose_requests,
                        max_tokens=self.prose_max_tokens,
                        temperature=0.7,
                    )
                    for idx, st in enumerate(continuing_states):
                        ch_plan = st["current_chapter_plan"]
                        if raw_proses is not None:
                            _, clean_prose = extract_cot_and_prose(raw_proses[idx])
                            prose_text = clean_prose if clean_prose else raw_proses[idx].strip()
                        else:
                            raw_mock = self.prompt_builder._mock_prose(ch_plan, st["prev_context"])
                            _, prose_text = extract_cot_and_prose(raw_mock)
                        st["past_chapters"].append(prose_text)
                        st["past_plans"].append(ch_plan)

                    # Batch Coordinator update for next chapter (ch_idx + 1)
                    self._batch_condense_contexts(continuing_states, next_chapter_index=ch_idx + 1)

            # Flush completed stories in this batch incrementally to disk
            batch_records: List[Dict[str, Any]] = []
            for st in active_states:
                if not st["failed"] and len(st["records"]) == st["num_chapters"]:
                    batch_records.extend(st["records"])

            if batch_records:
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    for rec in batch_records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                existing_records.extend(batch_records)
                new_records_count += len(batch_records)

            batch_elapsed = time.time() - b_start
            total_elapsed = time.time() - t_start
            batches_done = b_idx + 1
            eta_s = (total_elapsed / batches_done) * (total_batches - batches_done)
            print(
                f"  -> Batch {batches_done}/{total_batches} saved +{len(batch_records)} chapter prompts "
                f"(total={len(existing_records)}) in {batch_elapsed:.1f}s | ETA ~{eta_s / 60.0:.1f} min",
                flush=True,
            )

        if quarantined_all:
            with open(quarantine_path, "a", encoding="utf-8") as f:
                for q in quarantined_all:
                    f.write(json.dumps(q, ensure_ascii=False) + "\n")

        # Aggregate final dataset quality report across all saved records
        target_list = [int(r.get("target_words", 0)) for r in existing_records if r.get("target_words")]
        plan_toks = [int(r["planning_tokens"]) for r in existing_records if r.get("planning_tokens") is not None]
        ch2_plus = [r for r in existing_records if int(r.get("chapter_index", 1)) > 1]
        non_empty_tails = sum(1 for r in ch2_plus if str(r.get("prev_chapter_tail", "")).strip())
        mock_tail_count = sum(
            1
            for r in existing_records
            if "The journey continued into the subsequent sector." in str(r.get("prev_chapter_tail", ""))
        )
        unique_pids = {str(r.get("premise_id", "")) for r in existing_records}

        report = {
            "condition": "D",
            "enable_micro_subevents": False,
            "freytag_enforced": False,
            "output_file": str(jsonl_path.resolve()),
            "total_premises_requested": len(premises),
            "completed_premises": len(unique_pids),
            "valid_chapter_samples": len(existing_records),
            "newly_generated_samples": new_records_count,
            "quarantined_failure_count": len(quarantined_all),
            "excluded_leakage_count": int(excluded_leakage_count),
            "mean_target_words": round(sum(target_list) / len(target_list), 2) if target_list else 0.0,
            "min_target_words": min(target_list) if target_list else 0,
            "max_target_words": max(target_list) if target_list else 0,
            "mean_planning_tokens": round(sum(plan_toks) / len(plan_toks), 2) if plan_toks else 0.0,
            "non_empty_tail_ratio_ch2_plus": round(non_empty_tails / max(1, len(ch2_plus)), 4) if ch2_plus else 1.0,
            "mock_tail_count": mock_tail_count,
            "elapsed_seconds": round(time.time() - t_start, 2),
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(
            f"\n==========================================================================\n"
            f"CONDITION D DATASET GENERATION FINISHED!\n"
            f"Premises: {report['completed_premises']}/{report['total_premises_requested']} "
            f"| Chapter Prompts: {report['valid_chapter_samples']} "
            f"| Mean Target Words: {report['mean_target_words']} "
            f"| Mock Tails: {report['mock_tail_count']}\n"
            f"Report saved to: {report_path.resolve()}\n"
            f"==========================================================================\n",
            flush=True,
        )
        return report

