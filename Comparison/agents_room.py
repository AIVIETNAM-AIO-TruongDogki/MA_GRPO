"""
AGENTS' ROOM: Multi-Agent Collaboration for Long-Form Narrative Fiction Generation
(Google DeepMind, Huot et al., 2024)

Faithful 9-Agent Architecture + Central Orchestrator Scratchpad:
  1. Planning Sub-Agents (4 Roles writing sequentially to Shared Scratchpad):
     - [CONFLICT] Agent: Defines central dramatic conflict, inciting incident, and stakes.
     - [CHARACTER] Agent: Designs protagonist, mythological figure, voice, and psychology.
     - [SETTING] Agent: Establishes sensory world-building, temporal contrast, and atmosphere.
     - [PLOT] Agent: Synthesizes Scratchpad into a 5-Act Freytag's Pyramid outline with word budgets.
  2. Writing Sub-Agents (5 Sequential Roles drafting acts from Scratchpad + Rolling Story Context):
     - Act I: Exposition (18% word budget)
     - Act II: Rising Action (25% word budget)
     - Act III: Climax (25% word budget)
     - Act IV: Falling Action (17% word budget)
     - Act V: Resolution (15% word budget)
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import save_model_result_csv
from .qwen_backend import QwenAgentBackend, get_peak_vram_gb, reset_vram_stats


def run_agents_room_single(prompt: str, target_length: int, backend: QwenAgentBackend) -> str:
    """Execute the full 9-Agent AGENTS' ROOM pipeline for a single story prompt."""
    scratchpad: Dict[str, str] = {"PREMISE": prompt}

    # =========================================================================
    # PHASE 1: 4 PLANNING SUB-AGENTS -> SHARED SCRATCHPAD
    # =========================================================================
    # 1. [CONFLICT] Sub-Agent
    conflict_sys = (
        "You are the [CONFLICT] Planning Agent in a literary writers' room. "
        "Define the core inciting incident, internal vs. external conflicts, and dramatic stakes."
    )
    conflict_prompt = (
        f"Story Premise:\n{prompt}\n\n"
        "Write a concise [CONFLICT] specification (100-150 words) detailing the inciting incident, "
        "the central tension between the mortal and mythological worlds, and the climax resolution."
    )
    scratchpad["CONFLICT"] = backend.generate_chat(
        conflict_prompt, system_prompt=conflict_sys, max_new_tokens=250, temperature=0.6
    )

    # 2. [CHARACTER] Sub-Agent
    char_sys = (
        "You are the [CHARACTER] Planning Agent in a literary writers' room. "
        "Create vivid character sheets with distinct voices, motivations, flaws, and arcs."
    )
    char_prompt = (
        f"Story Premise:\n{prompt}\n\n"
        f"[CONFLICT] Notes:\n{scratchpad['CONFLICT']}\n\n"
        "Write a concise [CHARACTER] specification (120-160 words) profiling the main characters, "
        "their relationship dynamic, and how they transform by the end of the story."
    )
    scratchpad["CHARACTER"] = backend.generate_chat(
        char_prompt, system_prompt=char_sys, max_new_tokens=260, temperature=0.6
    )

    # 3. [SETTING] Sub-Agent
    setting_sys = (
        "You are the [SETTING] Planning Agent in a literary writers' room. "
        "Establish sensory world-building, atmosphere, and spatial-temporal anchors."
    )
    setting_prompt = (
        f"Story Premise:\n{prompt}\n\n"
        f"[CHARACTER] Notes:\n{scratchpad['CHARACTER']}\n\n"
        "Write a concise [SETTING] specification (100-140 words) detailing key locations, lighting, "
        "sounds, textures, and supernatural cues."
    )
    scratchpad["SETTING"] = backend.generate_chat(
        setting_prompt, system_prompt=setting_sys, max_new_tokens=240, temperature=0.6
    )

    # 4. [PLOT] Sub-Agent (Freytag's Pyramid 5-Act Blueprint)
    act_weights = [
        ("Act I: Exposition", 0.18, "Establish the setting, protagonist's life, and initial discovery."),
        ("Act II: Rising Action", 0.25, "Trigger the mythological encounter, deepen dialogue, and escalate complications."),
        ("Act III: Climax", 0.25, "Reach the peak revelation, confrontation, or divine turning point."),
        ("Act IV: Falling Action", 0.17, "Unfold the immediate aftermath, emotional fallout, or transformation."),
        ("Act V: Resolution", 0.15, "Provide poignant closure and fulfill the final requirement of the prompt."),
    ]
    act_budgets = [max(140, int(round(target_length * w))) for _, w, _ in act_weights]
    # Adjust last act so sum equals exact target_length
    act_budgets[-1] = max(120, target_length - sum(act_budgets[:-1]))

    plot_sys = (
        "You are the [PLOT] Orchestrator Agent in a literary writers' room. "
        "Synthesize the Scratchpad into a cohesive 5-Act Freytag's Pyramid outline."
    )
    budget_desc = "\n".join(
        f"- {name} (~{b} words): {desc}" for (name, _, desc), b in zip(act_weights, act_budgets)
    )
    plot_prompt = (
        f"Story Premise:\n{prompt}\n\n"
        f"[CONFLICT]: {scratchpad['CONFLICT']}\n"
        f"[CHARACTER]: {scratchpad['CHARACTER']}\n"
        f"[SETTING]: {scratchpad['SETTING']}\n\n"
        f"Required 5-Act Structure (Total: {target_length} words):\n{budget_desc}\n\n"
        "Provide a clear 5-Act beat sheet (2-3 concrete scene beats per Act). Do not write the story yet."
    )
    scratchpad["PLOT"] = backend.generate_chat(
        plot_prompt, system_prompt=plot_sys, max_new_tokens=400, temperature=0.6
    )

    # =========================================================================
    # PHASE 2: 5 SEQUENTIAL WRITING SUB-AGENTS (FREYTAG'S PYRAMID)
    # =========================================================================
    drafted_acts: List[str] = []
    words_written_so_far = 0

    for idx, ((act_name, _, act_goal), planned_budget) in enumerate(zip(act_weights, act_budgets)):
        # Dynamically calibrate remaining word budget so final length converges to target_length
        remaining_acts = len(act_weights) - idx
        remaining_target = max(150, target_length - words_written_so_far)
        if idx == len(act_weights) - 1:
            current_budget = remaining_target
        else:
            current_budget = int(0.6 * planned_budget + 0.4 * (remaining_target / remaining_acts))
            current_budget = max(140, current_budget)

        writer_sys = (
            f"You are the specialist Writing Agent for {act_name} in the AGENTS' ROOM architecture. "
            f"Your dramatic objective: {act_goal} "
            f"Write pure, immersive literary prose of approximately {current_budget} words. "
            "Do NOT output act titles, section headers, or meta-commentary."
        )

        # Rolling tail from previous acts for smooth transition
        prev_context = ""
        if drafted_acts:
            joined_prev = "\n\n".join(drafted_acts)
            tail_words = joined_prev.split()[-220:]
            prev_context = (
                f"\n[STORY WRITTEN SO FAR ({words_written_so_far} words) - ENDING EXCERPT]:\n"
                f"...{' '.join(tail_words)}\n"
            )

        writer_prompt = (
            f"[SHARED SCRATCHPAD]\n"
            f"- Premise: {prompt}\n"
            f"- Character & Setting: {scratchpad['CHARACTER'][:350]} | {scratchpad['SETTING'][:300]}\n"
            f"- 5-Act Plot Blueprint:\n{scratchpad['PLOT']}\n"
            f"{prev_context}\n"
            f"[WRITING TASK FOR {act_name.upper()}]\n"
            f"- Target Length for this section: strictly around {current_budget} words.\n"
            "- Continue seamlessly from the previous scene without repeating earlier events.\n"
            "- Output ONLY the story paragraphs:"
        )

        max_toks = min(1400, max(220, int(current_budget * 1.45)))
        act_prose = backend.generate_chat(
            writer_prompt,
            system_prompt=writer_sys,
            max_new_tokens=max_toks,
            temperature=0.72,
            top_p=0.92,
            repetition_penalty=1.06,
        )
        drafted_acts.append(act_prose.strip())
        words_written_so_far += len(act_prose.split())

    return "\n\n".join(drafted_acts)


def run_agents_room_benchmark(
    prompts: List[Dict[str, Any]],
    output_csv: Path,
    backend: Optional[QwenAgentBackend] = None,
):
    """Run AGENTS' ROOM across all benchmark prompts and save CSV."""
    own_backend = False
    if backend is None:
        backend = QwenAgentBackend("Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True)
        backend.load()
        own_backend = True
    else:
        backend.load()

    print("\n" + "=" * 85, flush=True)
    print("RUNNING MODEL [1/8]: AGENTS' ROOM (9-Agent Scratchpad + Freytag's Pyramid)", flush=True)
    print("=" * 85, flush=True)

    records = []
    for idx, item in enumerate(prompts, 1):
        p_id = item["id"]
        p_text = item["prompt"]
        t_len = int(item["target_length"])
        print(f" -> [AGENTS' ROOM] [{idx}/{len(prompts)}] {p_id} (Target: {t_len} words)...", flush=True)

        reset_vram_stats()
        t0 = time.time()
        raw_out = run_agents_room_single(p_text, t_len, backend)
        elapsed = round(max(0.01, time.time() - t0), 2)
        vram = get_peak_vram_gb()

        actual_w = len(raw_out.split())
        print(f"    Done in {elapsed}s | Words: {actual_w}/{t_len} | Peak VRAM: {vram} GB", flush=True)

        records.append({
            "Model": "AGENTS' ROOM",
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

