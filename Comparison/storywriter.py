"""
STORYWRITER: A Multi-Agent Framework for Long Story Generation with Event Graphs
(THU-KEG, 2025 - Faithful implementation of `StoryWriter/agent_try.py`)

Architecture aligned with `StoryWriter/agent_try.py`:
  1. Event Graph Outline & Critic Loop (`event_model` + `event_rerank_model`):
     - Generates structured events: `Event n: {EVENT} Setting:{SETTING} Character:{CHARACTER} Action:{ACTION} Conflict:{CONFLICT} Plot Twist:{PLOT_TWIST}`
     - Critic checks structural coherence and triggers regeneration if needed.
  2. Sub-Event Divider Agent (`sub_events`):
     - Divides each Event into 3 concrete sub-events (`Event 1.1, Event 1.2, Event 1.3`).
  3. Chapter Disperser Agent (`disperse_events` - Non-Linear Narration):
     - Allocates sub-events into sequential chapters with balanced word budgets.
  4. Chapter Writing + Story Critic + ReIO (`MessageRedact` 10% Condensation):
     - Writes chapters adhering to the sub-event plan while compressing prior chapters
       to 10% length (`MessageRedact.apply_transform` prompt from `StoryWriter/agent_try.py` line 87).
"""

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import save_model_result_csv
from .qwen_backend import QwenAgentBackend, get_peak_vram_gb, reset_vram_stats


# Exact condensation prompt from StoryWriter/agent_try.py line 87
STORYWRITER_REIO_CONDENSE_PROMPT = (
    "Please condense the following story to 10% percent of its original length, "
    "retaining the key plot points, character conflicts, and the twist or resolution at the end. "
    "Omit lengthy descriptions of the setting and minor dialogues, ensuring the condensed version "
    "still clearly conveys the core progression of the story."
)


def run_storywriter_single(prompt: str, target_length: int, backend: QwenAgentBackend) -> str:
    """Execute the full StoryWriter multi-agent pipeline (Event Graph -> Sub-events -> Chapters + ReIO)."""
    # =========================================================================
    # STEP 1: EVENT GRAPH GENERATION (`event_model` + `event_rerank_model`)
    # =========================================================================
    event_sys = (
        "Generate an event outline based on the premise. Here is an example:\n"
        "Event 1: {Title}\nSetting: {Setting}\nCharacter: {Character}\n"
        "Action: {Action}\nConflict: {Conflict}\nPlot Twist: {Plot Twist}\n"
    )
    event_user = (
        f"Premise: {prompt}\n\n"
        "Generate a 3-Event structured outline covering the beginning, middle, and ending of this premise."
    )
    event_outline = backend.generate_chat(
        event_user, system_prompt=event_sys, max_new_tokens=450, temperature=0.65
    )

    # Critic verification (`event_rerank_model`)
    rerank_sys = (
        "You are a story outline critic. Check if the events structure is coherent, faithful to the premise, "
        "and makes logical sense. If coherent, reply 'TERMINATE'. Otherwise, provide a revised 3-Event outline."
    )
    rerank_resp = backend.generate_chat(
        f"Premise:\n{prompt}\n\nGenerated Events:\n{event_outline}",
        system_prompt=rerank_sys,
        max_new_tokens=400,
        temperature=0.3,
    )
    if "TERMINATE" not in rerank_resp.upper() and "Event 1" in rerank_resp:
        event_outline = rerank_resp

    # =========================================================================
    # STEP 2: SUB-EVENT DECOMPOSITION (`sub_events` agent)
    # =========================================================================
    sub_events_sys = "You are an events divider. Your task is to divide each provided event into 3 concrete sub-events."
    sub_events_msg = (
        "Please review the events and divide them into 3 sub-events for each event in the form of:\n"
        "Event 1:\n- Event 1.1: ...\n- Event 1.2: ...\n- Event 1.3: ...\n"
        "Event 2:\n- Event 2.1: ...\n- Event 2.2: ...\n- Event 2.3: ...\n"
        "Event 3:\n- Event 3.1: ...\n- Event 3.2: ...\n- Event 3.3: ...\n\n"
        f"Events:\n{event_outline}"
    )
    sub_events_plan = backend.generate_chat(
        sub_events_msg, system_prompt=sub_events_sys, max_new_tokens=500, temperature=0.5
    )

    # =========================================================================
    # STEP 3: CHAPTER DISPERSION (`disperse_events` agent)
    # =========================================================================
    num_chapters = 3 if target_length <= 1450 else 4
    disperse_sys = (
        "I have a series of events that have been broken down into smaller sub-events. "
        "Your task is to distribute these sub-events across different chapters so the narrative flow "
        "remains coherent, respects logical relationships, and builds suspense."
    )
    disperse_msg = (
        f"Distribute the following sub-events across {num_chapters} sequential chapters:\n"
        f"{sub_events_plan}\n\n"
        "Reply in this form:\n"
        "Chapter 1: [Title] | Sub-events: Event 1.1, Event 1.2...\n"
        "Chapter 2: [Title] | Sub-events: ..."
    )
    chapter_plan = backend.generate_chat(
        disperse_msg, system_prompt=disperse_sys, max_new_tokens=350, temperature=0.5
    )

    # =========================================================================
    # STEP 4: CHAPTER WRITING + REIO (`MessageRedact` 10% CONDENSATION)
    # =========================================================================
    chapters_prose: List[str] = []
    condensed_history: List[str] = []
    words_written = 0

    for ch_idx in range(1, num_chapters + 1):
        remaining_chapters = num_chapters - ch_idx + 1
        remaining_words = max(180, target_length - words_written)
        ch_budget = max(180, int(round(remaining_words / remaining_chapters)))

        writer_sys = (
            "You are the StoryWriter Chapter Author (`story_writer_model`) guided by `story_critic`. "
            "Focus on the relationships between events, and do not let the story deviate from the outlines. "
            f"Write Chapter {ch_idx} in vivid, immersive literary prose of approximately {ch_budget} words. "
            "Output ONLY the narrative story text without chapter headers or meta-notes."
        )

        reio_block = ""
        if condensed_history:
            reio_block = (
                "\n[CONDENSED PRIOR CHAPTERS (ReIO 10% Memory)]:\n"
                + "\n".join(condensed_history)
                + "\n"
            )
            if chapters_prose:
                last_tail = " ".join(chapters_prose[-1].split()[-120:])
                reio_block += f"\n[IMMEDIATE PRECEDING PARAGRAPH]:\n...{last_tail}\n"

        writer_prompt = (
            f"Original Premise:\n{prompt}\n\n"
            f"Sub-Event Breakdown:\n{sub_events_plan[:700]}\n\n"
            f"Chapter Allocation Plan:\n{chapter_plan}\n"
            f"{reio_block}\n"
            f"Now write the complete narrative prose for Chapter {ch_idx} of {num_chapters} "
            f"(Target word count: ~{ch_budget} words). Begin directly with the story prose:"
        )

        max_toks = min(1450, max(260, int(ch_budget * 1.42)))
        ch_text = backend.generate_chat(
            writer_prompt,
            system_prompt=writer_sys,
            max_new_tokens=max_toks,
            temperature=0.72,
            top_p=0.92,
            repetition_penalty=1.06,
        )
        # Clean any accidental TERMINATE token
        ch_text = re.sub(r"\bTERMINATE\b", "", ch_text).strip()
        chapters_prose.append(ch_text)
        words_written += len(ch_text.split())

        # ReIO Condensation (`MessageRedact.apply_transform` when content > 150 chars)
        if ch_idx < num_chapters and len(ch_text) > 150:
            condensed = backend.generate_chat(
                f"{STORYWRITER_REIO_CONDENSE_PROMPT}\n\n{ch_text}",
                system_prompt="You are a concise plot condenser.",
                max_new_tokens=140,
                temperature=0.3,
            )
            condensed_history.append(f"- Chapter {ch_idx} Summary: {condensed.strip()}")

    return "\n\n".join(chapters_prose)


def run_storywriter_benchmark(
    prompts: List[Dict[str, Any]],
    output_csv: Path,
    backend: Optional[QwenAgentBackend] = None,
):
    """Run STORYWRITER across all benchmark prompts and save CSV."""
    own_backend = False
    if backend is None:
        backend = QwenAgentBackend("Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True)
        backend.load()
        own_backend = True
    else:
        backend.load()

    print("\n" + "=" * 85, flush=True)
    print("RUNNING MODEL [2/8]: STORYWRITER (Event Graph + Sub-Events + ReIO 10% Compression)", flush=True)
    print("=" * 85, flush=True)

    records = []
    for idx, item in enumerate(prompts, 1):
        p_id = item["id"]
        p_text = item["prompt"]
        t_len = int(item["target_length"])
        print(f" -> [STORYWRITER] [{idx}/{len(prompts)}] {p_id} (Target: {t_len} words)...", flush=True)

        reset_vram_stats()
        t0 = time.time()
        raw_out = run_storywriter_single(p_text, t_len, backend)
        elapsed = round(max(0.01, time.time() - t0), 2)
        vram = get_peak_vram_gb()

        actual_w = len(raw_out.split())
        print(f"    Done in {elapsed}s | Words: {actual_w}/{t_len} | Peak VRAM: {vram} GB", flush=True)

        records.append({
            "Model": "STORYWRITER",
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

