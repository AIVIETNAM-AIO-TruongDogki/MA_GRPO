"""
Stage 2: Context Coordination for MA-GRPO (ReIO with Cumulative Two-Tier Memory).

Architecture:
    Coordinator Agent (ReIO with Two-Tier Memory):
      - long_term_memory (hard-capped <= ~220 words): cumulatively tracks major milestones
        (max 8), active character states (max 6), unresolved plot threads (max 4), and
        revealed secrets (max 4) across ALL preceding chapters (Ch 1 .. Ch k-1).
      - short_term_memory: concise summary (60-150 words) + sentence-boundary-aligned
        `recent_tail_excerpt` (up to 180 words) of the immediately preceding chapter.
      - Falls back to structured state extraction strictly when offline or JSON parse fails.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional


COORDINATOR_TWO_TIER_PROMPT = """You are the Context Coordinator (ReIO) for a multi-agent long-form novel system.
Update the cumulative Two-Tier Context State by integrating the newly completed chapter into the previous Long-Term Memory.

Strict Conciseness & Continuity Rules:
1. `long_term_memory.major_events`: Preserve key milestones from earlier chapters AND append 1 concise sentence (<=18 words) for the newly completed chapter. Keep at most 8 chronological items.
2. `long_term_memory.character_states`: Update current location, emotional state, and key items/bonds for up to 6 main characters (<=20 words each).
3. `long_term_memory.unresolved_plot_threads`: Keep only currently open conflicts/mysteries (remove resolved ones; max 4 items).
4. `long_term_memory.revealed_secrets`: Keep major twists/revelations uncovered so far (max 4 items).
5. `short_term_memory.recent_chapter_summary`: Concise, factual summary of ONLY the immediately preceding chapter (60-150 words).

Respond strictly with valid JSON conforming to this schema:
{
  "long_term_memory": {
    "major_events": ["Ch 1: ...", "Ch 2: ..."],
    "character_states": {
      "CharacterName": "Location, state, and active goal/bond (<=20 words)"
    },
    "unresolved_plot_threads": ["Open thread 1"],
    "revealed_secrets": ["Major revelation 1"]
  },
  "short_term_memory": {
    "recent_chapter_summary": "Factual 60-150 word summary of the last chapter",
    "immediate_turning_point": "Exact transition moment at the end of the last chapter"
  }
}
"""


def clean_json_response(raw_text: str) -> str:
    """Extracts JSON string from potential markdown code fences."""
    raw_text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
    if match:
        return match.group(1).strip()
    return raw_text


def _truncate_words(text: str, max_words: int) -> str:
    words = str(text or "").strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(",;:") + "..."


def extract_clean_tail(text: str, max_words: int = 180, min_words: int = 75) -> str:
    """
    Extracts a sentence-boundary-aligned excerpt of up to `max_words` from the end of a chapter:
      1. Trims any dangling incomplete trailing fragment (e.g. if draft hit max_tokens mid-sentence).
      2. Takes up to `max_words` words from the clean ending.
      3. Advances the start to the first complete sentence boundary whenever enough context remains.
    """
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return ""

    # 1. Trim dangling mid-sentence fragment at the end (keep text up to last terminal punctuation)
    end_matches = list(re.finditer(r'[.!?]["”’\')\]]*(?=\s|$)', cleaned))
    if end_matches:
        last_end = end_matches[-1].end()
        # Only trim trailing fragment if we keep a meaningful body (>= 40% of text or >= 30 words)
        trimmed = cleaned[:last_end].strip()
        if len(trimmed.split()) >= min(30, max(1, int(len(cleaned.split()) * 0.4))):
            cleaned = trimmed

    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned

    # 2. Take trailing window of `max_words` words
    window = " ".join(words[-max_words:])

    # 3. Advance start past a partial leading sentence if the remaining text is still >= min_words
    #    (and the window didn't already start cleanly after a sentence end in the original text)
    prefix_last_word = words[-max_words - 1] if len(words) > max_words else ""
    already_clean_start = bool(re.search(r'[.!?]["”’\')\]]*$', prefix_last_word))
    if not already_clean_start:
        first_bound = re.search(r'[.!?]["”’\')\]]*\s+(?=[A-Z“"‘\'(\[])', window)
        if first_bound:
            candidate = window[first_bound.end() :].strip()
            if len(candidate.split()) >= min_words:
                return candidate
    return window


def compact_long_term_memory(
    ltm: Optional[Dict[str, Any]],
    previous_ltm: Optional[Dict[str, Any]] = None,
    max_events: int = 8,
    max_characters: int = 6,
    max_threads: int = 4,
    max_secrets: int = 4,
    max_item_words: int = 22,
) -> Dict[str, Any]:
    """
    Normalizes, cumulatively preserves, and hard-caps `long_term_memory` so it stays
    informative across Chapters 1..10 while remaining compact (<= ~220 words in prompts).
    """
    src = ltm if isinstance(ltm, dict) else {}
    prev = previous_ltm if isinstance(previous_ltm, dict) else {}

    def _dedup_list(items: List[Any], limit: int) -> List[str]:
        out: List[str] = []
        seen = set()
        for it in items:
            s = _truncate_words(str(it), max_item_words)
            key = re.sub(r"\W+", " ", s.lower()).strip()
            if s and key and key not in seen and key != "narrative begins":
                seen.add(key)
                out.append(s)
        return out[-limit:]

    raw_events = src.get("major_events") if isinstance(src.get("major_events"), list) else []
    prev_events = prev.get("major_events") if isinstance(prev.get("major_events"), list) else []
    # Preserve prior milestones if LLM omitted them during an incremental update
    merged_events = _dedup_list(list(prev_events) + list(raw_events), max_events)
    if not merged_events:
        merged_events = _dedup_list(raw_events, max_events)

    raw_chars = src.get("character_states") if isinstance(src.get("character_states"), dict) else {}
    prev_chars = prev.get("character_states") if isinstance(prev.get("character_states"), dict) else {}
    merged_chars: Dict[str, str] = {}
    for k, v in prev_chars.items():
        if str(k).strip():
            merged_chars[str(k).strip()] = _truncate_words(str(v), max_item_words)
    for k, v in raw_chars.items():
        if str(k).strip():
            merged_chars[str(k).strip()] = _truncate_words(str(v), max_item_words)
    # Keep up to `max_characters` (prioritizing most recently updated keys)
    if len(merged_chars) > max_characters:
        keys = list(merged_chars.keys())[-max_characters:]
        merged_chars = {k: merged_chars[k] for k in keys}

    raw_threads = src.get("unresolved_plot_threads") if isinstance(src.get("unresolved_plot_threads"), list) else []
    threads = _dedup_list(raw_threads, max_threads)

    raw_secrets = src.get("revealed_secrets") if isinstance(src.get("revealed_secrets"), list) else []
    prev_secrets = prev.get("revealed_secrets") if isinstance(prev.get("revealed_secrets"), list) else []
    secrets = _dedup_list(list(prev_secrets) + list(raw_secrets), max_secrets)

    return {
        "major_events": merged_events,
        "character_states": merged_chars,
        "unresolved_plot_threads": threads,
        "revealed_secrets": secrets,
    }


def format_long_term_memory_block(context_summary: Dict[str, Any], max_total_words: int = 220) -> str:
    """
    Formats a compact `[LONG-TERM STORY MEMORY]` prompt block from `context_summary['long_term_memory']`.
    Returns empty string for Chapter 1 / initial baseline so Ch 1 prompts remain clean.
    """
    if not isinstance(context_summary, dict):
        return ""
    if int(context_summary.get("completed_chapters", 0) or 0) <= 0:
        return ""
    ltm = context_summary.get("long_term_memory")
    if not isinstance(ltm, dict):
        return ""

    compact = compact_long_term_memory(ltm)
    lines: List[str] = []
    if compact["major_events"]:
        lines.append("Key Prior Events: " + " | ".join(compact["major_events"]))
    if compact["character_states"]:
        char_str = "; ".join(f"{k}: {v}" for k, v in compact["character_states"].items())
        lines.append(f"Character States: {char_str}")
    if compact["unresolved_plot_threads"]:
        lines.append("Open Threads: " + "; ".join(compact["unresolved_plot_threads"]))
    if compact["revealed_secrets"]:
        lines.append("Revealed Facts: " + "; ".join(compact["revealed_secrets"]))

    if not lines:
        return ""

    # Enforce hard total word cap (`max_total_words`) cleanly line-by-line
    block = "\n".join(lines)
    words = block.split()
    if len(words) > max_total_words:
        block = " ".join(words[:max_total_words]).rstrip(",;:") + "..."
    return f"[LONG-TERM STORY MEMORY]:\n{block}\n\n"


class CoordinatorAgent:
    """
    Coordinator Agent (ReIO with Cumulative Two-Tier Memory).
    Produces compact two-tier context states s_i directly from narrative history
    and previous Long-Term Memory without external validator overhead.
    """

    def __init__(
        self,
        llm_generate_fn: Optional[Callable[[str, str], str]] = None,
    ):
        self.generate_fn = llm_generate_fn

    def condense_context(
        self,
        next_chapter_index: int,
        past_chapters: List[str],
        past_plans: List[Dict[str, Any]],
        characters: Optional[List[str]] = None,
        previous_context: Optional[Dict[str, Any]] = None,
        next_chapter_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Produces cumulative two-tier context summary s_i for chapter i."""
        characters = characters or ["Protagonist"]

        if not past_chapters or next_chapter_index == 1:
            return {
                "chapter_index": 1,
                "completed_chapters": 0,
                "long_term_memory": {
                    "major_events": ["Narrative begins."],
                    "character_states": {c: "Present at story inception." for c in characters},
                    "unresolved_plot_threads": ["Introduction of core premise."],
                    "revealed_secrets": [],
                },
                "short_term_memory": {
                    "recent_chapter_index": 0,
                    "recent_chapter_summary": "Initial world-building and character baseline.",
                    "recent_tail_excerpt": "",
                    "immediate_turning_point": "Embarking on primary mission.",
                },
                "plot_summary": "Initial world-building and character baseline.",
                "character_statuses": [f"{c}: Present at story inception." for c in characters],
                "pending_threads": ["Introduction of core premise."],
                "recent_tail_excerpt": "",
            }

        prev_text = past_chapters[-1]
        recent_tail = extract_clean_tail(prev_text, max_words=180)

        return self._generate_two_tier_summary(
            next_chapter_index=next_chapter_index,
            past_chapters=past_chapters,
            past_plans=past_plans,
            characters=characters,
            recent_tail=recent_tail,
            previous_context=previous_context,
            next_chapter_plan=next_chapter_plan,
        )

    def _generate_two_tier_summary(
        self,
        next_chapter_index: int,
        past_chapters: List[str],
        past_plans: List[Dict[str, Any]],
        characters: List[str],
        recent_tail: str,
        previous_context: Optional[Dict[str, Any]] = None,
        next_chapter_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Calls LLM to cumulatively update two-tier memory, or derives it structurally when offline/parse-failed."""
        prev_ltm = (previous_context or {}).get("long_term_memory")
        if self.generate_fn:
            prev_ltm_json = (
                json.dumps(compact_long_term_memory(prev_ltm), ensure_ascii=False)
                if isinstance(prev_ltm, dict) and int((previous_context or {}).get("completed_chapters", 0) or 0) > 0
                else "{}"
            )
            user_prompt = f"""Total chapters completed: {len(past_chapters)}.
Upcoming chapter index: {next_chapter_index}.

[PREVIOUS LONG-TERM MEMORY (Preserve earlier milestones & update)]:
{prev_ltm_json}

[LAST CHAPTER PROSE (TAIL)]:
{past_chapters[-1][-2800:]}

[LAST CHAPTER PLAN GOAL]:
{(past_plans[-1] if past_plans else {}).get('chapter_goal', '')}
[LAST CHAPTER EXIT STATE]:
{(past_plans[-1] if past_plans else {}).get('exit_state', '')}

Update and return the Two-Tier Context State matching the JSON schema.
"""
            try:
                raw_res = self.generate_fn(COORDINATOR_TWO_TIER_PROMPT, user_prompt)
                data = json.loads(clean_json_response(raw_res))
                long_mem = compact_long_term_memory(
                    data.get("long_term_memory", {}),
                    previous_ltm=prev_ltm if int((previous_context or {}).get("completed_chapters", 0) or 0) > 0 else None,
                )
                short_mem = data.get("short_term_memory", {})
                if not isinstance(short_mem, dict):
                    short_mem = {}
                short_mem["recent_chapter_index"] = len(past_chapters)
                short_mem["recent_chapter_summary"] = _truncate_words(
                    short_mem.get("recent_chapter_summary", ""), max_words=180
                )
                short_mem["recent_tail_excerpt"] = recent_tail

                plot_summary = short_mem.get("recent_chapter_summary", "")
                char_statuses = [
                    f"{k}: {v}" for k, v in long_mem.get("character_states", {}).items()
                ] or [f"{c}: Active" for c in characters]
                pending = long_mem.get("unresolved_plot_threads", [])

                return {
                    "chapter_index": next_chapter_index,
                    "completed_chapters": len(past_chapters),
                    "long_term_memory": long_mem,
                    "short_term_memory": short_mem,
                    "plot_summary": plot_summary,
                    "character_statuses": char_statuses,
                    "pending_threads": pending,
                    "recent_tail_excerpt": recent_tail,
                }
            except Exception:
                pass

        return self._structured_state_extraction(
            next_chapter_index, past_chapters, past_plans, characters, recent_tail, previous_context, next_chapter_plan
        )

    def _structured_state_extraction(
        self,
        next_chapter_index: int,
        past_chapters: List[str],
        past_plans: List[Dict[str, Any]],
        characters: List[str],
        recent_tail: str,
        previous_context: Optional[Dict[str, Any]] = None,
        next_chapter_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Factual, non-generic structured state extraction directly from completed plans,
        actual narrative exit states, and cumulative long-term facts.
        """
        major_milestones = []
        for p in past_plans:
            title = p.get("chapter_title", f"Chapter {p.get('chapter_index')}")
            goal = p.get("chapter_goal", "Executed narrative sequence")
            exit_s = p.get("exit_state", "")
            major_milestones.append(f"{title}: {goal} -> Result: {exit_s}")

        last_plan = past_plans[-1] if past_plans else {}
        recent_summary = (
            f"In Chapter {last_plan.get('chapter_index', len(past_chapters))}, the protagonists "
            f"pursued '{last_plan.get('chapter_goal', 'their objective')}', culminating in: "
            f"'{last_plan.get('exit_state', 'active transition point')}'."
        )

        prev_long_mem = (previous_context or {}).get("long_term_memory", {})
        prior_char_states = dict(prev_long_mem.get("character_states", {}))
        for c in characters:
            prior_char_states[c] = f"Active at conclusion of Chapter {len(past_chapters)}."

        open_threads = list(prev_long_mem.get("unresolved_plot_threads", []))
        if last_plan.get("exit_state") and last_plan["exit_state"] not in open_threads:
            open_threads.append(f"Consequence of: {last_plan['exit_state']}")

        long_term_mem = compact_long_term_memory(
            {
                "major_events": major_milestones,
                "character_states": prior_char_states,
                "unresolved_plot_threads": open_threads[-4:],
                "revealed_secrets": list(prev_long_mem.get("revealed_secrets", [])),
            }
        )

        short_term_mem = {
            "recent_chapter_index": len(past_chapters),
            "recent_chapter_summary": _truncate_words(recent_summary, max_words=180),
            "recent_tail_excerpt": recent_tail,
            "immediate_turning_point": last_plan.get("exit_state", "Entering next phase."),
        }

        return {
            "chapter_index": next_chapter_index,
            "completed_chapters": len(past_chapters),
            "long_term_memory": long_term_mem,
            "short_term_memory": short_term_mem,
            "plot_summary": short_term_mem["recent_chapter_summary"],
            "character_statuses": [f"{k}: {v}" for k, v in long_term_mem["character_states"].items()],
            "pending_threads": long_term_mem["unresolved_plot_threads"],
            "recent_tail_excerpt": recent_tail,
        }
