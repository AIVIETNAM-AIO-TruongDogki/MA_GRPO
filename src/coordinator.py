"""
Stage 2: Context Coordination & Context Gate Validator for MA-GRPO (Two-Tier Memory & Multi-Signal Verification).

Architecture:
    1. Coordinator Agent (ReIO with Two-Tier Memory):
       - long_term_memory: retains cumulative major milestones, character states, and unresolved plot threads across all past chapters.
       - short_term_memory: captures detailed summary and exact recent tail excerpt of the immediately preceding chapter.
    2. Context Gate Validator Agent (V_s):
       - Evaluates candidate summary across 5 contexts:
         (previous_chapter_text, previous_chapter_plan, previous_context, candidate_context, next_chapter_plan).
       - Replaces crude word overlap with multi-signal verification:
         * Negation Contradiction Check (polar polarity inversion detection)
         * Entity-State Grounding & Consistency
         * Forward Completeness for next chapter requirements
       - Triggers factual structured fallback (not generic prose) if V_s < 0.80.
       - Returns complete validation metadata for dataset quality audits.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


COORDINATOR_TWO_TIER_PROMPT = """You are the Context Coordinator (ReIO) for a multi-agent long-form novel system.
Condense the narrative history into a structured Two-Tier Context State (<4000 tokens) for the writing agent.

Respond strictly with valid JSON conforming to this schema:
{
  "long_term_memory": {
    "major_events": ["Key milestones achieved across all past chapters"],
    "character_states": {
      "CharacterName": "Current physical location, emotional state, and active inventory/bonds"
    },
    "unresolved_plot_threads": ["Open conflicts and mysteries that remain unanswered"],
    "revealed_secrets": ["Major revelations or twists that occurred so far"]
  },
  "short_term_memory": {
    "recent_chapter_summary": "Detailed, factual summary of the immediately preceding chapter (<300 words)",
    "immediate_turning_point": "The exact transition or dilemma occurring at the end of the last chapter"
  }
}
"""

VALIDATOR_MULTI_SIGNAL_PROMPT = """You are the Context Gate Validator Agent (V_s).
Audit the candidate context summary against the actual preceding chapter text and upcoming chapter goals.

Rate two criteria from 0.0 to 1.0:
1. Faithfulness (S_faith): Does the summary contain only facts verified in the preceding chapter/history, with zero hallucination or negation contradictions?
2. Completeness (S_comp): Does the summary retain all critical plot developments, character states, and prerequisites needed for the upcoming chapter plan?

Respond strictly with JSON:
{
  "faithfulness_score": 0.95,
  "completeness_score": 0.90,
  "negation_contradictions": [],
  "unsupported_claims": [],
  "feedback": "Detailed justification of scores"
}
"""


def clean_json_response(raw_text: str) -> str:
    """Extracts JSON string from potential markdown code fences."""
    raw_text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
    if match:
        return match.group(1).strip()
    return raw_text


NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "none", "cannot", "can't", "don't",
    "doesn't", "didn't", "won't", "refused", "refuses", "failed", "fails",
    "distrusted", "distrusts", "denied", "denies", "betrayed", "betrays"
}

COMMON_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "this", "that",
    "these", "those", "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "it", "its", "itself", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "you", "your", "yours", "yourself", "yourselves", "i", "me", "my",
    "myself", "we", "us", "our", "ours", "ourselves", "chapter", "scene", "part", "story",
    "events", "begins", "continues", "concludes", "starts", "ends", "further", "must",
    "will", "would", "shall", "should", "can", "could", "may", "might", "also", "just",
    "then", "there", "here", "where", "when", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "than", "too", "very", "now", "so",
    "yet", "as", "well", "out", "down", "off", "back", "again", "only", "first", "next"
}


def detect_heuristic_negation_contradictions(
    summary_text: str,
    source_text: str,
) -> List[str]:
    """
    Detects potential polarity inversions (e.g. source says 'Elena trusted Marcus',
    summary says 'Elena did not trust Marcus').
    Filters out common stopwords and scopes search to same-sentence proximity (<= 10 words).
    """
    contradictions = []
    summary_sentences = [s.strip() for s in re.split(r"[.!?]+", summary_text) if s.strip()]
    source_sentences = [s.strip().lower() for s in re.split(r"[.!?\n]+", source_text) if s.strip()]
    source_words = set(re.findall(r"\b\w{4,}\b", source_text.lower())) - COMMON_STOPWORDS - NEGATION_WORDS

    for s_sent in summary_sentences:
        s_words = [w.lower() for w in re.findall(r"\b\w+\b", s_sent)]
        content_words = [
            w for w in s_words
            if len(w) >= 4 and w not in NEGATION_WORDS and w not in COMMON_STOPWORDS
            and (w in source_words or any(cw.startswith(w[:4]) for cw in source_words))
        ]

        if len(content_words) >= 2:
            # Check adjacent or prominent content word pairs
            for i in range(min(4, len(content_words))):
                for j in range(i + 1, min(i + 3, len(content_words))):
                    w1, w2 = content_words[i], content_words[j]
                    # Check summary span between w1 and w2 within <= 8 words
                    m_sum = re.search(r"\b" + re.escape(w1) + r"\b((?:\W+\w+){0,8}?\W+)\b" + re.escape(w2) + r"\b", s_sent.lower())
                    if not m_sum:
                        continue
                    sum_neg = any(nw in m_sum.group(1).split() for nw in NEGATION_WORDS)

                    # Look for w1 and w2 in the same source sentence within <= 10 words
                    for src_sent in source_sentences:
                        m_src = re.search(r"\b" + re.escape(w1[:4]) + r"\w*\b((?:\W+\w+){0,10}?\W+)\b" + re.escape(w2[:4]) + r"\w*\b", src_sent)
                        if m_src:
                            src_neg = any(nw in m_src.group(1).split() for nw in NEGATION_WORDS)
                            if sum_neg != src_neg:
                                contradictions.append(
                                    f"Polarity clash: Summary sentence '{s_sent}' asserts negation={sum_neg} between '{w1}' and '{w2}', but source asserts negation={src_neg}."
                                )
                                break
                    if contradictions:
                        break

    return contradictions


class ContextGateValidator:
    """
    Context Gate Validator evaluating candidate summary s_i across 5 inputs:
        V_s(s_i) = 0.5 * S_faith(s_i, C_{i-1}, C_{1...i-2}) + 0.5 * S_comp(s_i, E_{i-1}, E_i)
    Threshold tau = 0.80.
    """

    def __init__(
        self,
        llm_generate_fn: Optional[Callable[[str, str], str]] = None,
        threshold: float = 0.80,
    ):
        self.generate_fn = llm_generate_fn
        self.threshold = threshold

    def validate_summary(
        self,
        summary: Dict[str, Any],
        prev_chapter_text: str = "",
        prev_chapter_plan: Optional[Dict[str, Any]] = None,
        previous_context: Optional[Dict[str, Any]] = None,
        next_chapter_plan: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Validates faithfulness and completeness of candidate summary s_i
        using the complete 5-component context suite.
        """
        if not prev_chapter_text or not prev_chapter_text.strip():
            return {
                "faithfulness_score": 1.0,
                "completeness_score": 1.0,
                "composite_score": 1.0,
                "is_valid": True,
                "is_fallback": False,
                "feedback": "Initial chapter; summary validation bypassed.",
                "issues": [],
            }

        short_mem = summary.get("short_term_memory", {})
        recent_summary = short_mem.get("recent_chapter_summary", summary.get("plot_summary", ""))
        long_mem = summary.get("long_term_memory", {})

        issues = []
        neg_clashes = detect_heuristic_negation_contradictions(recent_summary, prev_chapter_text)
        if neg_clashes:
            issues.extend(neg_clashes)

        if self.generate_fn:
            next_goal = (next_chapter_plan or {}).get("chapter_goal", "Advance narrative.")
            prev_goal = (prev_chapter_plan or {}).get("chapter_goal", "Established story state.")

            user_prompt = f"""[PRECEDING CHAPTER PROSE EXCERPTS (Source of Truth)]:
--- BEGINNING OF CHAPTER ---
{prev_chapter_text[:1200]}
...
--- CONCLUSION OF CHAPTER ---
{prev_chapter_text[-1800:]}

[PRECEDING CHAPTER GOAL]: {prev_goal}
[UPCOMING CHAPTER GOAL]: {next_goal}

[PROPOSED CANDIDATE CONTEXT]:
Short-Term Recent Summary: {recent_summary}
Character States: {long_mem.get('character_states', summary.get('character_statuses', []))}
Unresolved Threads: {long_mem.get('unresolved_plot_threads', summary.get('pending_threads', []))}
Major Past Events: {long_mem.get('major_events', [])}

Audit faithfulness (0.0-1.0) and completeness (0.0-1.0).
"""
            try:
                raw_res = self.generate_fn(VALIDATOR_MULTI_SIGNAL_PROMPT, user_prompt)
                data = json.loads(clean_json_response(raw_res))
                faith = float(data.get("faithfulness_score", 0.85))
                comp = float(data.get("completeness_score", 0.85))
                feedback = str(data.get("feedback", "Validation processed."))
                llm_contradictions = data.get("negation_contradictions", [])
                if llm_contradictions:
                    issues.extend(llm_contradictions)
            except Exception:
                faith, comp = self._heuristic_evaluation(summary, prev_chapter_text, next_chapter_plan)
                feedback = "Fallback heuristic scoring applied."
        else:
            faith, comp = self._heuristic_evaluation(summary, prev_chapter_text, next_chapter_plan)
            feedback = "Heuristic multi-signal verification applied."

        # Penalty for detected negation contradictions
        if neg_clashes:
            if self.generate_fn:
                # LLM validator is active: only heavily penalize if LLM also flagged contradictions;
                # otherwise apply calibrated minor adjustment to prevent regex false positives from overturning 70B+ LLM judgment.
                if llm_contradictions:
                    faith = max(0.0, faith - 0.25 * len(set(neg_clashes) | set(llm_contradictions)))
                else:
                    faith = max(0.0, faith - 0.05 * len(neg_clashes))
            else:
                faith = max(0.0, faith - 0.25 * len(neg_clashes))

        composite = round(0.5 * faith + 0.5 * comp, 3)
        is_valid = composite >= self.threshold

        return {
            "faithfulness_score": round(faith, 3),
            "completeness_score": round(comp, 3),
            "composite_score": composite,
            "is_valid": is_valid,
            "is_fallback": not is_valid,
            "feedback": feedback if is_valid else f"V_s ({composite}) < threshold ({self.threshold}). Structured fallback triggered.",
            "issues": issues,
        }

    def _heuristic_evaluation(
        self,
        summary: Dict[str, Any],
        prev_text: str,
        next_plan: Optional[Dict[str, Any]],
    ) -> Tuple[float, float]:
        """
        Multi-signal heuristic verification:
        1. Entity Grounding: checks if active character names from source exist in summary.
        2. Content Coverage: checks key noun overlap while penalizing empty claims.
        3. Forward Completeness: checks whether requirements of upcoming plan are preserved.
        """
        prev_words = set(re.findall(r"\b\w{4,}\b", prev_text.lower()))
        plot_text = summary.get("short_term_memory", {}).get(
            "recent_chapter_summary", summary.get("plot_summary", "")
        )
        summary_words = set(re.findall(r"\b\w{4,}\b", plot_text.lower()))

        if not summary_words:
            return 0.3, 0.3

        overlap = len(summary_words & prev_words) / float(len(summary_words))
        # Grounded faithfulness
        faith = min(1.0, max(0.4, overlap * 0.8 + 0.2))

        # Completeness against next plan
        comp = 0.85
        if next_plan:
            active_chars = [c.lower() for c in next_plan.get("characters_active", [])]
            for c in active_chars:
                if c not in plot_text.lower() and c not in str(summary.get("long_term_memory", {})).lower():
                    comp -= 0.1
        comp = max(0.5, min(1.0, comp))

        return round(faith, 3), round(comp, 3)


class CoordinatorAgent:
    """
    Coordinator Agent (ReIO with Two-Tier Memory).
    Produces rich two-tier context states s_i with strict factual validation.
    """

    def __init__(
        self,
        llm_generate_fn: Optional[Callable[[str, str], str]] = None,
        validator: Optional[ContextGateValidator] = None,
    ):
        self.generate_fn = llm_generate_fn
        self.validator = validator or ContextGateValidator(llm_generate_fn)

    def condense_context(
        self,
        next_chapter_index: int,
        past_chapters: List[str],
        past_plans: List[Dict[str, Any]],
        characters: Optional[List[str]] = None,
        previous_context: Optional[Dict[str, Any]] = None,
        next_chapter_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Produces validated two-tier context summary s_i for chapter i.
        """
        characters = characters or ["Protagonist"]

        # Initial Chapter (Chapter 1)
        if not past_chapters or next_chapter_index == 1:
            initial_context = {
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
                "validation_metadata": {
                    "faithfulness_score": 1.0,
                    "completeness_score": 1.0,
                    "composite_score": 1.0,
                    "is_valid": True,
                    "is_fallback": False,
                    "fallback_reason": None,
                    "issues": [],
                },
            }
            return initial_context

        prev_text = past_chapters[-1]
        prev_tail_words = prev_text.strip().split()[-180:]
        recent_tail = " ".join(prev_tail_words)

        candidate_summary = self._generate_two_tier_summary(
            next_chapter_index=next_chapter_index,
            past_chapters=past_chapters,
            past_plans=past_plans,
            characters=characters,
            recent_tail=recent_tail,
            previous_context=previous_context,
            next_chapter_plan=next_chapter_plan,
        )

        val_result = self.validator.validate_summary(
            summary=candidate_summary,
            prev_chapter_text=prev_text,
            prev_chapter_plan=past_plans[-1] if past_plans else None,
            previous_context=previous_context,
            next_chapter_plan=next_chapter_plan,
        )

        if val_result.get("is_valid", False):
            candidate_summary["validation_metadata"] = val_result
            candidate_summary["validation_metadata"]["is_fallback"] = False
            candidate_summary["validation_metadata"]["fallback_reason"] = None
            return candidate_summary

        # Structured State Extraction Fallback (Non-Generic)
        fallback_summary = self._structured_state_extraction(
            next_chapter_index=next_chapter_index,
            past_chapters=past_chapters,
            past_plans=past_plans,
            characters=characters,
            recent_tail=recent_tail,
            previous_context=previous_context,
            next_chapter_plan=next_chapter_plan,
        )
        fallback_summary["validation_metadata"] = val_result
        fallback_summary["validation_metadata"]["is_fallback"] = True
        fallback_summary["validation_metadata"]["fallback_reason"] = val_result.get("feedback")
        return fallback_summary

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
        """Calls LLM to synthesize two-tier memory, or derives it structurally."""
        if self.generate_fn:
            user_prompt = f"""Total chapters completed: {len(past_chapters)}.
Upcoming chapter index: {next_chapter_index}.

[LAST CHAPTER PROSE (TAIL)]:
{past_chapters[-1][-2500:]}

[LAST CHAPTER PLAN GOAL]:
{(past_plans[-1] if past_plans else {}).get('chapter_goal', '')}
[LAST CHAPTER EXIT STATE]:
{(past_plans[-1] if past_plans else {}).get('exit_state', '')}

Synthesize Two-Tier Context State matching the JSON schema.
"""
            try:
                raw_res = self.generate_fn(COORDINATOR_TWO_TIER_PROMPT, user_prompt)
                data = json.loads(clean_json_response(raw_res))
                long_mem = data.get("long_term_memory", {})
                short_mem = data.get("short_term_memory", {})
                short_mem["recent_chapter_index"] = len(past_chapters)
                short_mem["recent_tail_excerpt"] = recent_tail

                # Compatibility flattening
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

        # Carry forward prior character states and update with recent active characters
        prev_long_mem = (previous_context or {}).get("long_term_memory", {})
        prior_char_states = dict(prev_long_mem.get("character_states", {}))
        for c in characters:
            prior_char_states[c] = f"Active at conclusion of Chapter {len(past_chapters)}."

        open_threads = list(prev_long_mem.get("unresolved_plot_threads", []))
        if last_plan.get("exit_state") and last_plan["exit_state"] not in open_threads:
            open_threads.append(f"Consequence of: {last_plan['exit_state']}")

        long_term_mem = {
            "major_events": major_milestones,
            "character_states": prior_char_states,
            "unresolved_plot_threads": open_threads[-5:],
            "revealed_secrets": list(prev_long_mem.get("revealed_secrets", [])),
        }

        short_term_mem = {
            "recent_chapter_index": len(past_chapters),
            "recent_chapter_summary": recent_summary,
            "recent_tail_excerpt": recent_tail,
            "immediate_turning_point": last_plan.get("exit_state", "Entering next phase."),
        }

        return {
            "chapter_index": next_chapter_index,
            "completed_chapters": len(past_chapters),
            "long_term_memory": long_term_mem,
            "short_term_memory": short_term_mem,
            "plot_summary": recent_summary,
            "character_statuses": [f"{k}: {v}" for k, v in prior_char_states.items()],
            "pending_threads": open_threads[-5:],
            "recent_tail_excerpt": recent_tail,
        }
