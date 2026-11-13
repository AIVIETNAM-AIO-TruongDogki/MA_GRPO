"""
Shared Story Arc Alignment, Event Graph Parsing & Token Utilities for Stage-1 Planning.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


STORY_ARC_STAGES: Tuple[str, ...] = (
    "Exposition",
    "Rising Action",
    "Climax",
    "Falling Action",
    "Resolution",
)

_STORY_ARC_ALIASES: Dict[str, str] = {
    "exposition": "Exposition",
    "opening": "Exposition",
    "introduction": "Exposition",
    "setup": "Exposition",
    "mo dau": "Exposition",
    "mở đầu": "Exposition",
    "rising action": "Rising Action",
    "rising": "Rising Action",
    "inciting incident": "Rising Action",
    "complication": "Rising Action",
    "escalation": "Rising Action",
    "that nut": "Rising Action",
    "thắt nút": "Rising Action",
    "climax": "Climax",
    "turning point": "Climax",
    "crisis": "Climax",
    "cao trao": "Climax",
    "cao trào": "Climax",
    "falling action": "Falling Action",
    "falling": "Falling Action",
    "aftermath": "Falling Action",
    "mo nut": "Falling Action",
    "mở nút": "Falling Action",
    "resolution": "Resolution",
    "denouement": "Resolution",
    "conclusion": "Resolution",
    "ending": "Resolution",
    "ket thuc": "Resolution",
    "kết thúc": "Resolution",
}


EVENT_GENERATION_PROMPT = """Generate a structured Macro Event Graph outline based on the premise.
Tag EVERY event with its dramatic Story Arc stage from:
[Exposition, Rising Action, Climax, Falling Action, Resolution].

Follow this exact format for each event:
Event 1: [Event Title]
Story Arc: [Exposition | Rising Action | Climax | Falling Action | Resolution]
Setting: [Time and Location]
Character: [Characters involved and their roles]
Action: [What happens in this event]
Conflict: [The struggle, opposition, or dilemma]
Plot Twist: [Unexpected revelation or turning point]
Depends On: [None or Event IDs e.g. E1]

Event 2: [Event Title]
Story Arc: ...
"""


def estimate_token_count(text: str, tokenizer: Any = None) -> int:
    """Count tokens using provided tokenizer or deterministic BPE word ratio (~1.33 tok/word)."""
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    words = len(text.split())
    return max(1, int(round(words * 1.33)))


def infer_story_arc_stage(index: int, total: int) -> str:
    """Map proportional event index (0..total-1) to Freytag's 5 Story Arc stages."""
    if total <= 1:
        return "Exposition"
    pos = index / max(1, total - 1)
    if pos < 0.20:
        return "Exposition"
    if pos < 0.55:
        return "Rising Action"
    if pos < 0.72:
        return "Climax"
    if pos < 0.88:
        return "Falling Action"
    return "Resolution"


def normalize_story_arc_label(label: Optional[str], index: int = 0, total: int = 5) -> str:
    """Normalize a raw story arc label into one of STORY_ARC_STAGES."""
    if label:
        cleaned = re.sub(
            r"[^a-zA-Z\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
            " ",
            str(label),
        ).strip().lower()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if cleaned in _STORY_ARC_ALIASES:
            return _STORY_ARC_ALIASES[cleaned]
        for alias, canonical in _STORY_ARC_ALIASES.items():
            if alias in cleaned:
                return canonical
    return infer_story_arc_stage(index, total)


def assign_story_arc_labels(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure every event carries a canonical 5-stage Story Arc tag."""
    total = len(events)
    for idx, ev in enumerate(events):
        raw_arc = ev.get("story_arc")
        ev["story_arc"] = normalize_story_arc_label(raw_arc, index=idx, total=total)
    return events


def compute_story_arc_alignment(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate Freytag 5-Stage Story Arc Alignment (S_arc):
      - stage_coverage: fraction of STORY_ARC_STAGES covered by the event graph.
      - order_monotonicity: fraction of consecutive transitions whose arc index does not regress.
      - story_arc_score: 0.5 * stage_coverage + 0.5 * order_monotonicity.
    """
    if not events:
        return {
            "story_arc_score": 0.0,
            "stage_coverage": 0.0,
            "order_monotonicity": 0.0,
            "present_stages": [],
            "missing_stages": list(STORY_ARC_STAGES),
            "sequence_violations": [],
        }

    total = len(events)
    stage_rank = {stage: idx for idx, stage in enumerate(STORY_ARC_STAGES)}
    arc_sequence = [
        normalize_story_arc_label(ev.get("story_arc"), index=i, total=total)
        for i, ev in enumerate(events)
    ]

    present = [s for s in STORY_ARC_STAGES if s in arc_sequence]
    missing = [s for s in STORY_ARC_STAGES if s not in arc_sequence]
    stage_coverage = round(len(present) / len(STORY_ARC_STAGES), 4)

    violations: List[str] = []
    if len(arc_sequence) <= 1:
        order_monotonicity = 1.0
    else:
        valid_transitions = 0
        for i in range(len(arc_sequence) - 1):
            r_cur = stage_rank[arc_sequence[i]]
            r_next = stage_rank[arc_sequence[i + 1]]
            if r_next >= r_cur:
                valid_transitions += 1
            else:
                violations.append(
                    f"Arc regression at E{i+1}->E{i+2}: {arc_sequence[i]} -> {arc_sequence[i+1]}"
                )
        order_monotonicity = round(valid_transitions / (len(arc_sequence) - 1), 4)

    story_arc_score = round(0.5 * stage_coverage + 0.5 * order_monotonicity, 4)
    return {
        "story_arc_score": story_arc_score,
        "stage_coverage": stage_coverage,
        "order_monotonicity": order_monotonicity,
        "present_stages": present,
        "missing_stages": missing,
        "arc_sequence": arc_sequence,
        "sequence_violations": violations,
    }


def parse_storywriter_events(text: str) -> List[Dict[str, str]]:
    """
    Parses StoryWriter-style formatted event text using regex matching.
    Extracts Setting, Character, Action, Conflict, Plot Twist, and optional Story Arc & Depends On.
    """
    if not text:
        return []
    event_blocks = re.findall(r"(Event \d+:.*?)(?=(?:\n\s*Event \d+:|\Z))", text, re.DOTALL)
    parsed_events = []

    field_lookahead = r"(?=\n(?:Story Arc|Setting|Character|Action|Conflict|Plot Twist|Depends On)|\Z)"
    total_blocks = len(event_blocks)

    for idx, block in enumerate(event_blocks):
        title_match = re.search(r"Event \d+:\s*(.*?)(?=\n|$)", block)
        arc_match = re.search(rf"Story Arc:\s*(.*?){field_lookahead}", block, re.DOTALL)
        setting_match = re.search(rf"Setting:\s*(.*?){field_lookahead}", block, re.DOTALL)
        char_match = re.search(rf"Character:\s*(.*?){field_lookahead}", block, re.DOTALL)
        action_match = re.search(rf"Action:\s*(.*?){field_lookahead}", block, re.DOTALL)
        conflict_match = re.search(rf"Conflict:\s*(.*?){field_lookahead}", block, re.DOTALL)
        twist_match = re.search(rf"Plot Twist:\s*(.*?){field_lookahead}", block, re.DOTALL)
        deps_match = re.search(rf"Depends On:\s*(.*?){field_lookahead}", block, re.DOTALL)

        raw_arc = arc_match.group(1).strip() if arc_match else None
        story_arc = normalize_story_arc_label(raw_arc, index=idx, total=total_blocks)

        event_dict: Dict[str, Any] = {
            "title": title_match.group(1).strip() if title_match else "Untitled Event",
            "story_arc": story_arc,
            "setting": setting_match.group(1).strip() if setting_match else "Unknown Setting",
            "character": char_match.group(1).strip() if char_match else "Protagonist",
            "action": action_match.group(1).strip() if action_match else "Story progresses",
            "conflict": conflict_match.group(1).strip() if conflict_match else "Obstacles encountered",
            "plot_twist": twist_match.group(1).strip() if twist_match else "None",
            "raw_text": block.strip(),
        }

        if deps_match:
            deps_str = deps_match.group(1).strip()
            found_ids = re.findall(r"\bE\d+\b", deps_str.upper())
            if found_ids:
                event_dict["depends_on"] = [d for d in found_ids if d != f"E{idx + 1}"]

        parsed_events.append(event_dict)

    return parsed_events


def ensure_event_ids(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach stable IDs, sequential causal dependencies, and Story Arc tags to an Event Graph."""
    total = len(events)
    for index, event in enumerate(events, 1):
        event.setdefault("event_id", f"E{index}")
        event.setdefault("depends_on", [f"E{index - 1}"] if index > 1 else [])
        event.setdefault("status", "pending")
        event["story_arc"] = normalize_story_arc_label(event.get("story_arc"), index=index - 1, total=total)
    return events


def extract_premise_keywords_and_characters(premise: str) -> Tuple[List[str], List[str]]:
    """Extract key nouns, proper names, and potential characters from premise text."""
    words = premise.split()
    proper_nouns = [
        re.sub(r"\W+", "", w) for w in words
        if w and w[0].isupper() and re.sub(r"\W+", "", w).lower() not in {
            "write", "a", "an", "the", "in", "on", "at", "between", "words",
            "story", "about", "there", "she", "he", "they", "when", "make", "this",
        }
    ]
    characters = list(dict.fromkeys([p for p in proper_nouns if len(p) > 1]))
    if not characters:
        characters = ["Protagonist", "Mentor", "Antagonist"]
    elif len(characters) == 1:
        characters.append("Ally")

    keywords = [re.sub(r"\W+", "", w).lower() for w in words if len(w) > 4][:12]
    return characters, keywords

