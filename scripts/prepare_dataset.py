"""
Prepares MA-GRPO chapter-level training dataset from tell-me-a-story benchmark data.
Outputs a clean JSONL dataset with schema required by GRPOTrainer:
  - prompt (str)
  - target_words (int)
  - context_summary (str)
  - prev_chapter_tail (str)
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_INPUT = PROJECT_ROOT.parent / "Dataset" / "tell_me_a_story_data" / "tell-me-a-story-test.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = OUTPUT_DIR / "chapters.jsonl"


def extract_target_length(text: str) -> int:
    """Attempts to find word count constraints in the prompt (e.g. '1300-1400 words', '1600 words')."""
    match = re.search(r"(\d{3,5})\s*[-–to]{1,3}\s*(\d{3,5})\s*words", text, re.IGNORECASE)
    if match:
        return (int(match.group(1)) + int(match.group(2))) // 2
    match = re.search(r"(?:between|around|least|approximately)?\s*(\d{3,5})\s*words", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 1200


def build_writing_prompt(title: str, target_words: int, premise: str, goal: str, tail: str = "") -> str:
    tail_section = f"\n[PRECEDING SCENE CONTINUATION (Pick up immediately from this scene)]:\n{tail}\n" if tail else ""
    return f"""You are an elite literary novelist. Write the complete narrative prose for {title}.
Total Target Length: Approximately {target_words} words.

[CONTEXT SUMMARY]:
{premise}

[CHAPTER GOAL]:
{goal}
{tail_section}
[SCENE BREAKDOWN & WORD BUDGETS]:
- Scene 1: Opening exposition and setting introduction (~300 words)
- Scene 2: Inciting tension and dialogue development (~500 words)
- Scene 3: Climax or resolution of current chapter goal (~400 words)

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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_INPUT.exists():
        print(f"Error: {DATASET_INPUT} does not exist.")
        return

    records = []
    with open(DATASET_INPUT, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            premise = item.get("inputs", "").strip()
            target_words = extract_target_length(premise)
            # Chapter 1
            records.append({
                "prompt": build_writing_prompt(f"Chapter 1 of Story {idx+1}", target_words, premise, "Introduce protagonists and inciting incident."),
                "target_words": target_words,
                "context_summary": premise[:300],
                "prev_chapter_tail": "",
            })
            # If target story excerpt exists, create Chapter 2 record with preceding tail
            target_text = item.get("targets", "").strip()
            if target_text:
                words = target_text.split()
                tail_excerpt = " ".join(words[-150:]) if len(words) >= 150 else target_text
                records.append({
                    "prompt": build_writing_prompt(f"Chapter 2 of Story {idx+1}", target_words, premise, "Escalate narrative tension and face core conflict.", tail=tail_excerpt),
                    "target_words": target_words,
                    "context_summary": f"Previously: {premise[:250]}...",
                    "prev_chapter_tail": tail_excerpt,
                })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(records)} chapter training prompts into {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

