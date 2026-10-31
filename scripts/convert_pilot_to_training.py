"""
Converts pilot_chapter_prompts.jsonl (from Stage 1 & Stage 2 pipeline)
into standard MA-GRPO training format:
  - prompt (str)
  - target_words (int)
  - context_summary (str)
  - prev_chapter_tail (str)
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT.parent / "pilot_dataset_artifacts" / "content" / "output_pilot_dataset" / "pilot_chapter_prompts.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "pilot_chapters.jsonl"

THINK_INSTRUCTIONS = """
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


def convert():
    if not INPUT_FILE.exists():
        print(f"File not found: {INPUT_FILE}")
        return

    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item.get("full_training_prompt", "").strip()
            if "Begin with <think> now:" not in prompt:
                prompt = prompt + "\n" + THINK_INSTRUCTIONS.strip()

            target_words = int(item.get("target_words", 1200))
            short_mem = item.get("context_state", {}).get("short_term_memory", {})
            recent_summary = short_mem.get("recent_chapter_summary", item.get("context_state", {}).get("plot_summary", ""))
            recent_tail = short_mem.get("recent_tail_excerpt", "")

            # Retain ALL rich structured fields of dataset_generator
            enriched = dict(item)
            enriched["full_training_prompt"] = prompt
            enriched["prompt"] = prompt
            enriched["target_words"] = target_words
            enriched["context_summary"] = recent_summary
            enriched["prev_chapter_tail"] = recent_tail
            records.append(enriched)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Successfully converted {len(records)} pilot prompts to {OUTPUT_FILE}")


if __name__ == "__main__":
    convert()

