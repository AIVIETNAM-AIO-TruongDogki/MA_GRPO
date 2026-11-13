"""
Two-Level Benchmark & Two-Track Evaluation Suite for Macro-Planning Ablations.

Implements all metrics & dataset loaders in the research proposal:
  - Track 1 (Planning Capacity via VR-CLI Next-Chapter Prediction):
      LONGSTORY Test Split / BookSum / PG-19 -> Group A Metrics:
        S_cov (Premise Coverage), S_prog (NLI Entailment Progression),
        S_causal (DAG Causal Coherence), S_comp (Conflict Completeness),
        S_arc (Freytag 5-Stage Story Arc Alignment),
        I_pi (Frozen-Model VR-CLI PPL Improvement), Planning Token Cost & Latency.
  - Track 2 (Writing Utility & Sufficiency via Open-Ended Story Generation):
      TELL ME A STORY / MoPS / WritingPrompts -> Group B Metrics:
        Total Words L, Length Score Sl (%), Early Termination Rate (%),
        S_adh (DOC NLI Plan Adherence across chunked chapters),
        Distinct-3/4 (%), Shannon Entropy & Sentence Burstiness,
        5-Dimension Rubric & C_weighted (3*Fatal + 2*Moderate + 1*Minor).
  - Perturbation Delta Engine (Clean - Perturbed) & H1-H4 Hypothesis Auditor.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.planning import (
    STORY_ARC_STAGES,
    compute_story_arc_alignment,
    extract_premise_keywords_and_characters,
)
from src.rewards import strip_think_tags


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent


# =====================================================================
# 1. TWO-TRACK DATASET LOADERS & PREMISE NORMALIZERS
# =====================================================================
def strip_length_constraints_from_premise(premise: str) -> str:
    """
    Remove conflicting short word-count instructions (e.g. 'between 1300-1400 words',
    'Make sure the word count is between 1,000 and 1,100.', 'Write a 700 word short story')
    from benchmark prompts like TELL ME A STORY while preserving POV, characters,
    setting, and plot directives for 8k-12k experiments.
    """
    if not premise:
        return ""
    text = premise.replace("\u00a0", " ").strip()
    num_range = r"[\d,]+(?:\s*(?:and|to|–|-)\s*[\d,]+)?"

    # 1. Standalone full sentences solely specifying word count / length
    text = re.sub(
        r"(?:^|\s)(?:Make sure |Ensure |Keep )?(?:the )?(?:story(?:'s)? )?(?:word\s+count|story\s+length|length)\s+(?:is|should\s+be|must\s+be)?\s*(?:at\s+least|around|about|over|under|less\s+than|more\s+than|roughly|between)?[^.]*?[\d,]+[^.]*?\.",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"(?:^|\s)(?:(?:The|This)\s+)?story(?:\s+length)?\s+(?:should|must)\s+(?:be\s+)?(?:(?:at\s+least|around|about|over|under|less\s+than|more\s+than|roughly|approximately|between)\s+)?{num_range}\s*words?(?:\s+(?:long|in\s+length))?\.",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?:^|\s)(?:(?:The|This)\s+)?story\s+(?:should|must)\s+be\s+(?:at\s+least|around|about|over|between)[^.]*?\bwords?(?:\s+(?:long|in\s+length))?\.",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # 2. Opening phrases:
    # - "Write between X-Y words telling a story ..." -> "Write a story ..."
    text = re.sub(
        rf"\bWrite\s+between\s+{num_range}\s*words\s+telling\s+a\s+story\b",
        "Write a story",
        text,
        flags=re.IGNORECASE,
    )
    # - "Write (me) a (less than) 700 / 1500 - 1530 word (epistolary/short) story ..."
    text = re.sub(
        rf"\bWrite\s+(me\s+)?an?\s+(?:(?:less|more)\s+than\s+|over\s+|under\s+|roughly\s+|around\s+|about\s+)?{num_range}[-\s]+words?\s+(?=(?:[a-z]+\s+){{0,2}}story\b)",
        lambda m: f"Write {m.group(1) or ''}a ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\ba\s+(?=[aeiouAEIOU][a-z]+\s+(?:[a-z]+\s+)?story\b)", "an ", text)
    # - "Write a story of/between X-Y words ..." or "Write a story, X to Y words, ..."
    text = re.sub(
        rf"\bWrite\s+(me\s+)?a\s+story(?:\s*,?\s*(?:of|between|around|about|approximately|roughly)?\s*{num_range}\s*words(?:\s+long)?\s*,?|\s+that\s+is\s+(?:roughly\s+|less\s+than\s+|more\s+than\s+|around\s+|about\s+|over\s+|at\s+least\s+)?{num_range}\s*words(?:\s+long)?\s*,?)",
        lambda m: f"Write {m.group(1) or ''}a story",
        text,
        flags=re.IGNORECASE,
    )

    # 3. Embedded clauses inside mixed sentences:
    # "... must be between X and Y words and written in ..." -> "... must be written in ..."
    text = re.sub(
        rf"\bbetween\s+{num_range}\s*words\s+and\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf",?\s*\b(?:around|about|approximately|roughly|at\s+least|less\s+than|over|between\s+[\d,]+\s*(?:and|to|–|-)|between)\s*{num_range}\s*words(?:\s+(?:long|in\s+length))?\s*,?",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    text = re.sub(r"\bstory,\s+(?:about|from)\b", lambda m: "story " + m.group(0).split()[-1], text, flags=re.IGNORECASE)
    return text.strip()


def load_tell_me_a_story_premises(
    data_path: Optional[str] = None,
    max_samples: int = 30,
    target_words: int = 10000,
    strip_length_constraints: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load open-ended story premises for Track 2 (TELL ME A STORY / MoPS / WritingPrompts).
    By default inspects Dataset/tell_me_a_story_data/tell-me-a-story-test.jsonl.
    """
    candidates: List[Path] = []
    if data_path:
        candidates.append(Path(data_path))
    candidates.extend([
        WORKSPACE_ROOT / "Dataset" / "tell_me_a_story_data" / "tell-me-a-story-test.jsonl",
        PROJECT_ROOT / "data" / "tell-me-a-story-test.jsonl",
        WORKSPACE_ROOT / "Dataset" / "tell_me_a_story_data" / "tell-me-a-story-validation.jsonl",
    ])

    resolved: Optional[Path] = next((c for c in candidates if c.exists()), None)
    records: List[Dict[str, Any]] = []

    if resolved is not None:
        with open(resolved, "r", encoding="utf-8-sig") as f:
            for idx, line in enumerate(f):
                if len(records) >= max_samples:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                raw_prompt = (
                    row.get("inputs")
                    or row.get("premise")
                    or row.get("prompt")
                    or ""
                ).strip()
                if not raw_prompt:
                    continue
                cleaned_prompt = (
                    strip_length_constraints_from_premise(raw_prompt)
                    if strip_length_constraints
                    else raw_prompt
                )
                records.append({
                    "id": str(row.get("example_id") or row.get("id") or f"premise_{idx:03d}"),
                    "source": resolved.name,
                    "genre": str(row.get("genre", "Creative Fiction")),
                    "raw_premise": raw_prompt,
                    "premise": cleaned_prompt,
                    "target_words": int(row.get("target_words", target_words)),
                    "gold_short_ref": row.get("targets", ""),
                })
        return records

    # Fallback to CANONICAL_20_PREMISES in dataset_generator if tell-me-a-story files are absent
    from src.dataset_generator import CANONICAL_20_PREMISES

    for item in CANONICAL_20_PREMISES[:max_samples]:
        records.append({
            "id": item["id"],
            "source": "CANONICAL_20_PREMISES",
            "genre": item.get("genre", "Speculative Fiction"),
            "raw_premise": item["premise"],
            "premise": item["premise"],
            "target_words": target_words,
            "gold_short_ref": "",
        })
    return records


def load_longstory_next_chapter_samples(
    data_path: Optional[str] = None,
    max_samples: int = 50,
    min_context_chapters: int = 1,
    allow_synthetic_demo: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load multi-chapter human-written novels for Track 1 (VR-CLI Next-Chapter Prediction).
    Supports LONGSTORY / BookSum / PG-19 JSON or JSONL schemas:
      Schema A: {"id": ..., "premise": ..., "chapters": ["ch1 text...", "ch2 text...", ...]}
                (or chapters as list of dicts with "text"/"content"/"story")
      Schema B: {"id": ..., "context_chapters": [...], "gold_chapter": "..."}
    Each sample exposes:
      (sample_id, premise, history_chapters C_{1..i-1}, history_text x, gold_chapter y_gold C_i).
    """
    if data_path:
        candidates = [Path(data_path)]
    else:
        candidates = [
            WORKSPACE_ROOT / "Dataset" / "longstory_test.jsonl",
            WORKSPACE_ROOT / "Dataset" / "LONGSTORY" / "test.jsonl",
            PROJECT_ROOT / "data" / "longstory_test.jsonl",
            PROJECT_ROOT / "data" / "booksum_chapters.jsonl",
            WORKSPACE_ROOT / "Dataset" / "booksum_chapters.jsonl",
        ]

    resolved: Optional[Path] = next((c for c in candidates if c.exists()), None)
    samples: List[Dict[str, Any]] = []

    if resolved is not None:
        raw_rows: List[Dict[str, Any]] = []
        if resolved.suffix.lower() == ".json":
            loaded = json.loads(resolved.read_text(encoding="utf-8-sig"))
            raw_rows = loaded if isinstance(loaded, list) else loaded.get("data", [])
        else:
            with open(resolved, "r", encoding="utf-8-sig") as f:
                for line in f:
                    if line.strip():
                        raw_rows.append(json.loads(line))

        for book_idx, row in enumerate(raw_rows):
            if len(samples) >= max_samples:
                break
            book_id = str(row.get("id") or row.get("book_id") or f"book_{book_idx:03d}")
            premise = str(row.get("premise") or row.get("outline") or row.get("summary") or "")

            # Schema B: Pre-split next-chapter pair
            if "gold_chapter" in row and ("history_text" in row or "context_chapters" in row):
                hist_list = row.get("context_chapters") or [row.get("history_text", "")]
                hist_text = "\n\n".join(str(c) for c in hist_list if c)
                gold_text = str(row["gold_chapter"]).strip()
                if hist_text and gold_text:
                    samples.append({
                        "sample_id": f"{book_id}_next",
                        "source": resolved.name,
                        "premise": premise,
                        "history_chapters": hist_list,
                        "history_text": hist_text,
                        "gold_chapter": gold_text,
                        "chapter_index": int(row.get("chapter_index", len(hist_list) + 1)),
                    })
                continue

            # Schema A: Full chapter sequence
            raw_chapters = row.get("chapters") or row.get("story") or []
            if isinstance(raw_chapters, str):
                raw_chapters = [p.strip() for p in re.split(r"\n\s*Chapter \d+[^\n]*\n", raw_chapters) if len(p.split()) > 80]

            norm_chapters: List[str] = []
            for ch in raw_chapters:
                if isinstance(ch, dict):
                    txt = str(ch.get("text") or ch.get("content") or ch.get("prose") or "").strip()
                else:
                    txt = str(ch).strip()
                if txt:
                    norm_chapters.append(txt)

            # Create next-chapter prediction samples starting at chapter index >= min_context_chapters
            for i in range(min_context_chapters, len(norm_chapters)):
                if len(samples) >= max_samples:
                    break
                hist = norm_chapters[:i]
                samples.append({
                    "sample_id": f"{book_id}_ch{i + 1}",
                    "source": resolved.name,
                    "premise": premise,
                    "history_chapters": hist,
                    "history_text": "\n\n".join(hist),
                    "gold_chapter": norm_chapters[i],
                    "chapter_index": i + 1,
                })
        return samples

    if not allow_synthetic_demo:
        checked = ", ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            "No LONGSTORY / BookSum / PG-19 chapter dataset file found for Track 1 VR-CLI. "
            f"Checked: [{checked}]. Pass `--track1-data <path.jsonl>` or use `--backend mock` for smoke testing."
        )

    # Synthetic multi-chapter demo data ONLY for --backend mock / unit tests
    demo_books = [
        {
            "sample_id": "demo_longstory_01_ch2",
            "source": "synthetic_demo",
            "premise": "Archivist Mara uncovers a submerged brass clockwork beneath Venice that records memories.",
            "history_chapters": [
                "Chapter 1: During the autumn flood, Mara descended into the inundated cellar below San Marco. Her lantern caught the gleam of rotating brass gears welded directly into Roman bedrock. Beside the turbine lay a waterproof bronze cylinder etched with her grandfather's initials and an unfinished warning about the midnight tide.",
            ],
            "history_text": (
                "Chapter 1: During the autumn flood, Mara descended into the inundated cellar below San Marco. "
                "Her lantern caught the gleam of rotating brass gears welded directly into Roman bedrock. "
                "Beside the turbine lay a waterproof bronze cylinder etched with her grandfather's initials and an unfinished warning about the midnight tide."
            ),
            "gold_chapter": (
                "Chapter 2: Mara carried the bronze cylinder to the dry restoration workshop before the bells struck ten. "
                "Inside the sealed tube rested perforated copper scrolls paired with handwritten tide tables. "
                "As she aligned the first scroll beneath a magnifier, footsteps paused outside the shuttered door, confirming someone else from the Venetian Maritime Council was already tracking the missing mechanism."
            ),
            "chapter_index": 2,
        },
        {
            "sample_id": "demo_longstory_02_ch2",
            "source": "synthetic_demo",
            "premise": "Astronomer Tariq intercepts an ancient repeating beacon orbiting Proxima Centauri.",
            "history_chapters": [
                "Chapter 1: In the high Atacama plateau, Tariq isolated a harmonic pulse hidden inside solar wind interference. The modulation repeated every forty-two minutes with prime-number intervals stamped in copper-band radio.",
            ],
            "history_text": (
                "Chapter 1: In the high Atacama plateau, Tariq isolated a harmonic pulse hidden inside solar wind interference. "
                "The modulation repeated every forty-two minutes with prime-number intervals stamped in copper-band radio."
            ),
            "gold_chapter": (
                "Chapter 2: At dawn, station director Helena ordered the primary dish pointed away from the beacon to avoid telemetry scrutiny. "
                "Refusing to lose the lock, Tariq secretly rerouted the secondary interferometer array to log the phase drift. "
                "The decoded spectrogram revealed coordinates pointing not to deep space, but to a buried relay beneath the Chilean salt flats."
            ),
            "chapter_index": 2,
        },
    ]
    return demo_books[:max_samples]


# =====================================================================
# 2. NLI SCORER (Used by S_prog, S_causal edge verification, DOC NLI S_adh)
# =====================================================================
class NLIScorer:
    """
    Wrapper around HuggingFace NLI SequenceClassification models
    (e.g. MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli or facebook/bart-large-mnli).
    Computes P(Entailment | premise, hypothesis) and P(Contradiction | premise, hypothesis).
    """

    def __init__(
        self,
        model_id: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        use_mock: bool = False,
        max_length: int = 512,
    ):
        self.model_id = model_id
        self.use_mock = use_mock
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.label_map: Dict[str, int] = {}
        self.backend_name = "mock_heuristic" if use_mock else model_id

    def load(self) -> None:
        if self.use_mock or self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
            self.model.eval()

            id2label = getattr(self.model.config, "id2label", {})
            for idx, lbl in id2label.items():
                low = str(lbl).lower()
                if "entail" in low:
                    self.label_map["entailment"] = int(idx)
                elif "contradict" in low:
                    self.label_map["contradiction"] = int(idx)
                elif "neutral" in low:
                    self.label_map["neutral"] = int(idx)
            # Standard fallback mapping if labels are generic LABEL_0/1/2
            self.label_map.setdefault("entailment", 0)
            self.label_map.setdefault("neutral", 1)
            self.label_map.setdefault("contradiction", 2)
        except Exception as e:
            print(f"[NLIScorer] Warning: failed loading {self.model_id} ({e}). Using heuristic fallback.")
            self.use_mock = True
            self.backend_name = "mock_heuristic"

    def unload(self) -> None:
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        self.model = None
        self.tokenizer = None
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def predict_probs(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Return {"entailment": p_e, "neutral": p_n, "contradiction": p_c}."""
        if not premise or not hypothesis:
            return {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}

        if not self.use_mock and self.model is None:
            self.load()

        if self.use_mock or self.model is None or self.tokenizer is None:
            return self._heuristic_nli(premise, hypothesis)

        import torch

        dev = next(self.model.parameters()).device
        enc = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(dev)
        with torch.no_grad():
            logits = self.model(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

        p_ent = float(probs[self.label_map.get("entailment", 0)])
        p_neu = float(probs[self.label_map.get("neutral", 1)]) if self.label_map.get("neutral", 1) < len(probs) else 0.0
        p_con = float(probs[self.label_map.get("contradiction", 2)]) if self.label_map.get("contradiction", 2) < len(probs) else 0.0
        return {
            "entailment": round(p_ent, 4),
            "neutral": round(p_neu, 4),
            "contradiction": round(p_con, 4),
        }

    @staticmethod
    def _heuristic_nli(premise: str, hypothesis: str) -> Dict[str, float]:
        """Deterministic lexical + factual-inversion proxy ONLY for mock/offline testing."""
        conflict_markers = {
            "dead", "died", "fatal", "desert", "vanished", "medieval",
            "shepherds", "banquet", "heatstroke", "unmotivated",
        }
        p_lower_words = set(re.findall(r"\b\w+\b", premise.lower()))
        h_lower_words = set(re.findall(r"\b\w+\b", hypothesis.lower()))
        p_tokens = set(re.findall(r"\b\w{4,}\b", premise.lower())) - conflict_markers
        h_tokens = set(re.findall(r"\b\w{4,}\b", hypothesis.lower())) - conflict_markers
        if not h_tokens or not p_tokens:
            return {"entailment": 0.05, "neutral": 0.90, "contradiction": 0.05}

        overlap = len(p_tokens & h_tokens) / max(1, len(h_tokens))
        p_has_conflict = bool(p_lower_words & conflict_markers)
        h_has_conflict = bool(h_lower_words & conflict_markers)

        if overlap > 0.35 and (p_has_conflict != h_has_conflict):
            p_con = round(min(0.95, 0.55 + overlap * 0.4), 4)
            p_ent = 0.05
            p_neu = round(max(0.0, 1.0 - p_con - p_ent), 4)
            return {"entailment": p_ent, "neutral": p_neu, "contradiction": p_con}

        p_ent = round(min(0.95, overlap), 4)
        p_con = 0.02
        p_neu = round(max(0.0, 1.0 - p_ent - p_con), 4)
        return {"entailment": p_ent, "neutral": p_neu, "contradiction": p_con}


# =====================================================================
# 3. GROUP A METRICS: MACRO PLANNING INTRINSIC QUALITY & VR-CLI
# =====================================================================
def _extract_sequence_units_for_plan(blueprint: Dict[str, Any]) -> List[str]:
    """Extract ordered textual units E_1..E_K from any Condition B/C/D/E plan."""
    events = blueprint.get("event_graph", {}).get("events", [])
    if events:
        units = []
        for ev in events:
            units.append(
                f"{ev.get('title', '')}. Setting: {ev.get('setting', '')}. "
                f"Action: {ev.get('action', '')}. Conflict: {ev.get('conflict', '')}. "
                f"Turn: {ev.get('plot_twist', '')}."
            )
        return units

    chapters = blueprint.get("chapters", [])
    if chapters:
        return [
            f"{ch.get('chapter_title', '')}: {ch.get('chapter_goal', '')}"
            for ch in chapters
        ]

    unstructured = blueprint.get("unstructured_plan", "")
    if unstructured:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", unstructured) if s.strip()]
        return sents
    return []


def compute_premise_coverage(premise: str, blueprint: Dict[str, Any]) -> float:
    """
    S_cov: Fraction of premise salient entities (proper names) & core keywords
    preserved inside the generated plan / Event Graph.
    """
    if not premise or not blueprint:
        return 0.0
    chars, keywords = extract_premise_keywords_and_characters(premise)
    salient = set(w.lower() for w in (chars + keywords) if w not in {"Protagonist", "Ally", "Mentor", "Antagonist"})
    if not salient:
        salient = set(re.findall(r"\b\w{5,}\b", premise.lower()))
    if not salient:
        return 1.0

    plan_blob = json.dumps(blueprint, ensure_ascii=False).lower()
    covered = sum(1 for tok in salient if tok in plan_blob)
    return round(covered / len(salient), 4)


def compute_nli_progression(blueprint: Dict[str, Any], nli_scorer: NLIScorer) -> Dict[str, Any]:
    """
    S_prog (NLI Entailment Formula):
      S_prog = 1.0 - (1 / (K - 1)) * sum_{i=1}^{K-1} P_NLI(Entailment | E_i, E_{i+1})
    """
    units = _extract_sequence_units_for_plan(blueprint)
    if len(units) < 2:
        return {"S_prog": 0.0, "entailment_probs": [], "num_units": len(units)}

    entail_probs: List[float] = []
    for i in range(len(units) - 1):
        res = nli_scorer.predict_probs(units[i], units[i + 1])
        entail_probs.append(res["entailment"])

    mean_entail = sum(entail_probs) / len(entail_probs)
    s_prog = round(max(0.0, min(1.0, 1.0 - mean_entail)), 4)
    return {
        "S_prog": s_prog,
        "mean_entailment": round(mean_entail, 4),
        "entailment_probs": entail_probs,
        "num_units": len(units),
    }


def compute_causal_coherence(blueprint: Dict[str, Any], nli_scorer: Optional[NLIScorer] = None) -> Dict[str, Any]:
    """
    S_causal: Evaluates Event Graph DAG structural connectivity, absence of forward/cyclic
    violations, and low NLI contradiction across causal parent->child edges.
    Conditions without an Event Graph (A, B, C) receive structural connectivity = 0.0.
    """
    events = blueprint.get("event_graph", {}).get("events", [])
    if not events:
        return {
            "S_causal": 0.0,
            "dag_connectivity": 0.0,
            "edge_consistency": 0.0,
            "isolated_nodes": [],
            "violations": ["No Event Graph structure present."],
        }

    id_to_idx = {ev.get("event_id", f"E{i+1}"): i for i, ev in enumerate(events)}
    id_to_ev = {ev.get("event_id", f"E{i+1}"): ev for i, ev in enumerate(events)}

    if len(events) == 1:
        return {
            "S_causal": 1.0,
            "dag_connectivity": 1.0,
            "edge_consistency": 1.0,
            "isolated_nodes": [],
            "violations": [],
        }

    connected_count = 0
    isolated: List[str] = []
    violations: List[str] = []
    edge_consistencies: List[float] = []

    for idx, ev in enumerate(events):
        eid = ev.get("event_id", f"E{idx+1}")
        deps = ev.get("depends_on", [])
        if idx == 0:
            # Root exposition event has no prior prerequisite needed
            connected_count += 1
            continue

        valid_prior_deps = [d for d in deps if d in id_to_idx and id_to_idx[d] < idx]
        invalid_deps = [d for d in deps if d not in id_to_idx or id_to_idx[d] >= idx]

        if invalid_deps:
            violations.append(f"{eid} has invalid/cyclic depends_on={invalid_deps}")

        if valid_prior_deps:
            connected_count += 1
            if nli_scorer is not None:
                for p_id in valid_prior_deps:
                    p_ev = id_to_ev[p_id]
                    p_txt = f"{p_ev.get('action', '')} {p_ev.get('plot_twist', '')}"
                    c_txt = f"{ev.get('action', '')} {ev.get('conflict', '')}"
                    p_nli = nli_scorer.predict_probs(p_txt, c_txt)
                    edge_consistencies.append(1.0 - p_nli["contradiction"])
        else:
            isolated.append(eid)
            violations.append(f"{eid} is isolated (no valid causal parent)")

    dag_connectivity = connected_count / len(events)
    edge_consistency = (
        sum(edge_consistencies) / len(edge_consistencies)
        if edge_consistencies
        else (1.0 if not violations else 0.7)
    )
    s_causal = round(0.6 * dag_connectivity + 0.4 * edge_consistency, 4)
    return {
        "S_causal": s_causal,
        "dag_connectivity": round(dag_connectivity, 4),
        "edge_consistency": round(edge_consistency, 4),
        "isolated_nodes": isolated,
        "violations": violations,
    }


def compute_conflict_completeness(blueprint: Dict[str, Any]) -> float:
    """
    S_comp: Checks whether opening conflicts are matched by Resolution/Falling Action
    stages and explicit closure cues in the latter third of the plan.
    """
    events = blueprint.get("event_graph", {}).get("events", [])
    chapters = blueprint.get("chapters", [])
    closure_keywords = {
        "resolve", "resolved", "resolution", "conclude", "concluded",
        "equilibrium", "aftermath", "legacy", "reconcile", "closure", "end",
    }

    if events:
        has_opening_conflict = any(bool(ev.get("conflict", "").strip()) for ev in events[: max(1, len(events) // 2)])
        tail_events = events[max(0, (2 * len(events)) // 3) :]
        has_resolution_stage = any(
            ev.get("story_arc") in {"Falling Action", "Resolution"}
            or any(kw in f"{ev.get('action', '')} {ev.get('plot_twist', '')} {ev.get('title', '')}".lower() for kw in closure_keywords)
            for ev in tail_events
        )
        if has_opening_conflict and has_resolution_stage:
            return 1.0
        if has_opening_conflict or has_resolution_stage:
            return 0.5
        return 0.0

    if chapters:
        tail_txt = f"{chapters[-1].get('chapter_goal', '')} {chapters[-1].get('exit_state', '')}".lower()
        return 0.8 if any(kw in tail_txt for kw in closure_keywords) else 0.35

    unstructured = blueprint.get("unstructured_plan", "").lower()
    return 0.5 if any(kw in unstructured for kw in closure_keywords) else 0.25


class VRCLIPerplexityEvaluator:
    """
    Evaluates Plan Information Gain / Predictive Utility (I_pi) on Track 1 (Next-Chapter Prediction):
      I_pi(x, y_gold, p) = ((PPL_frozen(y_gold | x) - PPL_frozen(y_gold | x, p)) / PPL_frozen(y_gold | x)) * 100
    Strictly enforces frozen context window budgeting so base context `x` is truncated identically
    in both baseline and planned runs, and computes cross-entropy loss strictly over `y_gold` tokens.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        max_context_tokens: int = 1024,
        max_plan_tokens: int = 384,
        max_target_tokens: int = 512,
        use_mock: bool = False,
    ):
        self.model_id = model_id
        self.max_context_tokens = max_context_tokens
        self.max_plan_tokens = max_plan_tokens
        self.max_target_tokens = max_target_tokens
        self.use_mock = use_mock
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        if self.use_mock or self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            use_cuda = torch.cuda.is_available()
            major = torch.cuda.get_device_capability(0)[0] if use_cuda else 0
            dtype = torch.bfloat16 if (use_cuda and major >= 8) else (torch.float16 if use_cuda else torch.float32)

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map="auto" if use_cuda else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self.model.eval()
        except Exception as e:
            print(f"[VRCLIPerplexityEvaluator] Warning: could not load frozen LM ({e}). Using mock fallback.")
            self.use_mock = True

    def unload(self) -> None:
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        self.model = None
        self.tokenizer = None
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _compute_conditional_ppl(self, prefix_ids: List[int], target_ids: List[int]) -> float:
        import torch

        input_list = prefix_ids + target_ids
        label_list = [-100] * len(prefix_ids) + target_ids
        dev = next(self.model.parameters()).device
        input_tensor = torch.tensor([input_list], dtype=torch.long, device=dev)
        label_tensor = torch.tensor([label_list], dtype=torch.long, device=dev)
        with torch.no_grad():
            loss = float(self.model(input_ids=input_tensor, labels=label_tensor).loss.item())
        return round(float(math.exp(min(loss, 12.0))), 4)

    def evaluate_information_gain(self, history_text: str, plan_text: str, gold_chapter: str) -> Dict[str, Any]:
        """Compute (PPL_base, PPL_planned, I_pi %) using fixed context token budget."""
        if not gold_chapter:
            return {"ppl_base": 0.0, "ppl_planned": 0.0, "I_pi": 0.0}

        if not self.use_mock and self.model is None:
            self.load()

        if self.use_mock or self.model is None or self.tokenizer is None:
            return self._mock_vrcli(history_text, plan_text, gold_chapter)

        # Keep trailing max_context_tokens of preceding chapter history x identical across both conditions
        x_ids = self.tokenizer.encode(f"[PRECEDING CHAPTERS]:\n{history_text}\n", add_special_tokens=False)
        if len(x_ids) > self.max_context_tokens:
            x_ids = x_ids[-self.max_context_tokens :]

        p_ids = self.tokenizer.encode(f"\n[NEXT CHAPTER PLAN]:\n{plan_text}\n", add_special_tokens=False)
        if len(p_ids) > self.max_plan_tokens:
            p_ids = p_ids[: self.max_plan_tokens]

        sep_ids = self.tokenizer.encode("\n[NEXT CHAPTER]:\n", add_special_tokens=False)
        y_ids = self.tokenizer.encode(gold_chapter, add_special_tokens=False)[: self.max_target_tokens]

        ppl_base = self._compute_conditional_ppl(x_ids + sep_ids, y_ids)
        ppl_planned = self._compute_conditional_ppl(x_ids + p_ids + sep_ids, y_ids)
        i_pi = ((ppl_base - ppl_planned) / ppl_base) * 100.0 if ppl_base > 0 else 0.0

        return {
            "ppl_base": round(ppl_base, 4),
            "ppl_planned": round(ppl_planned, 4),
            "I_pi": round(i_pi, 2),
            "backend": self.model_id,
        }

    @staticmethod
    def _mock_vrcli(history_text: str, plan_text: str, gold_chapter: str) -> Dict[str, Any]:
        """Deterministic proxy for local offline testing ONLY."""
        gold_kws = set(re.findall(r"\b\w{5,}\b", gold_chapter.lower()))
        plan_kws = set(re.findall(r"\b\w{5,}\b", plan_text.lower()))
        overlap = len(gold_kws & plan_kws) / max(1, len(gold_kws))
        # Also reward structural richness slightly beyond bare text
        has_structured_fields = all(k in plan_text for k in ["Setting:", "Action:", "Conflict:"])
        bonus = 0.08 if has_structured_fields else 0.02
        ppl_base = 12.50
        reduction = min(0.35, 0.05 + overlap * 0.45 + bonus)
        ppl_planned = round(ppl_base * (1.0 - reduction), 4)
        i_pi = round(((ppl_base - ppl_planned) / ppl_base) * 100.0, 2)
        return {
            "ppl_base": ppl_base,
            "ppl_planned": ppl_planned,
            "I_pi": i_pi,
            "backend": "mock_proxy",
        }


def evaluate_macro_plan_group_a(
    premise: str,
    blueprint: Dict[str, Any],
    nli_scorer: NLIScorer,
) -> Dict[str, Any]:
    """Compute all Group A Intrinsic Planning Metrics on any blueprint (Conditions A-E)."""
    events = blueprint.get("event_graph", {}).get("events", [])
    s_cov = compute_premise_coverage(premise, blueprint)
    prog_res = compute_nli_progression(blueprint, nli_scorer)
    causal_res = compute_causal_coherence(blueprint, nli_scorer=nli_scorer)
    s_comp = compute_conflict_completeness(blueprint)
    arc_res = compute_story_arc_alignment(events)
    planning_tokens = blueprint.get("planning_metadata", {}).get("planning_tokens", 0)
    planning_latency = blueprint.get("planning_metadata", {}).get("planning_latency_s", 0.0)

    return {
        "S_cov": s_cov,
        "S_prog": prog_res["S_prog"],
        "S_causal": causal_res["S_causal"],
        "dag_connectivity": causal_res["dag_connectivity"],
        "S_comp": s_comp,
        "S_arc": arc_res["story_arc_score"],
        "arc_stage_coverage": arc_res["stage_coverage"],
        "arc_order_monotonicity": arc_res["order_monotonicity"],
        "planning_tokens": planning_tokens,
        "planning_latency_s": planning_latency,
    }


# =====================================================================
# 4. GROUP B METRICS: STORY UTILITY, DOC NLI ADHERENCE & RUBRIC
# =====================================================================
def evaluate_length_and_termination(full_text: str, target_words: int) -> Dict[str, Any]:
    """Compute actual word count L, length compliance Sl (%), and early termination flag (<50% target)."""
    clean = strip_think_tags(full_text)
    actual_words = len(clean.split())
    if target_words <= 0:
        sl = 100.0
    else:
        sl = max(0.0, min(100.0, round(100.0 * (1.0 - abs(actual_words - target_words) / target_words), 2)))
    early_terminated = bool(target_words > 0 and actual_words < 0.5 * target_words)
    return {
        "actual_words": actual_words,
        "target_words": target_words,
        "length_score_sl": sl,
        "early_terminated": early_terminated,
    }


def _chunk_text_words(text: str, chunk_size: int = 180, stride: int = 110) -> List[str]:
    words = strip_think_tags(text).split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks = []
    for i in range(0, len(words), stride):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def evaluate_plan_adherence_doc_nli(
    chapters: List[str],
    blueprint: Dict[str, Any],
    nli_scorer: NLIScorer,
    premise: str = "",
    entail_threshold: float = 0.45,
) -> Dict[str, Any]:
    """
    DOC NLI Plan Adherence (S_adh):
      S_adh = (# planned event/chapter goals entailed across matching chapter chunks) / total_goals
    Chunks long chapters cleanly to prevent NLI 512-token truncation loss.
    For Condition A (no plan), decomposes the original premise sentences as adherence targets.
    """
    planned_chapters = blueprint.get("chapters", [])
    claims_per_chapter: List[Tuple[str, List[str]]] = []

    if planned_chapters and len(chapters) == len(planned_chapters):
        for ch_text, ch_plan in zip(chapters, planned_chapters):
            goals: List[str] = []
            if ch_plan.get("chapter_goal"):
                goals.append(str(ch_plan["chapter_goal"]))
            for beat in ch_plan.get("required_beats", []):
                if beat and beat not in goals:
                    goals.append(str(beat))
            if not goals and ch_plan.get("scene_outline"):
                goals.extend(str(s) for s in ch_plan["scene_outline"][:2])
            claims_per_chapter.append((ch_text, goals))
    else:
        # Condition A or single-blob comparison: evaluate premise sentences against full text
        full_text = "\n\n".join(chapters)
        premise_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", premise) if len(s.split()) >= 5]
        if not premise_sents and premise:
            premise_sents = [premise]
        claims_per_chapter.append((full_text, premise_sents))

    total_goals = 0
    adhered_count = 0
    max_probs: List[float] = []

    for ch_text, goals in claims_per_chapter:
        chunks = _chunk_text_words(ch_text)
        for goal in goals:
            total_goals += 1
            if not chunks:
                max_probs.append(0.0)
                continue
            best_p = max(nli_scorer.predict_probs(chunk, goal)["entailment"] for chunk in chunks)
            max_probs.append(best_p)
            if best_p >= entail_threshold:
                adhered_count += 1

    s_adh = round(adhered_count / max(1, total_goals), 4)
    mean_entail = round(sum(max_probs) / max(1, len(max_probs)), 4)
    return {
        "S_adh": s_adh,
        "mean_goal_entailment": mean_entail,
        "adhered_goals": adhered_count,
        "total_goals": total_goals,
    }


def evaluate_lexical_and_burstiness(full_text: str) -> Dict[str, Any]:
    """Compute Distinct-3 (%), Distinct-4 (%), Shannon Vocabulary Entropy, and Sentence Burstiness."""
    clean = strip_think_tags(full_text)
    words = clean.split()
    total_w = len(words)

    if total_w < 3:
        d3 = 100.0
    else:
        ng3 = set(zip(words, words[1:], words[2:]))
        d3 = round((len(ng3) / (total_w - 2)) * 100.0, 2)

    if total_w < 4:
        d4 = 100.0
    else:
        ng4 = set(zip(words, words[1:], words[2:], words[3:]))
        d4 = round((len(ng4) / (total_w - 3)) * 100.0, 2)

    lower_words = re.findall(r"\b\w+\b", clean.lower())
    if not lower_words:
        entropy = 0.0
    else:
        counts = Counter(lower_words)
        tot = len(lower_words)
        entropy = round(-sum((c / tot) * math.log2(c / tot) for c in counts.values()), 4)

    sentences = [s.strip() for s in re.split(r"[.!?]+", clean) if s.strip()]
    sent_lens = [len(s.split()) for s in sentences]
    if len(sent_lens) <= 1:
        burstiness = 0.0
    else:
        mean_l = sum(sent_lens) / len(sent_lens)
        var = sum((l - mean_l) ** 2 for l in sent_lens) / len(sent_lens)
        burstiness = round(math.sqrt(var), 2)

    return {
        "Distinct-3 (%)": d3,
        "Distinct-4 (%)": d4,
        "shannon_entropy": entropy,
        "burstiness_std": burstiness,
    }


class LLMJudge5DimEvaluator:
    """
    LLM-as-a-Judge Evaluator for the 5-Dimension Story Rubric & Severity-Weighted Inconsistency:
      - Dimensions (1-5): story_coherence (S_coh), plan_adherence, completeness, writing_quality, global_consistency
      - Severity Counts: n_fatal (*3), n_moderate (*2), n_minor (*1) -> C_weighted
      - Perturbation Behavior Diagnosis: Plan-Compliant vs Plan-Blind vs Coherent-Recovery
    Returns None values when no Judge LLM callback/API key is provided (never fabricates judge scores).
    """

    SYSTEM_PROMPT = """You are an expert literary critic evaluating an ultra-long AI-generated narrative.
Score the story strictly using JSON matching this schema:
{
  "story_coherence": 1-5,
  "plan_adherence": 1-5,
  "completeness": 1-5,
  "writing_quality": 1-5,
  "global_consistency": 1-5,
  "n_fatal_inconsistencies": int,
  "n_moderate_inconsistencies": int,
  "n_minor_inconsistencies": int,
  "perturbation_behavior": "Clean | Plan-Compliant | Plan-Blind | Coherent-Recovery",
  "rationale": "concise justification"
}
Severity rules:
- Fatal (weight 3): Resurrection without explanation, timeline collapse, contradictory identity.
- Moderate (weight 2): Sudden location/weapon/ability shift without transition.
- Minor (weight 1): Clothing, time-of-day, or small prop mismatch."""

    def __init__(self, judge_fn: Optional[Callable[[str, str], str]] = None, max_retries: int = 2):
        self.judge_fn = judge_fn
        self.max_retries = max(1, max_retries)

    @staticmethod
    def _empty_scores() -> Dict[str, Any]:
        return {
            "S_coh": None,
            "judge_plan_adherence": None,
            "judge_completeness": None,
            "judge_writing_quality": None,
            "judge_global_consistency": None,
            "C_weighted": None,
            "N_fatal": None,
            "N_moderate": None,
            "N_minor": None,
            "perturbation_behavior": None,
        }

    @staticmethod
    def _parse_judge_json(raw: str) -> Dict[str, Any]:
        cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", raw or "").strip()
        start = cleaned.find("{")
        if start == -1:
            raise ValueError("No JSON object found in judge output.")
        parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        if not isinstance(parsed, dict):
            raise ValueError("Judge output JSON root must be an object.")
        return parsed

    def evaluate(
        self,
        premise: str,
        plan_summary: str,
        story_text: str,
    ) -> Dict[str, Any]:
        if self.judge_fn is None:
            return self._empty_scores()

        # Sample head + middle + tail if text exceeds judge token comfort window while preserving arcs
        words = strip_think_tags(story_text).split()
        if len(words) > 6000:
            excerpt = (
                " ".join(words[:2000])
                + "\n\n[... MIDDLE SECTION ...]\n\n"
                + " ".join(words[len(words) // 2 - 1000 : len(words) // 2 + 1000])
                + "\n\n[... ENDING SECTION ...]\n\n"
                + " ".join(words[-2000:])
            )
        else:
            excerpt = " ".join(words)

        user_prompt = (
            f"[PREMISE]:\n{premise}\n\n"
            f"[ASSIGNED PLAN SUMMARY]:\n{plan_summary[:2500]}\n\n"
            f"[GENERATED STORY EXCERPT ({len(words)} total words)]:\n{excerpt}\n\n"
            "Respond strictly with ONE valid JSON object matching the schema."
        )

        parsed: Optional[Dict[str, Any]] = None
        last_err: Optional[Exception] = None
        for _ in range(self.max_retries):
            try:
                raw = self.judge_fn(self.SYSTEM_PROMPT, user_prompt)
                parsed = self._parse_judge_json(raw)
                break
            except Exception as exc:
                last_err = exc
                parsed = None

        if parsed is None:
            print(f"[LLMJudge5DimEvaluator] Warning: failed parsing judge JSON ({last_err}). Skipping sample.")
            return self._empty_scores()

        def _clamp_score(key: str) -> float:
            val = float(parsed.get(key, 0.0))
            return round(max(1.0, min(5.0, val)), 2)

        n_fatal = max(0, int(parsed.get("n_fatal_inconsistencies", 0)))
        n_mod = max(0, int(parsed.get("n_moderate_inconsistencies", 0)))
        n_min = max(0, int(parsed.get("n_minor_inconsistencies", 0)))
        c_weighted = 3 * n_fatal + 2 * n_mod + n_min

        return {
            "S_coh": _clamp_score("story_coherence"),
            "judge_plan_adherence": _clamp_score("plan_adherence"),
            "judge_completeness": _clamp_score("completeness"),
            "judge_writing_quality": _clamp_score("writing_quality"),
            "judge_global_consistency": _clamp_score("global_consistency"),
            "C_weighted": c_weighted,
            "N_fatal": n_fatal,
            "N_moderate": n_mod,
            "N_minor": n_min,
            "perturbation_behavior": parsed.get("perturbation_behavior"),
        }


# =====================================================================
# 5. PERTURBATION DELTA & H1-H4 HYPOTHESIS AUDITOR
# =====================================================================
def compute_perturbation_delta(
    clean_record: Dict[str, Any],
    perturbed_record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute Delta Score = Metric(Clean) - Metric(Perturbed).
    Higher positive delta_S_coh or delta_S_adh (or negative delta_C_weighted meaning
    more contradictions under perturbation) proves Writer sensitivity to the Plan.
    """
    def _diff(key: str) -> Optional[float]:
        a = clean_record.get(key)
        b = perturbed_record.get(key)
        if a is None or b is None:
            return None
        return round(float(a) - float(b), 4)

    delta_adh = _diff("S_adh")
    delta_coh = _diff("S_coh")
    delta_c = _diff("C_weighted")

    # Determine plan dependency verdict from NLI adherence / coherence drop
    if (delta_coh is not None and delta_coh >= 0.5) or (delta_adh is not None and delta_adh >= 0.15):
        diagnosis = "Plan-Sensitive (Depends deeply on Macro Plan)"
    elif delta_adh is not None and abs(delta_adh) < 0.05 and (delta_coh is None or abs(delta_coh) < 0.2):
        diagnosis = "Plan-Blind (Writer ignores plan mutations)"
    else:
        diagnosis = "Moderate Sensitivity"

    return {
        "perturbation_type": perturbed_record.get("perturbation_type", "unknown"),
        "delta_S_adh": delta_adh,
        "delta_S_coh": delta_coh,
        "delta_S_arc": _diff("S_arc"),
        "delta_S_causal": _diff("S_causal"),
        "delta_C_weighted": delta_c,
        "diagnosis": diagnosis,
    }


def audit_hypotheses_h1_to_h4(summary_by_condition: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate empirical evidence across Conditions A, B, C, D, E against Hypotheses H1-H4:
      H1: Partitioned planning (B/C/D/E) > Direct (A) on generated word length.
      H2: Macro Event Graph (D) > Unstructured (B) & Linear (C) on S_causal / S_arc / I_pi.
      H3: Macro-Only (D) achieves parity with Macro+Micro (E) while cutting >40% planning tokens.
      H4: Distinct-3 decreases in chunked/summarized conditions (C/D/E) vs free generation (A).
    """
    cond_a = summary_by_condition.get("A", {})
    cond_b = summary_by_condition.get("B", {})
    cond_c = summary_by_condition.get("C", {})
    cond_d = summary_by_condition.get("D", {})
    cond_e = summary_by_condition.get("E", {})

    audit: Dict[str, Any] = {}

    # H1
    if cond_a and cond_d:
        l_a = cond_a.get("actual_words", 0)
        l_d = cond_d.get("actual_words", 0)
        audit["H1_length_ceiling_broken"] = {
            "supported": l_d > l_a * 2.0 and l_d >= 8000,
            "L_A": l_a,
            "L_D": l_d,
        }

    # H2
    if cond_c and cond_d:
        sc_d = cond_d.get("S_causal", 0.0)
        sc_c = cond_c.get("S_causal", 0.0)
        arc_d = cond_d.get("S_arc", 0.0)
        audit["H2_macro_graph_advantage"] = {
            "supported": sc_d > sc_c and arc_d > 0.5,
            "S_causal_D": sc_d,
            "S_causal_C": sc_c,
            "S_arc_D": arc_d,
        }

    # H3
    if cond_d and cond_e:
        tok_d = cond_d.get("planning_tokens", 0)
        tok_e = cond_e.get("planning_tokens", 0)
        token_savings_pct = round(((tok_e - tok_d) / tok_e) * 100.0, 2) if tok_e > 0 else 0.0
        adh_d = cond_d.get("S_adh", 0.0)
        adh_e = cond_e.get("S_adh", 0.0)
        audit["H3_minimum_necessary_planning"] = {
            "supported": token_savings_pct >= 40.0 and abs(adh_d - adh_e) <= 0.10,
            "planning_tokens_D": tok_d,
            "planning_tokens_E": tok_e,
            "token_reduction_pct": token_savings_pct,
            "S_adh_D": adh_d,
            "S_adh_E": adh_e,
        }

    # H4
    if cond_a and cond_d:
        d3_a = cond_a.get("Distinct-3 (%)", 0.0)
        d3_d = cond_d.get("Distinct-3 (%)", 0.0)
        audit["H4_diversity_drop_justifying_rl"] = {
            "observed_drop": d3_d < d3_a,
            "Distinct3_A": d3_a,
            "Distinct3_D": d3_d,
        }

    return audit
