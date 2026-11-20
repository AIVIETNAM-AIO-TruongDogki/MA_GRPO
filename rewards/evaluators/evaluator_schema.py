"""Strict JSON parsing, clamping, and contradiction normalization for Phase 2 Evaluators."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def parse_first_json_object(raw_text: str) -> Dict[str, Any]:
    """Extracts the first valid JSON object from an LLM judge response."""
    cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", raw_text or "").strip()
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found in judge response.")
    parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(parsed, dict):
        raise ValueError("Judge JSON root must be an object.")
    return parsed


def clamp_unit_score(value: Any, default: float = 0.5) -> float:
    """Clamps a numeric score into [0.0, 1.0]; auto-scales 0-10 or 1-5 outputs if mistakenly emitted."""
    try:
        v = float(value)
    except Exception:
        return default
    if v > 1.0:
        if v <= 5.0:
            v = v / 5.0
        elif v <= 10.0:
            v = v / 10.0
        elif v <= 100.0:
            v = v / 100.0
    return round(max(0.0, min(1.0, v)), 4)


def normalize_contradictions(raw_items: Any) -> List[Dict[str, str]]:
    """Normalizes contradiction entries into `{severity: minor|major|critical, reason: ...}`."""
    if not isinstance(raw_items, list):
        return []
    out: List[Dict[str, str]] = []
    for item in raw_items:
        if isinstance(item, str) and item.strip():
            out.append({"severity": "minor", "reason": item.strip()})
            continue
        if not isinstance(item, dict):
            continue
        sev = str(item.get("severity", "minor")).strip().lower()
        if sev in {"fatal", "critical", "severe"}:
            sev = "critical"
        elif sev in {"major", "moderate"}:
            sev = "major"
        else:
            sev = "minor"
        reason = str(item.get("reason") or item.get("explanation") or item.get("description") or "").strip()
        if reason:
            out.append({"severity": sev, "reason": reason})
    return out


def normalize_evidence(raw_items: Any) -> List[Dict[str, str]]:
    if not isinstance(raw_items, list):
        return []
    out: List[Dict[str, str]] = []
    for item in raw_items:
        if isinstance(item, dict):
            out.append(
                {
                    "criterion": str(item.get("criterion", "")),
                    "text_span": str(item.get("text_span") or item.get("span") or ""),
                    "reason": str(item.get("reason", "")),
                }
            )
    return out
