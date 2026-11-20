"""Human-Preference Calibration Utilities (Spearman, Kendall's tau, Pairwise Accuracy)."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

from experiments.benchmarks.outline_bench.judge_outline import compute_spearman_correlation


def compute_kendall_tau(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pure-Python Kendall's tau-b rank correlation."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom = ((concordant + discordant + ties_x) * (concordant + discordant + ties_y)) ** 0.5
    if denom == 0.0:
        return 0.0
    return round((concordant - discordant) / denom, 4)


def compute_human_alignment_report(
    evaluator_scores: Sequence[float],
    human_scores: Sequence[float],
    pairwise_judgments: Sequence[Tuple[float, float, str]] = (),
) -> Dict[str, float]:
    """
    Computes Spearman rho, Kendall's tau-b, and optional pairwise agreement accuracy
    against human-annotated validation chapters (e.g. 100-200 pairs).
    Each entry in `pairwise_judgments` is `(eval_score_a, eval_score_b, human_winner)` where
    `human_winner in {"A", "B"}`.
    """
    report: Dict[str, float] = {
        "spearman_rho": compute_spearman_correlation(evaluator_scores, human_scores),
        "kendall_tau": compute_kendall_tau(evaluator_scores, human_scores),
    }
    if pairwise_judgments:
        correct = 0
        total = 0
        for sa, sb, winner in pairwise_judgments:
            w = winner.strip().upper()
            if w not in {"A", "B"}:
                continue
            total += 1
            pred = "A" if sa > sb else ("B" if sb > sa else "TIE")
            if pred == w:
                correct += 1
        report["pairwise_accuracy"] = round(correct / max(1, total), 4)
    return report
