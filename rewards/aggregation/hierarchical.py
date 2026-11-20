"""
Hierarchical Reward Aggregator & Unified `Phase2RewardEngine` supporting EXP-00 .. EXP-05.

Formula (EXP-02 Main Proposed Config):
    R_quality = alpha_global * R_global + (1 - alpha_global) * R_local     (alpha_global = 0.65)
    R_total   = R_quality - lambda_rep * P_rep - lambda_contra * P_contra - lambda_length * P_length - lambda_format * P_format
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rewards.base import (
    ConstraintPenalties,
    EvaluationInput,
    GlobalScores,
    LocalScores,
    RewardBreakdown,
)
from rewards.evaluators import (
    ConstraintEvaluator,
    GlobalChapterEvaluator,
    HolisticChapterEvaluator,
    LocalExpansionEvaluator,
    UnifiedPhase2Evaluator,
    build_judge_backend,
    verify_objective_constraints,
)
from rewards.evaluators.local_llm_backend import DEFAULT_LOCAL_32B_JUDGE_MODEL
from src.rewards import CompositeRewardEngine
from .arithmetic import (
    DEFAULT_GLOBAL_WEIGHTS,
    DEFAULT_LOCAL_WEIGHTS,
    aggregate_global_arithmetic,
    aggregate_local_arithmetic,
)
from .geometric import aggregate_global_geometric, aggregate_local_geometric


SUPPORTED_MODES = {
    "baseline_v2",       # EXP-00: Phase 1 baseline (src/rewards.py)
    "holistic",          # EXP-01: Single holistic judge score - P
    "hierarchical",      # EXP-02: 0.65 R_G + 0.35 R_L - P (Arithmetic)
    "local_only",        # EXP-03: R_L - P
    "global_only",       # EXP-04: R_G - P
    "hierarchical_geo",  # EXP-05: 0.65 R_G_geo + 0.35 R_L_geo - P (Geometric)
}


def load_reward_config(config_path: str | Path) -> Dict[str, Any]:
    """Loads a Phase 2 YAML or JSON reward config file."""
    p = Path(config_path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except ImportError:
        # Lightweight fallback parser if PyYAML is absent on minimal CPU env
        return _minimal_yaml_parse(text)


def _minimal_yaml_parse(text: str) -> Dict[str, Any]:
    """Minimal indentation-aware key: value YAML parser for CPU tests without PyYAML."""
    root: Dict[str, Any] = {}
    stack: List[tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if ":" not in line:
            continue
        key, val_str = line.strip().split(":", 1)
        key = key.strip()
        val_str = val_str.strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if not val_str:
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            if (val_str.startswith('"') and val_str.endswith('"')) or (val_str.startswith("'") and val_str.endswith("'")):
                val: Any = val_str[1:-1]
            elif val_str.lower() == "true":
                val = True
            elif val_str.lower() == "false":
                val = False
            elif val_str.lower() in {"null", "none"}:
                val = None
            else:
                try:
                    val = int(val_str)
                except ValueError:
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = val_str
            parent[key] = val
    return root


def aggregate_phase2_reward(
    mode: str,
    local_scores: Optional[LocalScores] = None,
    global_scores: Optional[GlobalScores] = None,
    holistic_score: Optional[float] = None,
    constraints: Optional[ConstraintPenalties] = None,
    alpha_global: float = 0.65,
    local_weights: Optional[Dict[str, float]] = None,
    global_weights: Optional[Dict[str, float]] = None,
    lambda_rep: float = 0.10,
    lambda_contra: float = 0.20,
    lambda_length: float = 0.10,
    lambda_format: float = 0.05,
    geo_epsilon: float = 0.05,
    clip_min: float = -0.5,
    clip_max: float = 1.0,
) -> RewardBreakdown:
    """Pure deterministic aggregator turning raw Evaluator outputs into a scalar GRPO reward."""
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported Phase 2 reward mode '{mode}'. Choose from {sorted(SUPPORTED_MODES)}.")

    use_geo = mode == "hierarchical_geo"
    r_local: Optional[float] = None
    r_global: Optional[float] = None

    if local_scores is not None:
        r_local = (
            aggregate_local_geometric(local_scores, local_weights, epsilon=geo_epsilon)
            if use_geo
            else aggregate_local_arithmetic(local_scores, local_weights)
        )
    if global_scores is not None:
        r_global = (
            aggregate_global_geometric(global_scores, global_weights, epsilon=geo_epsilon)
            if use_geo
            else aggregate_global_arithmetic(global_scores, global_weights)
        )

    if mode == "holistic":
        r_quality = float(holistic_score if holistic_score is not None else 0.5)
    elif mode == "local_only":
        r_quality = float(r_local if r_local is not None else 0.5)
    elif mode == "global_only":
        r_quality = float(r_global if r_global is not None else 0.5)
    else:
        rg = float(r_global if r_global is not None else 0.5)
        rl = float(r_local if r_local is not None else 0.5)
        r_quality = alpha_global * rg + (1.0 - alpha_global) * rl

    if constraints is not None:
        penalty_total = (
            lambda_rep * constraints.p_rep
            + lambda_contra * constraints.p_contra
            + lambda_length * constraints.p_length
            + lambda_format * constraints.p_format
        )
    else:
        penalty_total = 0.0

    r_total = max(clip_min, min(clip_max, r_quality - penalty_total))

    return RewardBreakdown(
        r_total=round(r_total, 4),
        r_quality=round(r_quality, 4),
        r_local=r_local,
        r_global=r_global,
        r_holistic=round(holistic_score, 4) if holistic_score is not None else None,
        penalty_total=round(penalty_total, 4),
        mode=mode,
        local_scores=local_scores,
        global_scores=global_scores,
        constraints=constraints,
    )


def apply_rlmr_dynamic_shift(
    breakdowns: List[RewardBreakdown],
    items: Sequence[EvaluationInput],
    gamma: float = 0.05,
    min_len_ratio: float = 0.85,
    max_len_ratio: float = 1.15,
    max_think_words: int = 400,
) -> None:
    """
    Minimal Orthogonal RLMR Dynamic Group Penalty (AAAI 2026, Eq. 4-7) in-place over each GRPO group.
      - Uses strictly raw quality `r_i = bd.r_quality` (zero fixed lambda double-counting).
      - Checks 3 pure-Python objective constraints: `V_length AND V_format AND V_no_loop`.
      - When 0 < k < n:
            delta = max(0, (n * r_max_vio + n * gamma - sum(r_i)) / (n - k))
            r'_i  = r_i (if compliant) else r_i - delta
        Mathematically guarantees every violating rollout gets r'_j <= mean(r') - gamma (negative GRPO advantage).
      - When k == 0: r'_i = r_i (pure quality optimization).
      - When k == n: safe soft fallback `r_i * length_compliance * format_factor * loop_factor` (no div-by-0, no wasted resampling).
    """
    groups: Dict[tuple, List[int]] = defaultdict(list)
    for idx, it in enumerate(items):
        key = (it.premise_id, it.chapter_index) if it.premise_id else ("__group__", 0)
        groups[key].append(idx)

    for grp_indices in groups.values():
        n = len(grp_indices)
        verifications = [
            verify_objective_constraints(
                items[i],
                min_len_ratio=min_len_ratio,
                max_len_ratio=max_len_ratio,
                max_think_words=max_think_words,
            )
            for i in grp_indices
        ]
        r_vals = [float(breakdowns[i].r_quality) for i in grp_indices]
        vio_positions = [pos for pos, v in enumerate(verifications) if not v["is_compliant"]]
        k = len(vio_positions)

        if 0 < k < n:
            r_max_vio = max(r_vals[pos] for pos in vio_positions)
            sum_r = sum(r_vals)
            delta = max(0.0, (n * r_max_vio + n * float(gamma) - sum_r) / float(n - k))
            for pos, idx in enumerate(grp_indices):
                bd = breakdowns[idx]
                v = verifications[pos]
                if v["is_compliant"]:
                    bd.r_total = round(r_vals[pos], 4)
                    bd.penalty_total = 0.0
                    applied_delta = 0.0
                else:
                    bd.r_total = round(r_vals[pos] - delta, 4)
                    bd.penalty_total = round(delta, 4)
                    applied_delta = round(delta, 4)
                bd.constraints = ConstraintPenalties(
                    p_length=round(1.0 - v["length_compliance"], 4),
                    p_rep=0.0 if v["v_no_loop"] else 1.0,
                    p_contra=0.0,
                    p_format=0.0 if v["v_format"] else 1.0,
                    prose_words=v["prose_words"],
                    think_words=v["think_words"],
                    details=v,
                )
                bd.metadata.update(
                    {
                        "rlmr_is_compliant": v["is_compliant"],
                        "rlmr_delta": applied_delta,
                        "rlmr_group_k": k,
                        "rlmr_group_n": n,
                        "rlmr_failed": v["failed_constraints"],
                    }
                )
        elif k == 0:
            for pos, idx in enumerate(grp_indices):
                bd = breakdowns[idx]
                v = verifications[pos]
                bd.r_total = round(r_vals[pos], 4)
                bd.penalty_total = 0.0
                bd.constraints = ConstraintPenalties(
                    p_length=0.0,
                    p_rep=0.0,
                    p_contra=0.0,
                    p_format=0.0,
                    prose_words=v["prose_words"],
                    think_words=v["think_words"],
                    details=v,
                )
                bd.metadata.update(
                    {
                        "rlmr_is_compliant": True,
                        "rlmr_delta": 0.0,
                        "rlmr_group_k": 0,
                        "rlmr_group_n": n,
                        "rlmr_failed": [],
                    }
                )
        else:
            # k == n: all rollouts violated -> soft fallback preserves learning signal without dropping the group
            for pos, idx in enumerate(grp_indices):
                bd = breakdowns[idx]
                v = verifications[pos]
                factor = v["length_compliance"] * (1.0 if v["v_format"] else 0.5) * (1.0 if v["v_no_loop"] else 0.7)
                adjusted = round(r_vals[pos] * factor, 4)
                bd.r_total = adjusted
                bd.penalty_total = round(max(0.0, r_vals[pos] - adjusted), 4)
                bd.constraints = ConstraintPenalties(
                    p_length=round(1.0 - v["length_compliance"], 4),
                    p_rep=0.0 if v["v_no_loop"] else 1.0,
                    p_contra=0.0,
                    p_format=0.0 if v["v_format"] else 1.0,
                    prose_words=v["prose_words"],
                    think_words=v["think_words"],
                    details=v,
                )
                bd.metadata.update(
                    {
                        "rlmr_is_compliant": False,
                        "rlmr_delta": bd.penalty_total,
                        "rlmr_group_k": k,
                        "rlmr_group_n": n,
                        "rlmr_failed": v["failed_constraints"],
                    }
                )


class Phase2RewardEngine:
    """
    End-to-end Phase 2 Reward Engine managing:
      - 32B Local HF 4-bit Judge (for Local, Global, Unified, or Holistic scoring)
      - Python + MiniLM Constraint Evaluator (0 LLM calls)
      - Aggregator (EXP-00 .. EXP-05)
      - Optional JSONL per-rollout component logging for stability & reward-hacking audits.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str | Path] = None,
        judge_backend_override: Optional[Any] = None,
        log_jsonl_path: Optional[str | Path] = None,
    ):
        cfg = dict(config or {})
        if config_path:
            loaded = load_reward_config(config_path)
            cfg = {**loaded, **cfg}

        self.config = cfg
        self.mode = str(cfg.get("mode", "hierarchical"))
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode: {self.mode}")

        agg_cfg = cfg.get("aggregation", {})
        self.alpha_global = float(agg_cfg.get("alpha_global", 0.65))
        self.local_weights = agg_cfg.get("local_weights", DEFAULT_LOCAL_WEIGHTS)
        self.global_weights = agg_cfg.get("global_weights", DEFAULT_GLOBAL_WEIGHTS)
        self.geo_epsilon = float(agg_cfg.get("geo_epsilon", 0.05))

        pen_cfg = cfg.get("penalties", {})
        self.penalty_mode = str(pen_cfg.get("penalty_mode", "fixed"))
        self.rlmr_gamma = float(pen_cfg.get("rlmr_gamma", 0.05))
        self.rlmr_min_len_ratio = float(pen_cfg.get("min_len_ratio", 0.85))
        self.rlmr_max_len_ratio = float(pen_cfg.get("max_len_ratio", 1.15))
        self.lambda_rep = float(pen_cfg.get("lambda_rep", 0.10))
        self.lambda_contra = float(pen_cfg.get("lambda_contra", 0.20))
        self.lambda_length = float(pen_cfg.get("lambda_length", 0.10))
        self.lambda_format = float(pen_cfg.get("lambda_format", 0.05))

        eval_cfg = cfg.get("evaluator", {})
        self.single_pass = bool(eval_cfg.get("single_pass", True))
        self.include_evidence = bool(eval_cfg.get("include_evidence", False))
        self.backend_type = str(eval_cfg.get("backend", "local_hf_4bit"))
        self.judge_model_id = str(eval_cfg.get("model_id", DEFAULT_LOCAL_32B_JUDGE_MODEL))
        self.judge_device = str(eval_cfg.get("device", "cuda:0"))
        self.eval_batch_size = int(eval_cfg.get("eval_batch_size", 2))
        self.max_input_tokens = int(eval_cfg.get("max_input_tokens", 3584))

        con_cfg = cfg.get("constraints", {})
        self.max_think_words = int(con_cfg.get("max_think_words", 400))
        use_embedder_default = (self.backend_type != "mock") and (self.penalty_mode != "rlmr_orthogonal")
        self.constraint_evaluator = ConstraintEvaluator(
            embedder_name=str(con_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
            device=str(con_cfg.get("device", "cpu")),
            use_embedder=bool(con_cfg.get("use_embedder", use_embedder_default)),
            redundancy_cos_threshold=float(con_cfg.get("redundancy_cos_threshold", 0.78)),
            max_think_words=self.max_think_words,
        )

        self._baseline_engine: Optional[CompositeRewardEngine] = None
        if self.mode == "baseline_v2":
            self._baseline_engine = CompositeRewardEngine(
                config=cfg.get("baseline_rewards", {}),
                device=str(con_cfg.get("device", "cpu")),
            )

        self.judge_backend = judge_backend_override
        if self.mode != "baseline_v2" and self.judge_backend is None:
            self.judge_backend = build_judge_backend(
                backend=self.backend_type,
                model_id=self.judge_model_id,
                device=self.judge_device,
                eval_batch_size=self.eval_batch_size,
                max_input_tokens=self.max_input_tokens,
            )

        self.local_eval = (
            LocalExpansionEvaluator(self.judge_backend, include_evidence=self.include_evidence)
            if self.judge_backend is not None
            else None
        )
        self.global_eval = (
            GlobalChapterEvaluator(self.judge_backend, include_evidence=self.include_evidence)
            if self.judge_backend is not None
            else None
        )
        self.unified_eval = (
            UnifiedPhase2Evaluator(self.judge_backend)
            if self.judge_backend is not None
            else None
        )
        self.holistic_eval = (
            HolisticChapterEvaluator(self.judge_backend)
            if self.judge_backend is not None
            else None
        )

        self.log_jsonl_path = Path(log_jsonl_path) if log_jsonl_path else None
        if self.log_jsonl_path is not None:
            self.log_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate_batch(self, items: Sequence[EvaluationInput]) -> List[RewardBreakdown]:
        """Scores a batch of rollouts (e.g. G=4 GRPO candidates) and returns full `RewardBreakdown`s."""
        if not items:
            return []

        breakdowns: List[RewardBreakdown] = []

        if self.mode == "baseline_v2":
            assert self._baseline_engine is not None
            for it in items:
                res = self._baseline_engine.compute_batch_rewards(
                    rollouts=[it.generated_chapter],
                    target_length=it.target_words,
                    context_summary=it.context_summary,
                    prev_chapter_tail=it.prev_chapter_tail,
                )
                r_tot = float(res["total_rewards"][0])
                con = self.constraint_evaluator.evaluate(it, contradictions=[])
                breakdowns.append(
                    RewardBreakdown(
                        r_total=round(r_tot, 4),
                        r_quality=round(float(res["r_quality"][0]), 4),
                        r_local=None,
                        r_global=None,
                        r_holistic=None,
                        penalty_total=0.0,
                        mode=self.mode,
                        constraints=con,
                        metadata={
                            "premise_id": it.premise_id,
                            "chapter_index": it.chapter_index,
                            "baseline_r_length": float(res["r_length"][0]),
                            "baseline_r_coherence": float(res["r_coherence"][0]),
                            "baseline_r_quality": float(res["r_quality"][0]),
                        },
                    )
                )
        elif self.mode == "holistic":
            assert self.holistic_eval is not None
            hol_outs = self.holistic_eval.evaluate_batch(items)
            for it, h in zip(items, hol_outs):
                con = self.constraint_evaluator.evaluate(it, contradictions=h["contradictions"])
                bd = aggregate_phase2_reward(
                    mode=self.mode,
                    holistic_score=h["holistic_score"],
                    constraints=con,
                    lambda_rep=self.lambda_rep,
                    lambda_contra=self.lambda_contra,
                    lambda_length=self.lambda_length,
                    lambda_format=self.lambda_format,
                )
                bd.metadata = {"premise_id": it.premise_id, "chapter_index": it.chapter_index}
                breakdowns.append(bd)
        elif self.mode == "local_only":
            assert self.local_eval is not None
            loc_outs = self.local_eval.evaluate_batch(items)
            for it, loc in zip(items, loc_outs):
                con = self.constraint_evaluator.evaluate(it, contradictions=[])
                bd = aggregate_phase2_reward(
                    mode=self.mode,
                    local_scores=loc,
                    constraints=con,
                    local_weights=self.local_weights,
                    lambda_rep=self.lambda_rep,
                    lambda_contra=self.lambda_contra,
                    lambda_length=self.lambda_length,
                    lambda_format=self.lambda_format,
                )
                bd.metadata = {"premise_id": it.premise_id, "chapter_index": it.chapter_index}
                breakdowns.append(bd)
        elif self.mode == "global_only":
            assert self.global_eval is not None
            glob_outs = self.global_eval.evaluate_batch(items)
            for it, glob in zip(items, glob_outs):
                con = self.constraint_evaluator.evaluate(it, contradictions=glob.contradictions)
                bd = aggregate_phase2_reward(
                    mode=self.mode,
                    global_scores=glob,
                    constraints=con,
                    global_weights=self.global_weights,
                    lambda_rep=self.lambda_rep,
                    lambda_contra=self.lambda_contra,
                    lambda_length=self.lambda_length,
                    lambda_format=self.lambda_format,
                )
                bd.metadata = {"premise_id": it.premise_id, "chapter_index": it.chapter_index}
                breakdowns.append(bd)
        else:
            # hierarchical (EXP-02) or hierarchical_geo (EXP-05)
            if self.single_pass:
                assert self.unified_eval is not None
                pairs = self.unified_eval.evaluate_batch(items)
                loc_outs = [p[0] for p in pairs]
                glob_outs = [p[1] for p in pairs]
            else:
                assert self.local_eval is not None and self.global_eval is not None
                loc_outs = self.local_eval.evaluate_batch(items)
                glob_outs = self.global_eval.evaluate_batch(items)

            for it, loc, glob in zip(items, loc_outs, glob_outs):
                con = self.constraint_evaluator.evaluate(it, contradictions=glob.contradictions)
                bd = aggregate_phase2_reward(
                    mode=self.mode,
                    local_scores=loc,
                    global_scores=glob,
                    constraints=con,
                    alpha_global=self.alpha_global,
                    local_weights=self.local_weights,
                    global_weights=self.global_weights,
                    lambda_rep=self.lambda_rep,
                    lambda_contra=self.lambda_contra,
                    lambda_length=self.lambda_length,
                    lambda_format=self.lambda_format,
                    geo_epsilon=self.geo_epsilon,
                )
                bd.metadata = {"premise_id": it.premise_id, "chapter_index": it.chapter_index}
                breakdowns.append(bd)

        if self.penalty_mode == "rlmr_orthogonal" and self.mode != "baseline_v2":
            apply_rlmr_dynamic_shift(
                breakdowns=breakdowns,
                items=items,
                gamma=self.rlmr_gamma,
                min_len_ratio=self.rlmr_min_len_ratio,
                max_len_ratio=self.rlmr_max_len_ratio,
                max_think_words=self.max_think_words,
            )

        if self.log_jsonl_path is not None:
            with open(self.log_jsonl_path, "a", encoding="utf-8") as f:
                for bd in breakdowns:
                    f.write(json.dumps(bd.to_flat_log(), ensure_ascii=False) + "\n")

        return breakdowns

    def release(self) -> None:
        if self.judge_backend is not None and hasattr(self.judge_backend, "release"):
            self.judge_backend.release()
