"""
Unified Intrinsic Reward Engine for MA-GRPO.

Contains the 3 standardized intrinsic mathematical reward components:
    1. Length Reward (LongWriter-Style Piecewise with Buffer Zone):
       r_length(o) = 1.0,                                    if L_lower <= len(o) <= L_upper
                   = len(o) / L_lower,                       if len(o) < L_lower
                   = (L_max - len(o)) / (L_max - L_upper),   if len(o) > L_upper
    2. Coherence & Diversity Reward (Transition Smoothness & Anti-Repetition):
       r_coherence(o) = CosineSim(End(C_{i-1}), Start(o)) * (|Uniq3Grams(o) \\ 3Grams(s_i)| / |Total3Grams(o)|)
    3. Quality Reward (Fluency, Vocabulary Entropy, Sentence Burstiness):
       r_quality(o) = alpha * r_fluency + beta * r_richness + gamma * r_burstiness
    4. Evaluator Station (MAGRPORewardEvaluator & CompositeRewardEngine) for TRL GRPOTrainer.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

try:
    import yaml
except ImportError:
    yaml = None


# =====================================================================
# 0. NARRATIVE REASONING & PROSE UTILITIES
# =====================================================================
def strip_think_tags(text: str) -> str:
    """
    Removes test-time reasoning tokens inside <think>...</think> if present.
    Ensures rewards are evaluated strictly on the actual narrative prose.
    """
    if not text:
        return ""
    clean = re.sub(r"<think>[\s\S]*?</think>", "", text)
    clean = re.sub(r"<think>[\s\S]*", "", clean)
    return clean.strip()


def count_words(text: str) -> int:
    """Counts words using whitespace splitting, excluding reasoning tokens."""
    if not text or not text.strip():
        return 0
    clean_text = strip_think_tags(text)
    return len(clean_text.split())


def extract_words(text: str) -> List[str]:
    """Tokenizes text into lowercase words, excluding reasoning tokens."""
    if not text:
        return []
    clean_text = strip_think_tags(text)
    return re.findall(r"\w+", clean_text.lower())


def extract_head_words(text: str, n_words: int = 150) -> str:
    """Extracts the first n words from text."""
    words = strip_think_tags(text).strip().split()
    return " ".join(words[:n_words]) if words else ""


def extract_tail_words(text: str, n_words: int = 150) -> str:
    """Extracts the last n words from text."""
    words = strip_think_tags(text).strip().split()
    return " ".join(words[-n_words:]) if words else ""


def get_ngrams(words: List[str], n: int = 3) -> List[Tuple[str, ...]]:
    """Generates a list of n-grams from a word list."""
    if len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def extract_length_bounds(
    prompt_text: str,
    target_length: Optional[int] = None,
    fallback_target: int = 1400,
    lower_ratio: float = 0.85,
    upper_ratio: float = 1.15,
    max_ratio: float = 2.0,
) -> Tuple[int, int, int]:
    """
    Extracts (L_lower, L_upper, L_max) from prompt text or target length.
    Supports ranges such as:
      - 'between 1300–1400 words' / 'between 1550 and 1600 words' -> L_lower=1300, L_upper=1400
      - 'at least 900 words' / 'over 800 words' -> L_lower=900, L_upper=round(900 * 1.15)
      - 'around 1600 words' -> L_lower=round(1600 * 0.85), L_upper=round(1600 * 1.15)
    L_max defaults to max_ratio * L_upper (2.0 * L_upper).
    """
    if not prompt_text:
        target = max(1, target_length if target_length and target_length > 0 else fallback_target)
        l_low = max(1, int(round(target * lower_ratio)))
        l_up = max(l_low, int(round(target * upper_ratio)))
        l_max = int(round(target * max_ratio))
        return l_low, l_up, l_max

    clean = prompt_text.replace(",", "")

    # 1. Match explicit range: 'between X and Y words' or 'between X to/–/- Y words'
    match = re.search(r"between\s+(\d+)\s*(?:and|to|–|-)\s*(\d+)\s*words", clean, re.IGNORECASE)
    if not match:
        match = re.search(r"(\d+)\s*(?:–|-)\s*(\d+)\s*words", clean, re.IGNORECASE)
    if match:
        v1, v2 = int(match.group(1)), int(match.group(2))
        l_low, l_up = min(v1, v2), max(v1, v2)
        l_max = int(round(l_up * max_ratio))
        return l_low, l_up, l_max

    # 2. Match minimum constraints: 'at least X words', 'over X words'
    match = re.search(r"(?:at least|over)\s+(\d+)\s*words", clean, re.IGNORECASE)
    if match:
        l_low = int(match.group(1))
        l_up = int(round(l_low * 1.20))
        l_max = int(round(l_up * max_ratio))
        return l_low, l_up, l_max

    # 3. Match 'Total Target Length: Approximately X words' or 'around X words' or single number
    match = re.search(r"Total Target Length:\s*Approximately\s*(\d+)\s*words", clean, re.IGNORECASE)
    if not match:
        match = re.search(r"(?:around|about|approximately)\s+(\d+)\s*words", clean, re.IGNORECASE)
    if not match:
        match = re.search(r"(\d{3,5})\s*words", clean, re.IGNORECASE)

    if match:
        target = int(match.group(1))
    elif target_length and target_length > 0:
        target = target_length
    else:
        target = fallback_target

    l_low = max(1, int(round(target * lower_ratio)))
    l_up = max(l_low, int(round(target * upper_ratio)))
    l_max = int(round(target * max_ratio))
    return l_low, l_up, l_max


# =====================================================================
# 1. LENGTH REWARD (LongWriter-Style Piecewise with Buffer Zone)
# =====================================================================
def reward_length(
    completions: List[str],
    target_length: int = 1200,
    l_lower: Optional[int] = None,
    l_upper: Optional[int] = None,
    l_max: Optional[int] = None,
    **kwargs: Any,
) -> List[float]:
    """
    Hàm thưởng độ dài LongWriter-Zero (Piecewise với Vùng an toàn & Phạt bất đối xứng):
    - Vùng an toàn: [0.85 * target_length, 1.15 * target_length] -> Điểm tuyệt đối 1.0
    - Viết quá ngắn (< 0.85 * target): Phạt tuyến tính dốc về 0: len(o) / L_lower
    - Viết quá dài (> 1.15 * target): Phạt nhẹ giảm dần về 0 tại l_max: (L_max - len(o)) / (L_max - L_upper)
    """
    if l_lower is None:
        l_lower = int(0.85 * target_length)
    if l_upper is None:
        l_upper = int(1.15 * target_length)
    if l_max is None:
        l_max = int(2.0 * target_length)

    l_lower = max(1, l_lower)
    l_upper = max(l_lower, l_upper)
    if l_max <= l_upper:
        l_max = l_upper + max(100, int(0.5 * l_upper))

    rewards = []
    for comp in completions:
        clean_text = strip_think_tags(comp)
        comp_len = len(clean_text.split())

        if l_lower <= comp_len <= l_upper:
            score = 1.0
        elif comp_len < l_lower:
            score = max(0.0, comp_len / float(l_lower))
        else:
            score = max(0.0, (l_max - comp_len) / float(l_max - l_upper))

        rewards.append(float(score))

    return rewards


def calculate_length_reward(
    text: str,
    target_length: int = 1200,
    l_lower: Optional[int] = None,
    l_upper: Optional[int] = None,
    l_max: Optional[int] = None,
    **kwargs: Any,
) -> float:
    """Convenience single-text wrapper for reward_length."""
    scores = reward_length(
        [text],
        target_length=target_length,
        l_lower=l_lower,
        l_upper=l_upper,
        l_max=l_max,
        **kwargs,
    )
    return scores[0] if scores else 0.0


def batch_length_rewards(
    texts: List[str],
    target_length: int = 1200,
    l_lower: Optional[int] = None,
    l_upper: Optional[int] = None,
    l_max: Optional[int] = None,
    **kwargs: Any,
) -> List[float]:
    """Computes length rewards for a batch of G rollouts."""
    return reward_length(
        texts,
        target_length=target_length,
        l_lower=l_lower,
        l_upper=l_upper,
        l_max=l_max,
        **kwargs,
    )


# =====================================================================
# 2. COHERENCE & DIVERSITY REWARD (Transition Smoothness & Anti-Repetition)
# =====================================================================
def extract_entity_trigrams(text: str) -> Set[Tuple[str, ...]]:
    """
    Trích xuất các 3-gram chứa từ viết hoa (tên riêng/thực thể, nhân vật)
    để không phạt nhầm khi câu chuyện lặp lại tên nhân vật/địa danh.
    Theo ma-grpo-proposal-v4.md Section 5.2 & 6.1.
    """
    words = text.split()
    entity_3grams: Set[Tuple[str, ...]] = set()
    for i in range(len(words) - 2):
        tri = words[i : i + 3]
        if any(w[0].isupper() for w in tri if len(w) > 0):
            entity_3grams.add(tuple(re.sub(r"\W+", "", w).lower() for w in tri))
    return entity_3grams


def calculate_anti_context_ratio(rollout_text: str, context_summary: str) -> float:
    r"""
    Computes anti-context 3-gram ratio with entity protection:
        ratio = |Uniq3Grams(rollout) \ (3Grams(summary) \ Entities)| / |Total3Grams(rollout)|
    """
    clean_text = strip_think_tags(rollout_text)
    comp_words = [re.sub(r"\W+", "", w).lower() for w in clean_text.split() if re.sub(r"\W+", "", w)]
    if len(comp_words) < 3:
        return 0.0

    summary_words = [re.sub(r"\W+", "", w).lower() for w in context_summary.split() if re.sub(r"\W+", "", w)]
    summary_3grams: Set[Tuple[str, ...]] = (
        set(tuple(summary_words[i : i + 3]) for i in range(len(summary_words) - 2))
        if len(summary_words) >= 3
        else set()
    )

    entities_in_summary = extract_entity_trigrams(context_summary)
    forbidden_3grams = summary_3grams - entities_in_summary

    comp_3grams = [tuple(comp_words[i : i + 3]) for i in range(len(comp_words) - 2)]
    total_3grams = len(comp_3grams)
    if total_3grams == 0:
        return 0.0

    uniq_new_3grams = set(comp_3grams) - forbidden_3grams
    ratio = len(uniq_new_3grams) / float(total_3grams)
    return min(1.0, max(0.0, float(ratio)))


class CoherenceRewardModel:
    """
    Thưởng Độ liên kết mạch văn & Chống lặp từ ngữ cảnh:
    Công thức: CosineSim(End(C_{i-1}), Start(o_i^{(k)})) * (|Uniq3Grams(o_i^{(k)}) \\ 3Grams(s_i)| / |Total3Grams(o_i^{(k)})|)
    """

    def __init__(
        self,
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self.embedder = None
        self.device = device
        self.embedder_name = embedder_name

    def _get_embedder(self):
        if self.embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self.embedder = SentenceTransformer(self.embedder_name, device=self.device)
            except Exception:
                self.embedder = None
        return self.embedder

    def calculate_cosine_sim(self, prev_end_text: str, current_start_text: str) -> float:
        if not prev_end_text or not current_start_text:
            return 1.0  # Mặc định 1.0 cho Chapter 1 (không có chương trước)

        embedder = self._get_embedder()
        if embedder is not None and torch is not None:
            try:
                embeddings = embedder.encode([prev_end_text, current_start_text], convert_to_tensor=True)
                sim = torch.nn.functional.cosine_similarity(
                    embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
                ).item()
                return max(0.0, float(sim))
            except Exception:
                pass

        # Fallback Jaccard nếu chưa có sentence-transformers hoặc lỗi GPU
        words_prev = set(re.findall(r"\w+", prev_end_text.lower()))
        words_curr = set(re.findall(r"\w+", current_start_text.lower()))
        if not words_prev or not words_curr:
            return 0.5
        jaccard = len(words_prev & words_curr) / float(len(words_prev | words_curr))
        return min(1.0, max(0.0, jaccard * 3.0))

    def _extract_entity_trigrams(self, text: str) -> Set[Tuple[str, ...]]:
        """Trích xuất các 3-gram chứa từ viết hoa (tên riêng/thực thể) để không phạt nhầm"""
        return extract_entity_trigrams(text)

    def __call__(
        self,
        completions: List[str],
        context_summary: str = "",
        prev_chapter_end: str = "",
        prev_chapter_tail: str = "",
        **kwargs: Any,
    ) -> List[float]:
        rewards = []
        prev_end = prev_chapter_end or prev_chapter_tail or ""

        # 1. Trích xuất 3-grams từ bản tóm tắt s_i có bảo vệ thực thể
        summary_words = [re.sub(r"\W+", "", w).lower() for w in context_summary.split() if re.sub(r"\W+", "", w)]
        summary_3grams: Set[Tuple[str, ...]] = (
            set(tuple(summary_words[i : i + 3]) for i in range(len(summary_words) - 2))
            if len(summary_words) >= 3
            else set()
        )

        # Trích xuất các thực thể trong summary để miễn trừ phạt
        entities_in_summary = self._extract_entity_trigrams(context_summary)
        forbidden_3grams = summary_3grams - entities_in_summary

        for comp in completions:
            clean_comp = strip_think_tags(comp)
            comp_words = [re.sub(r"\W+", "", w).lower() for w in clean_comp.split() if re.sub(r"\W+", "", w)]
            if len(comp_words) < 3:
                rewards.append(0.0)
                continue

            # 2. Phép trừ tập hợp có bảo vệ thực thể: Uniq3Grams(rollout) \ (3Grams(summary) \ Entities)
            comp_3grams = [tuple(comp_words[i : i + 3]) for i in range(len(comp_words) - 2)]
            total_3grams = len(comp_3grams)
            uniq_new_3grams = set(comp_3grams) - forbidden_3grams
            diversity_ratio = len(uniq_new_3grams) / float(total_3grams) if total_3grams > 0 else 0.0

            # 3. Mối nối vi mô giữa câu kết chương trước và 2 câu mở đầu rollout mới
            comp_sentences = [s.strip() for s in re.split(r"[.!?]+", clean_comp) if s.strip()]
            start_text = " ".join(comp_sentences[:2]) if comp_sentences else clean_comp[:200]
            sim_score = self.calculate_cosine_sim(prev_end, start_text)

            # Tổng hợp điểm Coherence
            composite_score = sim_score * diversity_ratio
            rewards.append(float(round(composite_score, 4)))

        return rewards

    def calculate_coherence_rewards(
        self,
        rollouts: List[str],
        context_summary: str = "",
        prev_chapter_tail: Optional[str] = None,
    ) -> List[float]:
        """Backwards compatibility alias for calculate_coherence_rewards."""
        return self(
            completions=rollouts,
            context_summary=context_summary,
            prev_chapter_tail=prev_chapter_tail or "",
        )


# Backward compatibility alias
CoherenceRewardEvaluator = CoherenceRewardModel


# =====================================================================
# 3. QUALITY REWARD (Fluency, Vocabulary Entropy, Sentence Burstiness)
# =====================================================================
def calculate_vocabulary_entropy(text: str) -> float:
    """
    Đo độ phong phú vốn từ bằng Shannon Entropy chuẩn hóa:
        H = - sum_v P(v) * log2(P(v))
        R_richness = H / log2(N_words) in [0.0, 1.0]
    """
    clean_text = strip_think_tags(text)
    words = re.findall(r"\w+", clean_text.lower())
    if not words:
        return 0.0

    vocab_counts = Counter(words)
    total_words = len(words)
    entropy = -sum((count / float(total_words)) * math.log2(count / float(total_words)) for count in vocab_counts.values())
    h_max = math.log2(total_words) if total_words > 1 else 1.0
    return min(1.0, max(0.0, float(entropy / h_max)))


def calculate_gaussian_burstiness(
    text: str,
    b_target: float = 11.0,
    tau_b: float = 4.0,
) -> float:
    """
    Đo nhịp điệu bằng hàm Gaussian centered quanh mốc b_target (~11.0).
    Chống bẫy Reward Hacking (1 câu 100 từ đan xen 1 câu 2 từ để StdDev bùng nổ vô tận).
    Theo ma-grpo-proposal-v4.md Section 5.3 & 6.1.
    """
    clean_text = strip_think_tags(text)
    sentences = [s.strip() for s in re.split(r"[.!?]+", clean_text) if len(s.strip()) > 0]
    if len(sentences) < 2:
        return 0.3
    lens = [len(re.findall(r"\w+", s)) for s in sentences]
    if np is not None:
        std_len = float(np.std(lens))
    else:
        import statistics

        std_len = float(statistics.stdev(lens)) if len(lens) > 1 else 0.0

    score = math.exp(-((std_len - b_target) ** 2) / (2.0 * (tau_b ** 2)))
    return float(round(score, 4))


def calculate_sigmoid_fluency(completion: str) -> float:
    """
    Đo độ mượt & chuẩn ngữ pháp bằng Sigmoid NLL / Cấu trúc từ vựng mượt.
    Chống lại sự sụt giảm quá dốc của hàm exponential PPL cũ.
    """
    clean_text = strip_think_tags(completion)
    words = re.findall(r"\w+", clean_text.lower())
    if len(words) < 10:
        return 0.1
    avg_word_len = sum(len(w) for w in words) / float(len(words))
    dist = abs(avg_word_len - 5.2)
    score = 1.0 / (1.0 + math.exp((dist - 1.5) * 2.0))
    return float(round(score, 4))


def calculate_burstiness(
    text: str,
    mu_0: float = 11.0,
    sigma_0: float = 4.0,
    **kwargs: Any,
) -> float:
    """Backwards compatible wrapper for burstiness calculation."""
    b_t = kwargs.get("b_target", mu_0)
    t_b = kwargs.get("tau_b", sigma_0)
    return calculate_gaussian_burstiness(text, b_target=b_t, tau_b=t_b)


class QualityRewardModel:
    """
    Thưởng Độ hay, Độ mượt & Nhịp điệu văn phong nội tại (Cải tiến sau Peer Review - v2 / Proposal v4):
    R_quality = alpha * R_fluency (Sigmoid NLL) + beta * R_richness (Entropy) + gamma * R_burstiness (Gaussian Target)
    """

    def __init__(
        self,
        alpha: float = 0.40,
        beta: float = 0.30,
        gamma: float = 0.30,
        b_target: float = 11.0,
        tau_b: float = 4.0,
        **kwargs: Any,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.b_target = float(kwargs.get("burstiness_b_target", kwargs.get("mu_0", b_target)))
        self.tau_b = float(kwargs.get("burstiness_tau_b", kwargs.get("sigma_0", tau_b)))

    def _calculate_richness(self, text: str) -> float:
        """Đo độ phong phú vốn từ bằng Shannon Entropy"""
        return calculate_vocabulary_entropy(text)

    def _calculate_gaussian_burstiness(self, text: str) -> float:
        """Đo nhịp điệu bằng hàm Gaussian centered quanh mốc b_target (~11)"""
        return calculate_gaussian_burstiness(text, b_target=self.b_target, tau_b=self.tau_b)

    def _calculate_burstiness(self, text: str) -> float:
        """Alias cho Gaussian Burstiness"""
        return self._calculate_gaussian_burstiness(text)

    def _calculate_sigmoid_fluency(self, completion: str) -> float:
        """Đo độ mượt & chuẩn ngữ pháp bằng Sigmoid NLL / Cấu trúc từ vựng mượt"""
        return calculate_sigmoid_fluency(completion)

    def _calculate_fluency(self, completion: str) -> float:
        """Alias cho Sigmoid Fluency"""
        return self._calculate_sigmoid_fluency(completion)

    def __call__(self, completions: List[str], **kwargs: Any) -> List[float]:
        rewards = []
        for raw_comp in completions:
            clean_comp = strip_think_tags(raw_comp)
            r_fluency = self._calculate_sigmoid_fluency(clean_comp)
            r_richness = self._calculate_richness(clean_comp)
            r_burstiness = self._calculate_gaussian_burstiness(clean_comp)

            r_quality = (
                self.alpha * r_fluency
                + self.beta * r_richness
                + self.gamma * r_burstiness
            )
            rewards.append(float(round(r_quality, 4)))

        return rewards

    def calculate_quality_rewards(self, rollouts: List[str]) -> Dict[str, List[float]]:
        """Backwards compatibility helper for legacy pipeline tests."""
        fluency_scores = [self._calculate_sigmoid_fluency(r) for r in rollouts]
        richness_scores = [self._calculate_richness(r) for r in rollouts]
        burstiness_scores = [self._calculate_gaussian_burstiness(r) for r in rollouts]
        total_scores = self(rollouts)
        return {
            "total": total_scores,
            "fluency": fluency_scores,
            "richness": richness_scores,
            "burstiness": burstiness_scores,
            "ppl": [0.0] * len(rollouts),
        }


# Backward compatibility alias
QualityRewardEvaluator = QualityRewardModel


# =====================================================================
# 4. TRẠM TỔNG HỢP REWARD CHO TRL GRPOTRAINER (v2)
# =====================================================================
class MAGRPORewardEvaluatorV2:
    """
    Trạm phân phối hàm reward chuẩn v2 theo ma-grpo-proposal-v4.md cho TRL GRPOTrainer.
    Cung cấp get_reward_functions() gồm:
      1. length_reward_wrapper
      2. coherence_reward_wrapper
      3. quality_reward_wrapper
    """

    def __init__(
        self,
        target_length: int = 1200,
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        alpha: float = 0.40,
        beta: float = 0.30,
        gamma: float = 0.30,
        b_target: float = 11.0,
        tau_b: float = 4.0,
    ):
        self.target_length = target_length
        self.coherence_fn = CoherenceRewardModel(embedder_name=embedder_name, device=device)
        self.quality_fn = QualityRewardModel(
            alpha=alpha, beta=beta, gamma=gamma, b_target=b_target, tau_b=tau_b
        )

    def get_reward_functions(self):
        """Trả về danh sách các hàm reward v2 tương thích TRL GRPOTrainer"""

        def length_reward_wrapper(completions: List[str], **kwargs: Any) -> List[float]:
            kwargs_copy = dict(kwargs)
            t_len = (
                kwargs_copy.pop("target_length", None)
                or kwargs_copy.pop("target_words", None)
                or self.target_length
            )
            return reward_length(completions, target_length=t_len, **kwargs_copy)

        def coherence_reward_wrapper(completions: List[str], **kwargs: Any) -> List[float]:
            return self.coherence_fn(completions, **kwargs)

        def quality_reward_wrapper(completions: List[str], **kwargs: Any) -> List[float]:
            return self.quality_fn(completions, **kwargs)

        return [length_reward_wrapper, coherence_reward_wrapper, quality_reward_wrapper]


MAGRPORewardEvaluator = MAGRPORewardEvaluatorV2


class CompositeRewardEngine:
    """
    Main unified reward engine for MA-GRPO. Combines reward_length,
    CoherenceRewardModel, and QualityRewardModel into a single
    mathematical reward tensor for G candidate rollouts.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = device or "cpu"

        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                if yaml is not None:
                    cfg = yaml.safe_load(f) or {}
                else:
                    cfg = {}
                config = cfg.get("rewards", {})
        elif config is None:
            config = {}

        weights = config.get("weights", {})
        self.w_length = float(weights.get("w_length", 0.40))
        self.w_coherence = float(weights.get("w_coherence", 0.30))
        self.w_quality = float(weights.get("w_quality", 0.30))

        coh_cfg = config.get("coherence", {})
        embedder_name = coh_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.coherence_evaluator = CoherenceRewardModel(embedder_name=embedder_name, device=self.device)

        qual_cfg = config.get("quality", {})
        alpha = float(qual_cfg.get("alpha_fluency", 0.40))
        beta = float(qual_cfg.get("beta_richness", 0.30))
        gamma = float(qual_cfg.get("gamma_burstiness", 0.30))
        b_target = float(qual_cfg.get("b_target", qual_cfg.get("burstiness_b_target", qual_cfg.get("burstiness_mu0", 11.0))))
        tau_b = float(qual_cfg.get("tau_b", qual_cfg.get("burstiness_tau_b", qual_cfg.get("burstiness_sigma0", 4.0))))
        self.quality_evaluator = QualityRewardModel(
            alpha=alpha, beta=beta, gamma=gamma, b_target=b_target, tau_b=tau_b
        )

    def compute_batch_rewards(
        self,
        rollouts: Optional[List[str]] = None,
        target_length: int = 1200,
        context_summary: str = "",
        prev_chapter_tail: Optional[str] = None,
        l_lower: Optional[int] = None,
        l_upper: Optional[int] = None,
        l_max: Optional[int] = None,
        completions: Optional[List[str]] = None,
        target_words: Optional[int] = None,
        prev_chapter_end: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Computes total composite rewards for G candidate rollouts.

        Returns:
            Dict containing:
                - 'total_rewards': torch.Tensor of shape (G,) or list of float
                - 'r_length': List[float]
                - 'r_coherence': List[float]
                - 'r_quality': List[float]
                - 'details': sub-components dict
        """
        texts = rollouts if rollouts is not None else (completions or [])
        t_len = target_words if target_words is not None else target_length
        prev_end = prev_chapter_end or prev_chapter_tail or ""

        r_length = reward_length(
            texts,
            target_length=t_len,
            l_lower=l_lower,
            l_upper=l_upper,
            l_max=l_max,
        )

        r_coherence = self.coherence_evaluator(
            completions=texts,
            context_summary=context_summary,
            prev_chapter_tail=prev_end,
            prev_chapter_end=prev_end,
        )

        r_quality = self.quality_evaluator(completions=texts)

        total_rewards = []
        for l_val, c_val, q_val in zip(r_length, r_coherence, r_quality):
            r_tot = self.w_length * l_val + self.w_coherence * c_val + self.w_quality * q_val
            total_rewards.append(round(r_tot, 4))

        if torch is not None:
            try:
                tensor_rewards = torch.tensor(total_rewards, dtype=torch.float32, device=self.device)
            except Exception:
                tensor_rewards = total_rewards
        else:
            tensor_rewards = total_rewards

        return {
            "total_rewards": tensor_rewards,
            "r_length": r_length,
            "r_coherence": r_coherence,
            "r_quality": r_quality,
            "details": {
                "r_length": r_length,
                "r_coherence": r_coherence,
                "r_quality": r_quality,
            },
        }
