"""
Unified Intrinsic Reward Engine for MA-GRPO (Phase 1).
Contains all 3 intrinsic mathematical reward components in a single consolidated file:
    1. R_length: Rule-based O(1) chapter target length compliance.
    2. R_coherence: Sentence embedding transition Cosine Similarity + Anti-Context 3-Gram ratio.
    3. R_quality: Fluency (local PPL), Richness (Shannon Entropy), and Burstiness (Sentence variation).
    4. R_total: Composite weighted sum for GRPO Advantage calculation.

Formula:
    R_total(o_i^{(k)}) = w_1 * R_length + w_2 * R_coherence + w_3 * R_quality
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import yaml



def strip_think_tags(text: str) -> str:
    """
    Removes test-time reasoning tokens inside <think>...</think> if present.
    Ensures rewards are evaluated strictly on the actual chapter prose.
    """
    if not text:
        return ""
    clean = re.sub(r"<think>[\s\S]*?</think>", "", text)
    clean = re.sub(r"<think>[\s\S]*", "", clean)
    return clean.strip()


def count_words(text: str) -> int:
    """Counts words using word-boundary regex, excluding reasoning tokens."""
    if not text or not text.strip():
        return 0
    clean_text = strip_think_tags(text)
    return len(re.findall(r"\b\w+\b", clean_text))


def calculate_length_reward(text: str, target_length: int) -> float:
    """
    Computes length reward for a single rollout:
        R_length(o) = max(0.0, 1.0 - abs(len(o) - L_target) / L_target)
    """
    if target_length <= 0:
        raise ValueError(f"target_length must be positive, got {target_length}")

    actual_len = count_words(text)
    if actual_len == 0:
        return 0.0

    deviation = abs(actual_len - target_length)
    score = 1.0 - (deviation / float(target_length))
    return max(0.0, float(score))


def batch_length_rewards(texts: List[str], target_length: int) -> List[float]:
    """Computes length rewards for a batch of G rollouts."""
    return [calculate_length_reward(t, target_length) for t in texts]



def extract_words(text: str) -> List[str]:
    """Tokenizes text into lowercase words, excluding reasoning tokens."""
    if not text:
        return []
    clean_text = strip_think_tags(text)
    return re.findall(r"\b\w+\b", clean_text.lower())


def extract_tail_words(text: str, n_words: int = 150) -> str:
    """Extracts the last n words from text."""
    words = text.strip().split()
    return " ".join(words[-n_words:]) if words else ""


def extract_head_words(text: str, n_words: int = 150) -> str:
    """Extracts the first n words from text."""
    words = text.strip().split()
    return " ".join(words[:n_words]) if words else ""


def get_ngrams(words: List[str], n: int = 3) -> List[Tuple[str, ...]]:
    """Generates a list of n-grams from a word list."""
    if len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def calculate_anti_context_ratio(rollout_text: str, context_summary: str) -> float:
    """
    Computes anti-context 3-gram ratio:
        ratio = |Uniq3Grams(rollout) \\ 3Grams(summary)| / |Total3Grams(rollout)|

    Rewards novel phrasing while severely penalizing verbatim copying of
    the coordinator's context summary.
    """
    rollout_words = extract_words(rollout_text)
    if len(rollout_words) < 3:
        return 0.0

    rollout_3grams = get_ngrams(rollout_words, 3)
    total_3grams = len(rollout_3grams)
    if total_3grams == 0:
        return 0.0

    summary_words = extract_words(context_summary)
    summary_3grams: Set[Tuple[str, ...]] = set(get_ngrams(summary_words, 3)) if summary_words else set()

    uniq_rollout_3grams = set(rollout_3grams)
    novel_3grams = uniq_rollout_3grams - summary_3grams

    ratio = len(novel_3grams) / float(total_3grams)
    return min(1.0, max(0.0, float(ratio)))


class CoherenceRewardEvaluator:
    """
    Evaluates narrative continuity between chapters (CosineSim)
    combined with Anti-Context 3-Gram novelty.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Optional[str] = None,
        tail_words: int = 150,
        head_words: int = 150,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tail_words = tail_words
        self.head_words = head_words
        self.model = SentenceTransformer(model_name, device=self.device)

    def calculate_coherence_rewards(
        self,
        rollouts: List[str],
        context_summary: str,
        prev_chapter_tail: Optional[str] = None,
    ) -> List[float]:
        import torch
        import torch.nn.functional as F

        if not rollouts:
            return []

        anti_context_scores = [
            calculate_anti_context_ratio(r, context_summary) for r in rollouts
        ]

        if not prev_chapter_tail or not prev_chapter_tail.strip():
            transition_scores = [1.0] * len(rollouts)
        else:
            tail_snippet = extract_tail_words(prev_chapter_tail, self.tail_words)
            with torch.no_grad():
                tail_emb = self.model.encode(
                    tail_snippet,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=self.device,
                )
                tail_emb = F.normalize(tail_emb, p=2, dim=-1)

                head_snippets = [extract_head_words(r, self.head_words) for r in rollouts]
                head_embs = self.model.encode(
                    head_snippets,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=self.device,
                )
                head_embs = F.normalize(head_embs, p=2, dim=-1)

                cosine_sims = torch.sum(head_embs * tail_emb, dim=-1)
                transition_scores = torch.clamp(cosine_sims, min=0.0, max=1.0).tolist()

        coherence_scores = [
            round(float(trans * anti), 4)
            for trans, anti in zip(transition_scores, anti_context_scores)
        ]
        return coherence_scores



def calculate_vocabulary_entropy(text: str) -> float:
    """
    Computes normalized Shannon Entropy on word frequency distribution:
        H = - sum_v P(v) * log2(P(v))
        R_richness = H / log2(N_words) in [0.0, 1.0]
    """
    words = extract_words(text)
    total_words = len(words)
    if total_words <= 1:
        return 0.0

    counts = Counter(words)
    entropy = 0.0
    for count in counts.values():
        p = count / float(total_words)
        entropy -= p * math.log2(p)

    h_max = math.log2(total_words)
    if h_max <= 0:
        return 0.0

    return min(1.0, max(0.0, float(entropy / h_max)))


def calculate_burstiness(
    text: str,
    mu_0: float = 6.0,
    sigma_0: float = 4.0,
) -> float:
    """
    Measures sentence length variation (pacing):
        R_burstiness = Sigmoid((StdDev(sentence_lengths) - mu_0) / sigma_0)
    """
    clean_text = strip_think_tags(text)
    sentences = re.split(r"[.!?]+", clean_text)
    sentence_lengths = [
        len(extract_words(s)) for s in sentences if s.strip()
    ]

    if len(sentence_lengths) < 2:
        return 0.0

    std_dev = float(np.std(sentence_lengths))
    z = (std_dev - mu_0) / (sigma_0 if sigma_0 > 0 else 1.0)
    score = 1.0 / (1.0 + math.exp(-z))
    return min(1.0, max(0.0, float(score)))


class QualityRewardEvaluator:
    """
    Intrinsic prose quality evaluator measuring local Perplexity (Qwen2.5-0.5B),
    vocabulary entropy, and sentence burstiness.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: Optional[str] = None,
        torch_dtype: Any = None,
        sigma_ref: float = 10.0,
        alpha: float = 0.40,
        beta: float = 0.35,
        gamma: float = 0.25,
        mu_0: float = 6.0,
        sigma_0: float = 4.0,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sigma_ref = sigma_ref
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        ).to(self.device)
        self.model.eval()

    def calculate_perplexity(self, texts: List[str]) -> List[float]:
        """Calculates token-level perplexity for a batch of texts."""
        import torch

        if not texts:
            return []

        ppl_scores = []
        with torch.no_grad():
            for text in texts:
                if not text.strip():
                    ppl_scores.append(100.0)
                    continue

                encodings = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(self.device)

                input_ids = encodings.input_ids
                outputs = self.model(input_ids, labels=input_ids.clone())
                loss = outputs.loss
                ppl = math.exp(loss.item()) if loss.item() < 20 else 1e5
                ppl_scores.append(min(1e4, float(ppl)))

        return ppl_scores

    def calculate_quality_rewards(self, rollouts: List[str]) -> Dict[str, List[float]]:
        if not rollouts:
            return {"total": [], "fluency": [], "richness": [], "burstiness": [], "ppl": []}

        ppls = self.calculate_perplexity(rollouts)
        fluency_scores = [round(math.exp(-p / self.sigma_ref), 4) for p in ppls]
        richness_scores = [round(calculate_vocabulary_entropy(r), 4) for r in rollouts]
        burstiness_scores = [round(calculate_burstiness(r, self.mu_0, self.sigma_0), 4) for r in rollouts]

        total_quality = [
            round(self.alpha * f + self.beta * r + self.gamma * b, 4)
            for f, r, b in zip(fluency_scores, richness_scores, burstiness_scores)
        ]

        return {
            "total": total_quality,
            "fluency": fluency_scores,
            "richness": richness_scores,
            "burstiness": burstiness_scores,
            "ppl": [round(p, 2) for p in ppls],
        }



class CompositeRewardEngine:
    """
    Main unified reward engine for MA-GRPO. Combines R_length, R_coherence,
    and R_quality into a single mathematical reward tensor for G rollouts.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        try:
            import torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            self.device = device or "cpu"

        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                config = cfg.get("rewards", {})
        elif config is None:
            config = {}

        weights = config.get("weights", {})
        self.w_length = float(weights.get("w_length", 0.35))
        self.w_coherence = float(weights.get("w_coherence", 0.35))
        self.w_quality = float(weights.get("w_quality", 0.30))

        coh_cfg = config.get("coherence", {})
        self.coherence_evaluator = CoherenceRewardEvaluator(
            model_name=coh_cfg.get("embedding_model", "BAAI/bge-small-en-v1.5"),
            device=self.device,
            tail_words=int(coh_cfg.get("tail_words", 150)),
            head_words=int(coh_cfg.get("head_words", 150)),
        )

        qual_cfg = config.get("quality", {})
        self.quality_evaluator = QualityRewardEvaluator(
            model_name=qual_cfg.get("fluency_evaluator", "Qwen/Qwen2.5-0.5B"),
            device=self.device,
            sigma_ref=float(qual_cfg.get("fluency_sigma_ref", 10.0)),
            alpha=float(qual_cfg.get("alpha_fluency", 0.40)),
            beta=float(qual_cfg.get("beta_richness", 0.35)),
            gamma=float(qual_cfg.get("gamma_burstiness", 0.25)),
            mu_0=float(qual_cfg.get("burstiness_mu0", 6.0)),
            sigma_0=float(qual_cfg.get("burstiness_sigma0", 4.0)),
        )

    def compute_batch_rewards(
        self,
        rollouts: List[str],
        target_length: int,
        context_summary: str,
        prev_chapter_tail: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Computes total composite rewards for G candidate rollouts.

        Returns:
            Dict containing:
                - 'total_rewards': torch.Tensor of shape (G,)
                - 'r_length': List[float]
                - 'r_coherence': List[float]
                - 'r_quality': List[float]
                - 'details': sub-components (fluency, richness, burstiness, ppl)
        """
        r_length = batch_length_rewards(rollouts, target_length)

        r_coherence = self.coherence_evaluator.calculate_coherence_rewards(
            rollouts=rollouts,
            context_summary=context_summary,
            prev_chapter_tail=prev_chapter_tail,
        )

        quality_res = self.quality_evaluator.calculate_quality_rewards(rollouts)
        r_quality = quality_res["total"]

        total_rewards = []
        for l_val, c_val, q_val in zip(r_length, r_coherence, r_quality):
            r_tot = self.w_length * l_val + self.w_coherence * c_val + self.w_quality * q_val
            total_rewards.append(round(r_tot, 4))

        try:
            import torch
            tensor_rewards = torch.tensor(total_rewards, dtype=torch.float32, device=self.device)
        except ImportError:
            tensor_rewards = total_rewards

        return {
            "total_rewards": tensor_rewards,
            "r_length": r_length,
            "r_coherence": r_coherence,
            "r_quality": r_quality,
            "details": {
                "fluency": quality_res["fluency"],
                "richness": quality_res["richness"],
                "burstiness": quality_res["burstiness"],
                "ppl": quality_res["ppl"],
            },
        }
