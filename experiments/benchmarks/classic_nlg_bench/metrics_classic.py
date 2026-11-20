"""
Traditional Surface-Level NLG Metrics for Full-Story Evaluation (`ClassicNLG-Bench`):
  - Reference-based Overlap:
      BLEU-1, BLEU-2, BLEU-3, BLEU-4 (Papineni et al. with add-1 smoothing + brevity penalty)
      ROUGE-L (Exact LCS F1 for short texts; Lin 2004 ROUGE-Lsum union-LCS for long stories)
  - Reference-free Lexical Diversity:
      Distinct-1 (%), Distinct-2 (%), Distinct-3 (%)
      Entropy-2 (Bigram Shannon Entropy in bits)
  - Reference-free Fluency / Surprisal:
      Perplexity (strided sliding-window causal LM perplexity via HuggingFace, e.g. gpt2 / Qwen2.5-0.5B)
"""

from __future__ import annotations

import gc
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.rewards import strip_think_tags


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", strip_think_tags(text).lower())


def compute_bleu_1_to_4(candidate_text: str, reference_text: str) -> Dict[str, Optional[float]]:
    """
    Computes cumulative BLEU-1, BLEU-2, BLEU-3, BLEU-4 (0–100 scale) with standard
    brevity penalty and Chen & Cherry Method 1 (+1 smoothing for n >= 2).
    Returns None values when `reference_text` is empty.
    """
    cand = _tokenize_words(candidate_text)
    ref = _tokenize_words(reference_text)
    if not cand or not ref:
        return {"BLEU-1": None, "BLEU-2": None, "BLEU-3": None, "BLEU-4": None}

    c_len = len(cand)
    r_len = len(ref)
    bp = 1.0 if c_len > r_len else math.exp(1.0 - (r_len / float(c_len)))

    precisions: List[float] = []
    for n in range(1, 5):
        if c_len < n or r_len < n:
            precisions.append(1e-9)
            continue
        cand_ngrams = Counter(tuple(cand[i : i + n]) for i in range(c_len - n + 1))
        ref_ngrams = Counter(tuple(ref[i : i + n]) for i in range(r_len - n + 1))
        clipped = sum(min(cnt, ref_ngrams[ng]) for ng, cnt in cand_ngrams.items())
        total = max(1, c_len - n + 1)
        if n == 1:
            p_n = clipped / float(total) if clipped > 0 else 1e-9
        else:
            p_n = (clipped + 1.0) / float(total + 1.0)
        precisions.append(max(1e-9, p_n))

    scores: Dict[str, Optional[float]] = {}
    log_sum = 0.0
    for k in range(1, 5):
        log_sum += math.log(precisions[k - 1])
        bleu_k = bp * math.exp(log_sum / float(k)) * 100.0
        scores[f"BLEU-{k}"] = round(max(0.0, min(100.0, bleu_k)), 2)
    return scores


def _lcs_length_dp(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    """Space-optimized O(min(N, M)) dynamic programming LCS length."""
    if not seq_a or not seq_b:
        return 0
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a
    prev = [0] * (len(seq_b) + 1)
    for tok_a in seq_a:
        curr = [0] * (len(seq_b) + 1)
        for j, tok_b in enumerate(seq_b, 1):
            if tok_a == tok_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = curr[j - 1] if curr[j - 1] >= prev[j] else prev[j]
        prev = curr
    return prev[-1]


def _lcs_matched_indices(ref_sent: Sequence[str], cand_sent: Sequence[str]) -> set:
    """Returns indices in `ref_sent` participating in an LCS with `cand_sent`."""
    m, n = len(ref_sent), len(cand_sent)
    if m == 0 or n == 0:
        return set()
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        ri = ref_sent[i - 1]
        row = dp[i]
        prev_row = dp[i - 1]
        for j in range(1, n + 1):
            if ri == cand_sent[j - 1]:
                row[j] = prev_row[j - 1] + 1
            else:
                row[j] = row[j - 1] if row[j - 1] >= prev_row[j] else prev_row[j]
    matched = set()
    i, j = m, n
    while i > 0 and j > 0:
        if ref_sent[i - 1] == cand_sent[j - 1]:
            matched.add(i - 1)
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return matched


def compute_rouge_l(candidate_text: str, reference_text: str) -> Optional[float]:
    """
    Computes ROUGE-L F1 (0–100 scale):
      - Exact token-level LCS when both sequences <= 600 tokens.
      - Fast summary-level ROUGE-Lsum (Lin 2004 union LCS across top candidate sentences)
        when evaluating multi-thousand-word stories.
    """
    cand_toks = _tokenize_words(candidate_text)
    ref_toks = _tokenize_words(reference_text)
    if not cand_toks or not ref_toks:
        return None

    if len(cand_toks) <= 600 and len(ref_toks) <= 600:
        lcs = _lcs_length_dp(ref_toks, cand_toks)
    else:
        ref_sents = [
            _tokenize_words(s)
            for s in re.split(r"[.!?\n]+", strip_think_tags(reference_text))
            if _tokenize_words(s)
        ]
        cand_sents = [
            _tokenize_words(s)
            for s in re.split(r"[.!?\n]+", strip_think_tags(candidate_text))
            if _tokenize_words(s)
        ]
        cand_sets = [set(s) for s in cand_sents]
        total_union_lcs = 0
        for r_s in ref_sents:
            r_set = set(r_s)
            # Pre-filter top-8 candidate sentences by unigram overlap for fast long-story evaluation
            scored = [
                (len(r_set & c_set), idx)
                for idx, c_set in enumerate(cand_sets)
                if r_set & c_set
            ]
            scored.sort(reverse=True)
            union_idx = set()
            for _, c_idx in scored[:8]:
                union_idx |= _lcs_matched_indices(r_s[:120], cand_sents[c_idx][:120])
                if len(union_idx) == len(r_s):
                    break
            total_union_lcs += len(union_idx)
        lcs = min(total_union_lcs, len(ref_toks), len(cand_toks))

    prec = lcs / float(len(cand_toks))
    rec = lcs / float(len(ref_toks))
    if prec + rec == 0.0:
        return 0.0
    f1 = (2.0 * prec * rec) / (prec + rec)
    return round(f1 * 100.0, 2)


def compute_distinct_and_entropy(text: str) -> Dict[str, float]:
    """
    Computes:
      - Distinct-1 (%): unique unigrams / total unigrams
      - Distinct-2 (%): unique bigrams / total bigrams
      - Distinct-3 (%): unique trigrams / total trigrams
      - Entropy-2 (bits): Shannon entropy of the bigram distribution -sum p(b) log2 p(b)
    """
    toks = _tokenize_words(text)
    n = len(toks)
    if n == 0:
        return {"Distinct-1 (%)": 0.0, "Distinct-2 (%)": 0.0, "Distinct-3 (%)": 0.0, "Entropy-2": 0.0}

    d1 = round((len(set(toks)) / float(n)) * 100.0, 2)
    if n < 2:
        return {"Distinct-1 (%)": d1, "Distinct-2 (%)": 100.0, "Distinct-3 (%)": 100.0, "Entropy-2": 0.0}

    bigrams = list(zip(toks, toks[1:]))
    bi_counts = Counter(bigrams)
    n_bi = len(bigrams)
    d2 = round((len(bi_counts) / float(n_bi)) * 100.0, 2)
    ent2 = round(-sum((c / n_bi) * math.log2(c / n_bi) for c in bi_counts.values()), 4)

    if n < 3:
        d3 = 100.0
    else:
        trigrams = set(zip(toks, toks[1:], toks[2:]))
        d3 = round((len(trigrams) / float(n - 2)) * 100.0, 2)

    return {
        "Distinct-1 (%)": d1,
        "Distinct-2 (%)": d2,
        "Distinct-3 (%)": d3,
        "Entropy-2": ent2,
    }


class SlidingWindowPerplexityScorer:
    """
    Computes document-level Perplexity over ultra-long stories (8k–10k+ words) using
    strided sliding windows so context length limits never truncate or crash evaluation.
    Defaults to `gpt2` (fast, standard literature baseline) or any causal LM (e.g. `Qwen/Qwen2.5-0.5B`).
    """

    def __init__(
        self,
        model_id: str = "gpt2",
        stride: int = 512,
        max_length: int = 1024,
        max_eval_tokens: int = 16384,
        use_mock: bool = False,
    ):
        self.model_id = model_id
        self.stride = stride
        self.max_length = max_length
        self.max_eval_tokens = max_eval_tokens
        self.use_mock = use_mock
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        if self.use_mock or self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            use_cuda = torch.cuda.is_available()
            dtype = torch.float16 if use_cuda else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            if use_cuda:
                self.model = self.model.to("cuda")
            self.model.eval()
            cfg_max = getattr(self.model.config, "n_positions", None) or getattr(
                self.model.config, "max_position_embeddings", self.max_length
            )
            if isinstance(cfg_max, int) and cfg_max > 0:
                self.max_length = min(self.max_length, cfg_max)
        except Exception as exc:
            print(f"[PerplexityScorer] Warning: failed loading {self.model_id} ({exc}); falling back to mock proxy.")
            self.use_mock = True

    def unload(self) -> None:
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def score(self, text: str) -> float:
        clean = strip_think_tags(text).strip()
        if not clean:
            return 0.0

        if not self.use_mock and self.model is None:
            self.load()

        if self.use_mock or self.model is None or self.tokenizer is None:
            return self._mock_perplexity(clean)

        import torch

        enc = self.tokenizer(clean, return_tensors="pt", truncation=False)
        input_ids = enc["input_ids"][0][: self.max_eval_tokens]
        seq_len = int(input_ids.size(0))
        if seq_len < 2:
            return 1.0

        dev = next(self.model.parameters()).device
        nll_sum = 0.0
        n_tokens = 0
        prev_end = 0

        for begin in range(0, seq_len, self.stride):
            end = min(begin + self.max_length, seq_len)
            trg_len = end - prev_end
            chunk = input_ids[begin:end].unsqueeze(0).to(dev)
            target_ids = chunk.clone()
            if trg_len < chunk.size(1):
                target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                out = self.model(chunk, labels=target_ids)
                # HF CrossEntropyLoss averages over non-ignored shifted labels (trg_len - 1 on first window)
                valid_count = int((target_ids[:, 1:] != -100).sum().item())
                if valid_count > 0:
                    nll_sum += float(out.loss.item()) * valid_count
                    n_tokens += valid_count

            prev_end = end
            if end == seq_len:
                break

        mean_nll = nll_sum / max(1, n_tokens)
        return round(float(math.exp(min(mean_nll, 12.0))), 3)

    @staticmethod
    def _mock_perplexity(text: str) -> float:
        """Deterministic lexical-repetition proxy ONLY for CPU --mock testing."""
        stats = compute_distinct_and_entropy(text)
        # Repetitive text -> artificially low PPL (~8-12); diverse fluent prose -> ~16-26
        d2 = stats["Distinct-2 (%)"] / 100.0
        return round(8.0 + 18.0 * d2, 3)


def evaluate_classic_nlg_single(
    story_text: str,
    reference_text: str = "",
    ppl_scorer: Optional[SlidingWindowPerplexityScorer] = None,
) -> Dict[str, Optional[float]]:
    """Computes all 10 traditional NLG metrics for a single story."""
    bleu = compute_bleu_1_to_4(story_text, reference_text) if reference_text else {
        "BLEU-1": None,
        "BLEU-2": None,
        "BLEU-3": None,
        "BLEU-4": None,
    }
    rouge_l = compute_rouge_l(story_text, reference_text) if reference_text else None
    div = compute_distinct_and_entropy(story_text)
    ppl = ppl_scorer.score(story_text) if ppl_scorer is not None else None

    return {
        "BLEU-1": bleu["BLEU-1"],
        "BLEU-2": bleu["BLEU-2"],
        "BLEU-3": bleu["BLEU-3"],
        "BLEU-4": bleu["BLEU-4"],
        "ROUGE-L": rouge_l,
        "Distinct-1 (%)": div["Distinct-1 (%)"],
        "Distinct-2 (%)": div["Distinct-2 (%)"],
        "Distinct-3 (%)": div["Distinct-3 (%)"],
        "Entropy-2": div["Entropy-2"],
        "Perplexity": ppl,
    }
