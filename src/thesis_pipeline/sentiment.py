#!/usr/bin/env python3
"""
Targeted transition sentiment from cached 10-K text (pure Python, no API).

Pipeline:
1) Pick latest cached filing per ticker-year from cache_manifest.csv
2) Extract Item 1 (Business) section
3) Split into sentences
4) Keep sentences that mention transition terms
5) Score sentence tone using small finance + transition lexicons
6) Aggregate to filing-level stance metrics
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from tqdm import tqdm

from thesis_pipeline.edgar_io import html_to_text


# Match numeric and common OCR/roman variants (e.g., "Item IA", "Item II"),
# including spaced OCR variants such as "I TEM 1".
ITEM_HEADER_RE = re.compile(
    r"\b(?:item|i\s*tem)\s+([0-9]{1,2}[a-z]?|[ivxlcdm]{1,4}[a-z]?)(?:\s*[\.\-:—]\s*|\s+)",
    re.I,
)
ITEM12_COMBINED_RE = re.compile(
    r"\bitems?\s+1\s*[\.\)]?\s*(?:and|&)\s*2\s*[\.\-:—)]?\s*(?:business\s+and\s+properties)?",
    re.I,
)
ITEM1_BUSINESS_RE = re.compile(
    r"\b(?:item|i\s*tem)\s+1\s+(?:business|description\s+of\s+business)\b",
    re.I,
)
ITEM12_BUSINESS_RE = re.compile(
    r"\bitems?\s+1\s*[\.\)]?\s*(?:and|&)\s*2\s+(?:business|business\s+and\s+properties|properties)\b",
    re.I,
)
ITEM1_ITEM2_INLINE_RE = re.compile(
    r"\b(?:item|i\s*tem)\s+1\s*[\.\-:]?\s*(?:business|description\s+of\s+business)\s+and\s+"
    r"(?:item|i\s*tem)\s+2\s*[\.\-:]?\s*(?:properties|property)\b",
    re.I,
)
ITEM1A_INLINE_RE = re.compile(r"\b(?:item|i\s*tem)\s*(?:1a|ia|l[a])\s*[\.\-:]?", re.I)
RISK_FACTORS_RE = re.compile(r"(?:^|\n)\s*risk\s+factors\b", re.I)
RISK_SENTENCE_RE = re.compile(
    r"\b(?:risk factors?|risks?\s+related\s+to|subject to a number of risks?|high-risk activities|"
    r"regulatory compliance issues|catastrophic events|material adverse|significant risks?|"
    r"risks?,\s*including)\b",
    re.I,
)
TOC_NOISE_RE = re.compile(r"\btable\s+of\s+contents\b|\bpart\s+[ivxlcdm]+\b", re.I)
RISK_BOILERPLATE_RE = re.compile(
    r"\b(?:could adversely affect|subject to a number of risks|principal risks?|"
    r"failure to (?:obtain|maintain|complete)|project delays?|cost overruns?|"
    r"including the following|may be subject to significant risks?)\b",
    re.I,
)
RISK_CONTEXT_RE = re.compile(
    r"\b(?:factors affecting|could be affected by numerous factors|may be affected by|"
    r"subject to uncertainty|uncertain outcomes?|subject to significant uncertainty)\b",
    re.I,
)
ACCOUNTING_MECH_RE = re.compile(
    r"\b(?:net cash flow|investing activities|financing activities|operating activities|"
    r"for the year ended|compared to|receipts?|proceeds?|payment(?:s)?|decreased|increased|"
    r"million|billion|gain on sales?|same period)\b",
    re.I,
)
CORE_TRANSITION_RE = re.compile(
    r"\b(?:energy transition|clean energy|renewable(?:s)?|solar|wind|battery|storage|grid|"
    r"transmission|electrification|electric vehicle|ev charging|hydrogen|carbon capture|ccus|"
    r"decarboniz(?:e|ation)|emissions?|methane|efficien(?:t|cy)|geothermal|biofuel|"
    r"nuclear|heat pump)\b",
    re.I,
)
TRANSITION_ACTION_RE = re.compile(
    r"\b(?:invest(?:ed|ing|ment)?|build(?:ing|s)?|deploy(?:ed|ment)?|expand(?:ed|ing)?|"
    r"improv(?:e|ed|ing)|moderniz(?:e|ation)|decarbon(?:ize|ization)|electrif(?:y|ication)|"
    r"renewable|solar|wind|battery|storage|hydrogen|carbon capture|geothermal|efficien(?:t|cy))\b",
    re.I,
)
TRANSITION_ACTION_VERB_RE = re.compile(
    r"\b(?:invest(?:ed|ing)?|build(?:ing)?|deploy(?:ed|ing)?|expand(?:ed|ing)?|"
    r"improv(?:e|ed|ing)|moderniz(?:e|ed|ing)|decarboniz(?:e|ed|ing)|electrif(?:y|ied|ying)|"
    r"retir(?:e|ed|ing)|commission(?:ed|ing)|construct(?:ed|ing)|install(?:ed|ing)|"
    r"implement(?:ed|ing)|adopt(?:ed|ing)|acquir(?:e|ed|ing))\b",
    re.I,
)
CONTINGENCY_RE = re.compile(r"\b(?:contingent upon|subject to|dependent upon|required to)\b", re.I)
PREREQ_TERMS_RE = re.compile(
    r"\b(?:obtaining|obtain|receipt|receiving|negotiation|negotiating|securing|secure|"
    r"approvals?|permits?|rights of way|interconnect)\b",
    re.I,
)
BULLET_RE = re.compile(r"[•▪◦]")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(])")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")

REPEAT_STOPWORDS = {
    "about",
    "above",
    "after",
    "against",
    "because",
    "before",
    "being",
    "below",
    "between",
    "could",
    "would",
    "should",
    "their",
    "there",
    "these",
    "those",
    "which",
    "while",
    "where",
    "under",
    "through",
    "during",
    "including",
    "regarding",
}


# Compact finance-style polarity lexicons (extend as needed).
POS_WORDS = {
    "accelerate",
    "advantage",
    "benefit",
    "beneficial",
    "competitive",
    "confidence",
    "create",
    "efficient",
    "efficiency",
    "expand",
    "favorable",
    "favourable",
    "gain",
    "growth",
    "improve",
    "improved",
    "improvement",
    "innovative",
    "lead",
    "leading",
    "opportunity",
    "opportunities",
    "optimize",
    "positive",
    "progress",
    "profitable",
    "resilient",
    "strength",
    "strong",
    "successful",
    "support",
    "upside",
    "value",
    "decarbonize",
    "decarbonization",
    "electrification",
    "modernization",
    "renewable",
    "storage",
    "transition",
}

NEG_WORDS = {
    "adverse",
    "barrier",
    "burden",
    "challenge",
    "challenging",
    "compliance",
    "constrained",
    "constraint",
    "costly",
    "decline",
    "delay",
    "downside",
    "exposure",
    "headwind",
    "impairment",
    "lawsuit",
    "litigation",
    "negative",
    "penalty",
    "pressure",
    "regulatory",
    "risk",
    "risky",
    "stranded",
    "uncertain",
    "uncertainty",
    "volatile",
    "volatility",
    "weakness",
    "overrun",
    "underperform",
    "infeasible",
    "curtailment",
}

FAVORABLE_PHRASES = [
    "lower emissions",
    "reduced emissions",
    "cost savings",
    "cost reduction",
    "improve efficiency",
    "improved efficiency",
    "long-term opportunity",
    "long term opportunity",
    "competitive advantage",
    "growth opportunity",
    "strong demand",
    "increasing demand",
    "grid modernization",
    "energy storage",
    "renewable natural gas",
    "carbon capture",
    "clean energy",
    "decarbonization strategy",
]

UNFAVORABLE_PHRASES = [
    "higher costs",
    "compliance cost",
    "regulatory burden",
    "transition risk",
    "policy risk",
    "stranded asset",
    "stranded assets",
    "adverse impact",
    "material adverse",
    "supply constraint",
    "execution risk",
    "cost overrun",
    "permit delay",
    "project delay",
    "net cash flow used in investing activities",
]

NEGATORS = {
    "not",
    "no",
    "never",
    "none",
    "without",
    "lack",
    "lacks",
    "lacking",
    "less",
    "cannot",
    "can't",
    "isn't",
    "wasn't",
    "weren't",
    "don't",
    "doesn't",
    "didn't",
}

INTENSIFIERS = {
    "very": 1.2,
    "highly": 1.3,
    "strongly": 1.35,
    "significantly": 1.45,
    "substantially": 1.45,
    "materially": 1.45,
    "meaningfully": 1.3,
}


def parse_years(raw: str) -> List[int]:
    out: List[int] = []
    for chunk in raw.split(","):
        c = chunk.strip()
        if not c:
            continue
        if "-" in c:
            a, b = c.split("-", 1)
            out.extend(list(range(int(a), int(b) + 1)))
        else:
            out.append(int(c))
    return sorted(set(out))


def parse_tickers(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace(",", " ")
    return sorted({x.strip().upper() for x in text.split() if x.strip()})


def load_terms(path: Path) -> List[str]:
    terms: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        terms.append(t)
    return terms


def build_term_patterns(terms: Sequence[str]) -> List[re.Pattern]:
    pats: List[re.Pattern] = []
    for term in terms:
        esc = re.escape(term.strip())
        esc = esc.replace(r"\ ", r"(?:\s|-)")
        pats.append(re.compile(rf"\b{esc}\b", re.I))
    return pats


def map_text_path(cache_path: str, text_cache_dir: Path) -> Optional[Path]:
    normalized = str(cache_path).replace("\\", "/")
    rel: Optional[Path] = None
    if normalized.startswith("cached_edgar/"):
        rel = Path(normalized.split("cached_edgar/", 1)[1])
    elif normalized.startswith("data/cache/edgar_html/"):
        rel = Path(normalized.split("data/cache/edgar_html/", 1)[1])
    elif "/cached_edgar/" in normalized:
        rel = Path(normalized.split("/cached_edgar/", 1)[1])
    elif "/data/cache/edgar_html/" in normalized:
        rel = Path(normalized.split("/data/cache/edgar_html/", 1)[1])
    else:
        # Fallback: recover path suffix after an 'edgar_html' segment.
        p = Path(normalized)
        parts = list(p.parts)
        if "edgar_html" in parts:
            i = parts.index("edgar_html")
            if i + 1 < len(parts):
                rel = Path(*parts[i + 1 :])
    if rel is None:
        return None
    return text_cache_dir / rel.with_suffix(".txt")


def map_html_path(cache_path: str, cache_dir: Path = Path("data/cache/edgar_html")) -> Optional[Path]:
    normalized = str(cache_path).replace("\\", "/")
    rel: Optional[Path] = None
    if normalized.startswith("cached_edgar/"):
        rel = Path(normalized.split("cached_edgar/", 1)[1])
    elif normalized.startswith("data/cache/edgar_html/"):
        rel = Path(normalized.split("data/cache/edgar_html/", 1)[1])
    elif "/cached_edgar/" in normalized:
        rel = Path(normalized.split("/cached_edgar/", 1)[1])
    elif "/data/cache/edgar_html/" in normalized:
        rel = Path(normalized.split("/data/cache/edgar_html/", 1)[1])
    p = Path(normalized)
    if p.exists():
        return p
    if rel is not None:
        return cache_dir / rel
    parts = list(p.parts)
    if "edgar_html" in parts:
        i = parts.index("edgar_html")
        if i + 1 < len(parts):
            return cache_dir / Path(*parts[i + 1 :])
    return None


def load_cached_filing_text(cache_path: str, text_cache_dir: Path) -> Tuple[Optional[str], Optional[Path], Optional[str]]:
    text_path = map_text_path(cache_path, text_cache_dir)
    if text_path is None:
        return None, None, "unmapped_cache_path"
    if text_path.exists():
        text = text_path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            return text, text_path, None
    html_path = map_html_path(cache_path)
    if html_path is None:
        return None, text_path, "unmapped_cache_path"
    if not html_path.is_absolute():
        html_path = (Path.cwd() / html_path).resolve()
    if not html_path.exists():
        return None, text_path, "missing_text_cache"
    try:
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        text = html_to_text(html)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(text, encoding="utf-8")
        if text.strip():
            return text, text_path, None
        return None, text_path, "empty_text"
    except Exception:
        return None, text_path, "missing_text_cache"


def normalize_item_label(raw: str) -> str:
    x = re.sub(r"\s+", "", str(raw)).upper()
    # Common OCR confusion where "I" and "1" are swapped.
    x = x.replace("L", "1")
    roman_map = {
        "I": "1",
        "IA": "1A",
        "IB": "1B",
        "II": "2",
        "III": "3",
        "IV": "4",
        "V": "5",
        "VI": "6",
        "VII": "7",
        "VIII": "8",
        "IX": "9",
        "X": "10",
    }
    if x in roman_map:
        return roman_map[x]
    return x


def extract_item1_business(text: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Extract Item 1 section using header boundaries.
    Chooses the longest Item 1 -> Item 1A/2 segment to avoid TOC fragments.
    """
    matches = [(m.start(), normalize_item_label(m.group(1))) for m in ITEM_HEADER_RE.finditer(text)]
    if not matches:
        return None, None, None

    starts = [p for p, label in matches if label == "1"]
    combined_like_starts = [m.start() for m in ITEM12_COMBINED_RE.finditer(text)]
    combined_like_starts.extend(m.start() for m in ITEM12_BUSINESS_RE.finditer(text))
    combined_like_starts.extend(m.start() for m in ITEM1_ITEM2_INLINE_RE.finditer(text))
    starts.extend(m.start() for m in ITEM1_BUSINESS_RE.finditer(text))
    starts.extend(combined_like_starts)
    starts = sorted(set(starts))
    if not starts:
        return None, None, None
    combined_like_starts = sorted(set(combined_like_starts))

    best: Optional[Tuple[int, int]] = None
    best_len = -1
    for start in starts:
        end = None
        skip_item2 = start in set(combined_like_starts) or bool(
            re.search(
                r"\b(?:item|i\s*tem)\s+1\b.{0,120}\b(?:item|i\s*tem)\s+2\b",
                text[start : min(len(text), start + 180)],
                re.I | re.S,
            )
        )
        for p, label in matches:
            if p <= start:
                continue
            if label in {"1A", "1B"} or (label == "2" and not skip_item2):
                end = p
                break
        if end is None:
            # Combined Item 1/2 sections typically run until Item 1A or Item 3.
            for p, label in matches:
                if p <= start:
                    continue
                if label in {"1A", "1B", "3"}:
                    end = p
                    break
        if end is None:
            continue
        seg_len = end - start
        if seg_len > best_len:
            best_len = seg_len
            best = (start, end)

    if best is None:
        return None, None, None
    start, end = best
    segment = text[start:end]

    # Defensive trim: if Item 1A/`Risk Factors` is embedded due malformed headers,
    # clip early so only business-overview language is retained.
    boundary_candidates: List[int] = []
    m_item1a = ITEM1A_INLINE_RE.search(segment)
    if m_item1a:
        boundary_candidates.append(m_item1a.start())
    m_risk = RISK_FACTORS_RE.search(segment)
    if m_risk:
        boundary_candidates.append(m_risk.start())
    if boundary_candidates:
        clip = min(boundary_candidates)
        if clip > 1000:
            end = start + clip
            segment = text[start:end]

    if end - start < 300:
        return None, None, None
    return segment, start, end


def sentence_split(text: str, min_chars: int = 30) -> List[str]:
    # Normalize bullets into sentence breaks so risk/action lists are segmented.
    cleaned = BULLET_RE.sub(". ", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    raw = SENT_SPLIT_RE.split(cleaned)
    out = []
    for s in raw:
        s2 = re.sub(r"\s+", " ", s).strip(" -.;:")
        if len(s2) >= min_chars:
            out.append(s2)
    return out


def contains_transition(sentence: str, pats: Sequence[re.Pattern]) -> bool:
    return any(p.search(sentence) for p in pats)


def transition_match_count(sentence: str, pats: Sequence[re.Pattern]) -> int:
    return int(sum(1 for p in pats if p.search(sentence)))


def sentence_filter_reason(sentence: str) -> str:
    s = re.sub(r"\s+", " ", sentence).strip()
    low = s.lower()
    if not s:
        return "empty"
    if TOC_NOISE_RE.search(s):
        return "toc_noise"
    if low.startswith("risk factors") or low.startswith("risks related to"):
        return "risk_header"
    # After sentence_split bullet chars are normalized, so infer list-like text by punctuation/terms.
    semicolon_count = s.count(";")
    dotted_list_count = s.count(" . ")
    prereq_hits = len(PREREQ_TERMS_RE.findall(s))
    has_core_transition = bool(CORE_TRANSITION_RE.search(s))
    if not has_core_transition:
        return "non_transition_context"
    if semicolon_count >= 8 and len(s) > 220:
        return "long_checklist"
    if dotted_list_count >= 2 and semicolon_count >= 1:
        return "long_checklist"
    if RISK_SENTENCE_RE.search(s):
        # Keep short mentions, drop risk-list style passages.
        if semicolon_count >= 2 or len(s) > 210:
            return "risk_boilerplate"
    if RISK_CONTEXT_RE.search(s) and (semicolon_count >= 1 or len(s) > 170):
        return "risk_boilerplate"
    if RISK_BOILERPLATE_RE.search(s) and (semicolon_count >= 1 or len(s) > 190):
        return "risk_boilerplate"
    # Drop long prerequisite/checklist language (common false positives for VADER).
    if CONTINGENCY_RE.search(s) and (semicolon_count >= 3 or prereq_hits >= 5):
        return "contingency_checklist"
    risk_count = len(re.findall(r"\brisk(?:s|y)?\b", low))
    if risk_count >= 3 and (semicolon_count >= 1 or len(s) > 170):
        return "risk_boilerplate"
    if ACCOUNTING_MECH_RE.search(s):
        money_count = len(re.findall(r"(?:\$?\d+(?:\.\d+)?\s*(?:million|billion|%))", low))
        has_comp_delta = bool(
            re.search(r"\b(?:increased|decreased|compared to|primarily due to|offset by)\b", low)
        )
        has_action_verb = bool(TRANSITION_ACTION_VERB_RE.search(s))
        # Suppress accounting-style narration unless clear transition action appears.
        if "net cash flow" in low or "investing activities" in low:
            return "accounting_mechanics"
        if re.search(r"\b(?:gain on sales?|sale of assets?|same period)\b", low) and not has_action_verb:
            return "accounting_mechanics"
        if has_comp_delta and not has_action_verb:
            return "accounting_mechanics"
        if has_comp_delta and not has_core_transition:
            return "accounting_mechanics"
        if money_count >= 2 and has_comp_delta and not has_action_verb:
            return "accounting_mechanics"
        if not TRANSITION_ACTION_RE.search(s) and len(s) > 170:
            return "accounting_mechanics"
    return "ok"


def is_business_overview_sentence(sentence: str) -> bool:
    return sentence_filter_reason(sentence) == "ok"


def normalize_for_repeat(sentence: str) -> str:
    s = sentence.lower()
    s = re.sub(r"\b\d+(?:\.\d+)?\b", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    toks = [t for t in s.split() if len(t) >= 4 and t not in REPEAT_STOPWORDS]
    return " ".join(toks)


def is_cross_year_repeat(
    normalized: str,
    token_set: Set[str],
    prior_entries: Sequence[Tuple[str, Set[str]]],
    jaccard_threshold: float,
) -> bool:
    if not normalized or len(token_set) < 8:
        return False
    for prior_norm, prior_tokens in prior_entries:
        if normalized == prior_norm:
            return True
        union = token_set | prior_tokens
        if not union:
            continue
        jaccard = len(token_set & prior_tokens) / len(union)
        if jaccard >= jaccard_threshold:
            return True
    return False


def _token_weights(toks: Sequence[str], idx: int) -> float:
    left = toks[max(0, idx - 2) : idx]
    w = 1.0
    for tok in left:
        w *= INTENSIFIERS.get(tok, 1.0)
    return w


def _is_negated(toks: Sequence[str], idx: int) -> bool:
    window = toks[max(0, idx - 3) : idx]
    return any(t in NEGATORS for t in window)


def score_sentence_lexicon(sentence: str) -> Dict[str, float]:
    s = sentence.lower()
    toks = [w.lower() for w in WORD_RE.findall(sentence)]

    pos_total = 0.0
    neg_total = 0.0
    for i, tok in enumerate(toks):
        weight = _token_weights(toks, i)
        negated = _is_negated(toks, i)
        if tok in POS_WORDS:
            if negated:
                neg_total += weight
            else:
                pos_total += weight
        elif tok in NEG_WORDS:
            if negated:
                pos_total += weight
            else:
                neg_total += weight

    fav_hits = float(sum(1 for ph in FAVORABLE_PHRASES if ph in s))
    unfav_hits = float(sum(1 for ph in UNFAVORABLE_PHRASES if ph in s))
    pos_total += fav_hits
    neg_total += unfav_hits
    denom = pos_total + neg_total + 1
    polarity = float((pos_total - neg_total) / denom)

    return {
        "polarity": polarity,
        "pos_hits": float(pos_total),
        "neg_hits": float(neg_total),
    }


class SentenceScorer:
    def __init__(self, method: str, finbert_model: str, batch_size: int, finbert_max_length: int) -> None:
        self.method = method
        self.batch_size = batch_size
        self.finbert_model = finbert_model
        self.finbert_max_length = int(finbert_max_length)
        self._vader = None
        self._finbert = None
        self._init_backend()

    def _init_backend(self) -> None:
        if self.method == "lexicon":
            return
        if self.method == "vader":
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            except Exception as e:
                raise RuntimeError(
                    "VADER backend requires `vaderSentiment`. Install with: "
                    "`./.venv/bin/pip install vaderSentiment`"
                ) from e
            self._vader = SentimentIntensityAnalyzer()
            return
        if self.method == "finbert":
            try:
                from transformers import pipeline
                import torch
            except Exception as e:
                raise RuntimeError(
                    "FinBERT backend requires `transformers` and `torch`. Install with: "
                    "`./.venv/bin/pip install transformers torch`"
                ) from e
            device: object = -1
            if torch.cuda.is_available():
                device = 0
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            self._finbert = pipeline(
                "text-classification",
                model=self.finbert_model,
                tokenizer=self.finbert_model,
                top_k=None,
                truncation=True,
                max_length=self.finbert_max_length,
                device=device,
            )
            return
        raise ValueError(f"Unknown scorer method: {self.method}")

    def _score_vader(self, sentence: str) -> Dict[str, float]:
        if self._vader is None:
            raise RuntimeError("VADER scorer not initialized")
        res = self._vader.polarity_scores(sentence)
        return {
            "polarity": float(res.get("compound", 0.0)),
            "pos_hits": float(res.get("pos", np.nan)),
            "neg_hits": float(res.get("neg", np.nan)),
        }

    @staticmethod
    def _normalize_hf_scores(raw: object) -> List[Dict[str, float]]:
        """
        Handle transformers pipeline output shape differences across versions:
        - list[dict(label, score)]
        - dict(label, score)
        - list[list[dict(label, score)]]
        """
        if isinstance(raw, dict):
            return [raw]
        if isinstance(raw, list):
            if not raw:
                return []
            if isinstance(raw[0], dict):
                return raw  # type: ignore[return-value]
            if isinstance(raw[0], list):
                flat: List[Dict[str, float]] = []
                for group in raw:
                    if isinstance(group, list):
                        for item in group:
                            if isinstance(item, dict):
                                flat.append(item)
                return flat
        return []

    @classmethod
    def _parse_hf_labels(cls, scores: object) -> Tuple[float, float]:
        rows = cls._normalize_hf_scores(scores)
        pos = 0.0
        neg = 0.0
        for row in rows:
            label = str(row.get("label", "")).lower()
            val = float(row.get("score", 0.0))
            if "pos" in label:
                pos = val
            elif "neg" in label:
                neg = val
        return pos, neg

    def score_many(self, sentences: Sequence[str]) -> List[Dict[str, float]]:
        if not sentences:
            return []
        if self.method == "lexicon":
            return [score_sentence_lexicon(s) for s in sentences]
        if self.method == "vader":
            return [self._score_vader(s) for s in sentences]
        if self.method == "finbert":
            if self._finbert is None:
                raise RuntimeError("FinBERT scorer not initialized")
            out: List[Dict[str, float]] = []
            for i in range(0, len(sentences), self.batch_size):
                batch = list(sentences[i : i + self.batch_size])
                preds = self._finbert(batch)
                if batch and preds and isinstance(preds, list) and isinstance(preds[0], dict):
                    # Version where batch returns list[dict(label,score)] with one label per sentence.
                    preds_by_sentence = [[x] for x in preds]
                else:
                    preds_by_sentence = preds
                for scores in preds_by_sentence:
                    pos, neg = self._parse_hf_labels(scores)
                    out.append(
                        {
                            "polarity": float(pos - neg),
                            "pos_hits": float(pos),
                            "neg_hits": float(neg),
                        }
                    )
            return out
        raise ValueError(f"Unknown scorer method: {self.method}")


def select_latest_cached_filing(
    manifest: pd.DataFrame, tickers: Sequence[str], years: Sequence[int]
) -> pd.DataFrame:
    d = manifest.copy()
    d["ticker"] = d["ticker"].astype(str).str.strip().str.upper()
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["filing_date"] = pd.to_datetime(d["filing_date"], errors="coerce")
    d = d[d["ticker"].isin(set(tickers))]
    d = d[d["year"].isin(set(years))]
    d = d[d["status"].isin(["cached", "already_cached"])]
    d = d.sort_values(["ticker", "year", "filing_date"], ascending=[True, True, False])
    d = d.drop_duplicates(["ticker", "year"], keep="first")
    return d[["ticker", "year", "cik", "accession", "doc", "filing_date", "cache_path"]].copy()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Transition sentiment in Item 1 for cached 10-K text")
    ap.add_argument("--tickers-file", type=str, default="data/raw/universe/transition_100_tickers.txt")
    ap.add_argument("--cache-manifest", type=str, default="data/cache/edgar_html/cache_manifest.csv")
    ap.add_argument("--text-cache-dir", type=str, default="data/cache/edgar_text")
    ap.add_argument("--substantive-file", type=str, default="assets/terms/substantive_terms.txt")
    ap.add_argument("--years", type=str, default="2015-2024")
    ap.add_argument(
        "--scorer",
        type=str,
        default="lexicon",
        choices=["lexicon", "vader", "finbert"],
        help="Sentence sentiment backend. finbert is highest-standard local model.",
    )
    ap.add_argument("--finbert-model", type=str, default="ProsusAI/finbert")
    ap.add_argument("--finbert-max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--min-sentence-chars", type=int, default=30)
    ap.add_argument(
        "--min-transition-term-hits",
        type=int,
        default=1,
        help="Require at least K substantive-term matches per sentence.",
    )
    ap.add_argument("--max-filings", type=int, default=0, help="Process at most N filings (0 = all).")
    ap.add_argument(
        "--disable-business-filter",
        action="store_true",
        help="Disable risk/accounting sentence filters inside Item 1.",
    )
    ap.add_argument(
        "--drop-cross-year-boilerplate",
        action="store_true",
        help="Drop near-duplicate transition sentences repeated for the same ticker in prior years.",
    )
    ap.add_argument(
        "--repeat-jaccard-threshold",
        type=float,
        default=0.88,
        help="Token-set Jaccard threshold for cross-year near-duplicate removal (default: 0.88).",
    )
    ap.add_argument(
        "--progress-log-every-filings",
        type=int,
        default=500,
        help="Emit a plain filing-progress log line every N filings (0 disables heartbeat logs).",
    )
    ap.add_argument("--out-file", type=str, default="data/processed/search/sentiment/transition_item1_sentiment.csv")
    ap.add_argument("--sentences-out-file", type=str, default="data/processed/search/sentiment/transition_item1_sentence_scores.csv")
    ap.add_argument("--dropped-file", type=str, default="data/processed/search/sentiment/transition_item1_dropped.csv")
    args = ap.parse_args()

    tickers = parse_tickers(Path(args.tickers_file))
    years = parse_years(args.years)
    text_cache_dir = Path(args.text_cache_dir)

    manifest = pd.read_csv(args.cache_manifest)
    chosen = select_latest_cached_filing(manifest, tickers=tickers, years=years)
    chosen = chosen.sort_values(["ticker", "year"]).reset_index(drop=True)
    if args.max_filings and args.max_filings > 0:
        chosen = chosen.head(int(args.max_filings)).copy()

    transition_pats = build_term_patterns(load_terms(Path(args.substantive_file)))
    scorer = SentenceScorer(
        method=args.scorer,
        finbert_model=args.finbert_model,
        batch_size=args.batch_size,
        finbert_max_length=int(args.finbert_max_length),
    )

    rows: List[Dict[str, object]] = []
    sent_rows: List[Dict[str, object]] = []
    dropped: List[Dict[str, object]] = []
    prior_sentences_by_ticker: Dict[str, List[Tuple[str, Set[str]]]] = {}

    total_filings = len(chosen)
    filing_start = time.perf_counter()
    filing_heartbeat_every = max(0, int(args.progress_log_every_filings))

    for idx, (_, r) in enumerate(tqdm(chosen.iterrows(), total=total_filings, desc="Scoring filings", unit="filing"), start=1):
        if filing_heartbeat_every and (idx == 1 or idx % filing_heartbeat_every == 0 or idx == total_filings):
            elapsed = time.perf_counter() - filing_start
            avg_per_filing = elapsed / float(idx)
            filings_left = max(0, total_filings - idx)
            eta = avg_per_filing * filings_left
            print(
                f"[FILING_PROGRESS] {idx}/{total_filings} | "
                f"elapsed={format_duration(elapsed)} | "
                f"avg_per_filing={avg_per_filing:.2f}s | "
                f"filings_left={filings_left} | eta={format_duration(eta)}",
                flush=True,
            )
        ticker = str(r["ticker"])
        year = int(r["year"])
        text, text_path, load_reason = load_cached_filing_text(str(r["cache_path"]), text_cache_dir)
        if text is None or text_path is None:
            dropped.append(
                {
                    "ticker": ticker,
                    "year": year,
                    "reason": load_reason or "missing_text_cache",
                    "text_path": str(text_path) if text_path is not None else None,
                }
            )
            continue

        item1_text, item1_start, item1_end = extract_item1_business(text)
        if not item1_text:
            dropped.append({"ticker": ticker, "year": year, "reason": "item1_not_found"})
            continue

        sentences = sentence_split(item1_text, min_chars=int(args.min_sentence_chars))
        min_term_hits = max(1, int(args.min_transition_term_hits))
        transition_candidates = [
            s for s in sentences if transition_match_count(s, transition_pats) >= min_term_hits
        ]
        transition_sentences: List[str] = []
        filtered_risk = 0
        filtered_accounting = 0
        filtered_noise = 0
        for s in transition_candidates:
            if args.disable_business_filter:
                transition_sentences.append(s)
                continue
            reason = sentence_filter_reason(s)
            if reason == "ok":
                transition_sentences.append(s)
            elif reason in {"risk_header", "risk_boilerplate"}:
                filtered_risk += 1
            elif reason == "accounting_mechanics":
                filtered_accounting += 1
            else:
                filtered_noise += 1

        scored_sentences: List[str] = []
        filtered_repeat = 0
        prior_entries = prior_sentences_by_ticker.get(ticker, [])
        if args.drop_cross_year_boilerplate:
            for s in transition_sentences:
                norm = normalize_for_repeat(s)
                toks = set(norm.split())
                if is_cross_year_repeat(
                    normalized=norm,
                    token_set=toks,
                    prior_entries=prior_entries,
                    jaccard_threshold=float(args.repeat_jaccard_threshold),
                ):
                    filtered_repeat += 1
                    continue
                scored_sentences.append(s)
                if norm:
                    prior_entries.append((norm, toks))
            prior_sentences_by_ticker[ticker] = prior_entries
        else:
            scored_sentences = transition_sentences

        scores = scorer.score_many(scored_sentences)
        polarity = np.array([x["polarity"] for x in scores], dtype=float) if scores else np.array([], dtype=float)
        pos_hits = np.array([x["pos_hits"] for x in scores], dtype=float) if scores else np.array([], dtype=float)
        neg_hits = np.array([x["neg_hits"] for x in scores], dtype=float) if scores else np.array([], dtype=float)

        if len(polarity) > 0:
            pos_share = float(np.mean(polarity > 0.05))
            neg_share = float(np.mean(polarity < -0.05))
            stance = pos_share - neg_share
            mean_pol = float(np.mean(polarity))
            median_pol = float(np.median(polarity))
        else:
            pos_share = np.nan
            neg_share = np.nan
            stance = np.nan
            mean_pol = np.nan
            median_pol = np.nan

        rows.append(
            {
                "ticker": ticker,
                "year": year,
                "cik": r["cik"],
                "accession": r["accession"],
                "doc": r["doc"],
                "filing_date": r["filing_date"],
                "text_path": str(text_path),
                "item1_start": item1_start,
                "item1_end": item1_end,
                "n_item1_sentences": len(sentences),
                "n_transition_sentences_raw": len(transition_candidates),
                "n_transition_sentences": len(scored_sentences),
                "n_transition_filtered_risk": filtered_risk,
                "n_transition_filtered_accounting": filtered_accounting,
                "n_transition_filtered_noise": filtered_noise,
                "n_transition_filtered_repeat": filtered_repeat,
                "scorer": args.scorer,
                "transition_sentiment_mean": mean_pol,
                "transition_sentiment_median": median_pol,
                "transition_pos_share": pos_share,
                "transition_neg_share": neg_share,
                "transition_stance_index": stance,
                "transition_pos_hits_total": float(np.sum(pos_hits)) if len(pos_hits) else np.nan,
                "transition_neg_hits_total": float(np.sum(neg_hits)) if len(neg_hits) else np.nan,
            }
        )

        for i, (sentence, sc) in enumerate(zip(scored_sentences, scores), start=1):
            sent_rows.append(
                {
                    "ticker": ticker,
                    "year": year,
                    "sentence_idx": i,
                    "sentence": sentence,
                    "scorer": args.scorer,
                    "polarity": sc["polarity"],
                    "pos_hits": sc["pos_hits"],
                    "neg_hits": sc["neg_hits"],
                }
            )

    out_file = Path(args.out_file)
    sent_file = Path(args.sentences_out_file)
    dropped_file = Path(args.dropped_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sent_file.parent.mkdir(parents=True, exist_ok=True)
    dropped_file.parent.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(rows).sort_values(["year", "ticker"]) if rows else pd.DataFrame()
    sent_df = pd.DataFrame(sent_rows).sort_values(["year", "ticker", "sentence_idx"]) if sent_rows else pd.DataFrame()
    dropped_df = pd.DataFrame(dropped).sort_values(["year", "ticker", "reason"]) if dropped else pd.DataFrame()

    out_df.to_csv(out_file, index=False)
    sent_df.to_csv(sent_file, index=False)
    dropped_df.to_csv(dropped_file, index=False)

    print(f"[OUT] {out_file} ({len(out_df)} rows)")
    print(f"[OUT] {sent_file} ({len(sent_df)} rows)")
    print(f"[OUT] {dropped_file} ({len(dropped_df)} rows)")
    if not out_df.empty:
        print(
            "[INFO] mean transition_stance_index:",
            float(out_df["transition_stance_index"].dropna().mean()) if out_df["transition_stance_index"].notna().any() else np.nan,
        )


if __name__ == "__main__":
    main()



def compute_forward_1y(returns_file: Path, years: Sequence[int], min_months: int) -> pd.DataFrame:
    ret = pd.read_csv(returns_file, usecols=["Ticker", "DlyCalDt", "DlyRet"])
    ret["ticker"] = ret["Ticker"].astype(str).str.upper().str.strip()
    ret["date"] = pd.to_datetime(ret["DlyCalDt"], errors="coerce")
    ret["ret"] = pd.to_numeric(ret["DlyRet"], errors="coerce")
    ret = ret.dropna(subset=["ticker", "date", "ret"]).copy()

    rows = []
    for y in years:
        start = pd.Timestamp(f"{int(y)}-12-31")
        end = pd.Timestamp(f"{int(y) + 1}-12-31")
        sub = ret[(ret["date"] > start) & (ret["date"] <= end)].copy()
        if sub.empty:
            continue
        grp = sub.groupby("ticker", as_index=False).agg(
            ret_1y=("ret", lambda x: float(np.prod(1.0 + x.values) - 1.0)),
            n_months=("date", lambda x: x.dt.to_period("M").nunique()),
        )
        grp = grp[grp["n_months"] >= int(min_months)].copy()
        if grp.empty:
            continue
        grp["year"] = int(y)
        rows.append(grp[["ticker", "year", "ret_1y", "n_months"]])
    if not rows:
        return pd.DataFrame(columns=["ticker", "year", "ret_1y", "n_months"])
    return pd.concat(rows, ignore_index=True)


def winsorize_by_year(df: pd.DataFrame, col: str, q_low: float, q_high: float) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in df.groupby("year").groups.items():
        s = pd.to_numeric(df.loc[idx, col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        lo = s.quantile(q_low)
        hi = s.quantile(q_high)
        out.loc[idx] = s.clip(lo, hi)
    return out


def zscore_by_year(df: pd.DataFrame, col: str) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in df.groupby("year").groups.items():
        s = pd.to_numeric(df.loc[idx, col], errors="coerce")
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd <= 0:
            continue
        out.loc[idx] = (s - s.mean()) / sd
    return out


def evaluate_sentiment_file(sent_df: pd.DataFrame, fwd: pd.DataFrame, missing_score_policy: str = "drop") -> pd.DataFrame:
    d = sent_df.copy()
    d["ticker"] = d["ticker"].astype(str).str.upper().str.strip()
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d = d.dropna(subset=["ticker", "year"]).copy()
    d["year"] = d["year"].astype(int)

    m = d.merge(fwd, on=["ticker", "year"], how="inner")
    if m.empty:
        return pd.DataFrame()

    cand: Dict[str, pd.Series] = {}
    for c in [
        "transition_stance_index",
        "transition_sentiment_mean",
        "transition_sentiment_median",
        "transition_pos_share",
        "transition_neg_share",
        "transition_pos_hits_total",
        "transition_neg_hits_total",
        "n_transition_sentences",
        "n_transition_filtered_repeat",
        "hit_balance",
        "hit_ratio",
    ]:
        if c in m.columns:
            cand[c] = pd.to_numeric(m[c], errors="coerce")

    rows = []
    for score_name, raw in cand.items():
        base = m[["ticker", "year", "ret_1y"]].copy()
        base[score_name] = raw
        if missing_score_policy == "zero":
            base[score_name] = base[score_name].fillna(0.0)
        elif missing_score_policy != "drop":
            raise ValueError(f"Unsupported missing_score_policy: {missing_score_policy}")

        for transform in ["raw", "winsor_1_99", "winsor_5_95"]:
            d2 = base.copy()
            if transform == "winsor_1_99":
                d2[score_name] = winsorize_by_year(d2, score_name, 0.01, 0.99)
            elif transform == "winsor_5_95":
                d2[score_name] = winsorize_by_year(d2, score_name, 0.05, 0.95)
            d2[f"z_{score_name}"] = zscore_by_year(d2, score_name)
            d2 = d2.dropna(subset=[f"z_{score_name}", "ret_1y"]).copy()
            if len(d2) < 30:
                continue
            fit = smf.ols(f"ret_1y ~ z_{score_name} + C(year)", data=d2).fit(cov_type="HC1")
            rows.append({
                "score_name": score_name,
                "transform": transform,
                "missing_score_policy": missing_score_policy,
                "N": int(fit.nobs),
                "beta": float(fit.params.get(f"z_{score_name}", np.nan)),
                "t_stat": float(fit.tvalues.get(f"z_{score_name}", np.nan)),
                "p_value": float(fit.pvalues.get(f"z_{score_name}", np.nan)),
                "adj_r2": float(fit.rsquared_adj),
            })
    return pd.DataFrame(rows).sort_values(["p_value", "adj_r2"], ascending=[True, False]) if rows else pd.DataFrame()
