#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


import argparse
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULTS = {
    "years": "2015-2024",
    "eval_years": "2015-2024",
    "min_months_forward": 9,
    "batch_size": 64,
    "finbert_model": "ProsusAI/finbert",
    "substantive_files": "assets/terms/substantive_terms.txt,assets/terms/substantive_terms_core_transition.txt",
    "char_grid": "15,20,30,50",
    "min_transition_term_hits_grid": "1,2",
    "repeat_jaccard_grid": "0.80,0.88,0.94",
    "finbert_max_length_grid": "192,256",
    "business_filter_modes": "on,off",
    "repeat_modes": "on,off",
    "missing_score_policy": "both",
    "r1000": {
        "tickers_file": Path("data/raw/universe/tickers.txt"),
        "returns_file": Path("data/raw/returns/daily_ret_10y_full_r1000.csv"),
        "out_dir": Path("data/processed/search/sentiment/r1000"),
    },
    "t100": {
        "tickers_file": Path("data/raw/universe/transition_100_tickers.txt"),
        "returns_file": Path("data/raw/returns/daily_ret_10y.csv"),
        "out_dir": Path("data/processed/search/sentiment/t100"),
    },
}

from thesis_pipeline.sentiment import compute_forward_1y, evaluate_sentiment_file
from thesis_pipeline.sentiment import (
    SentenceScorer,
    build_term_patterns,
    extract_item1_business,
    is_cross_year_repeat,
    load_terms,
    load_cached_filing_text,
    normalize_for_repeat,
    parse_tickers,
    parse_years,
    select_latest_cached_filing,
    sentence_filter_reason,
    sentence_split,
    transition_match_count,
)


@dataclass(frozen=True)
class T100FinBERTConfig:
    substantive_file: str
    min_sentence_chars: int
    business_filter_on: bool
    drop_cross_year_boilerplate: bool
    repeat_jaccard_threshold: float
    min_transition_term_hits: int
    finbert_max_length: int

    @property
    def config_id(self) -> str:
        stem = Path(self.substantive_file).stem.lower()
        if stem == "substantive_terms":
            terms = "full"
        elif "core" in stem:
            terms = "core"
        else:
            terms = stem.replace("-", "_")
        filt = "filter_on" if self.business_filter_on else "filter_off"
        rep = "repeat_on" if self.drop_cross_year_boilerplate else "repeat_off"
        j = f"j{str(self.repeat_jaccard_threshold).replace('.', '')}"
        return (
            f"finbert_{terms}_c{self.min_sentence_chars}_h{self.min_transition_term_hits}_"
            f"{filt}_{rep}_{j}_f{self.finbert_max_length}"
        )

    @property
    def substantive_key(self) -> str:
        return sanitize_key(self.substantive_file)


def sanitize_key(path_like: str) -> str:
    stem = Path(path_like).stem.lower()
    return stem.replace("-", "_").replace(" ", "_")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_csv_list(raw: str) -> List[str]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("CSV list cannot be empty.")
    return vals


def parse_int_grid(raw: str) -> List[int]:
    vals = sorted({int(x.strip()) for x in str(raw).split(",") if x.strip()})
    if not vals:
        raise ValueError("Integer grid cannot be empty.")
    return vals


def parse_float_grid(raw: str) -> List[float]:
    vals = sorted({float(x.strip()) for x in str(raw).split(",") if x.strip()})
    if not vals:
        raise ValueError("Float grid cannot be empty.")
    return vals


def parse_bool_modes(raw: str) -> List[bool]:
    vals = []
    for x in parse_csv_list(raw):
        y = x.lower()
        if y in {"on", "true", "1", "yes"}:
            vals.append(True)
        elif y in {"off", "false", "0", "no"}:
            vals.append(False)
        else:
            raise ValueError(f"Unsupported bool mode: {x}")
    return sorted(set(vals))


def build_grid(
    substantive_files: Sequence[str],
    char_grid: Sequence[int],
    min_transition_term_hits_grid: Sequence[int],
    business_filter_modes: Sequence[bool],
    repeat_modes: Sequence[bool],
    repeat_jaccard_grid: Sequence[float],
    finbert_max_length_grid: Sequence[int],
) -> List[T100FinBERTConfig]:
    cfgs: Dict[str, T100FinBERTConfig] = {}
    for sf in substantive_files:
        for chars in char_grid:
            for min_hits in min_transition_term_hits_grid:
                for filt_on in business_filter_modes:
                    for drop_rep in repeat_modes:
                        j_grid = repeat_jaccard_grid if drop_rep else [0.88]
                        for jacc in j_grid:
                            for max_len in finbert_max_length_grid:
                                cfg = T100FinBERTConfig(
                                    substantive_file=sf,
                                    min_sentence_chars=int(chars),
                                    business_filter_on=bool(filt_on),
                                    drop_cross_year_boilerplate=bool(drop_rep),
                                    repeat_jaccard_threshold=float(jacc),
                                    min_transition_term_hits=int(min_hits),
                                    finbert_max_length=int(max_len),
                                )
                                cfgs[cfg.config_id] = cfg
    return [cfgs[k] for k in sorted(cfgs.keys())]


def build_base_cache(
    tickers_file: Path,
    cache_manifest: Path,
    text_cache_dir: Path,
    years_raw: str,
    term_files: Sequence[str],
    min_sentence_chars_floor: int,
    cache_dir: Path,
    max_filings: int = 0,
    progress_log_every_filings: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filings_cache = cache_dir / "filings_base.csv"
    sentences_cache = cache_dir / "candidate_sentences_base.csv"
    dropped_cache = cache_dir / "parser_dropped.csv"

    if filings_cache.exists() and sentences_cache.exists() and dropped_cache.exists():
        filings = pd.read_csv(filings_cache)
        sentences = pd.read_csv(sentences_cache)
        dropped = pd.read_csv(dropped_cache)
        return filings, sentences, dropped

    tickers = parse_tickers(tickers_file)
    years = parse_years(years_raw)
    manifest = pd.read_csv(cache_manifest)
    chosen = select_latest_cached_filing(manifest, tickers=tickers, years=years)
    chosen = chosen.sort_values(["ticker", "year"]).reset_index(drop=True)
    if max_filings and max_filings > 0:
        chosen = chosen.head(int(max_filings)).copy()

    term_specs: Dict[str, List] = {
        sanitize_key(sf): build_term_patterns(load_terms(Path(sf))) for sf in term_files
    }

    filing_rows: List[Dict[str, object]] = []
    sentence_rows: List[Dict[str, object]] = []
    dropped_rows: List[Dict[str, object]] = []

    total_filings = len(chosen)
    cache_start = time.perf_counter()
    filing_heartbeat_every = max(0, int(progress_log_every_filings))

    for idx, (_, r) in enumerate(
        tqdm(chosen.iterrows(), total=total_filings, desc="Building base sentence cache", unit="filing"),
        start=1,
    ):
        if filing_heartbeat_every and (idx == 1 or idx % filing_heartbeat_every == 0 or idx == total_filings):
            elapsed = time.perf_counter() - cache_start
            avg_per_filing = elapsed / float(idx)
            filings_left = max(0, total_filings - idx)
            eta = avg_per_filing * filings_left
            print(
                f"[BASE_CACHE_PROGRESS] {idx}/{total_filings} | "
                f"elapsed={format_duration(elapsed)} | "
                f"avg_per_filing={avg_per_filing:.2f}s | "
                f"filings_left={filings_left} | eta={format_duration(eta)}",
                flush=True,
            )
        ticker = str(r["ticker"])
        year = int(r["year"])
        text, text_path, load_reason = load_cached_filing_text(str(r["cache_path"]), text_cache_dir)
        if text is None or text_path is None:
            dropped_rows.append(
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
            dropped_rows.append({"ticker": ticker, "year": year, "reason": "item1_not_found"})
            continue

        sentences = sentence_split(item1_text, min_chars=int(min_sentence_chars_floor))
        filing_rows.append(
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
                "n_item1_sentences_floor": len(sentences),
            }
        )

        for idx, sentence in enumerate(sentences, start=1):
            row: Dict[str, object] = {
                "ticker": ticker,
                "year": year,
                "sentence_idx": idx,
                "sentence": sentence,
                "char_len": len(sentence),
            }
            any_match = False
            for term_key, pats in term_specs.items():
                hits = int(transition_match_count(sentence, pats))
                row[f"hits__{term_key}"] = hits
                any_match = any_match or (hits > 0)
            if not any_match:
                continue
            row["filter_reason"] = sentence_filter_reason(sentence)
            norm = normalize_for_repeat(sentence)
            row["repeat_norm"] = norm
            row["repeat_tokens"] = " ".join(sorted(set(norm.split())))
            sentence_rows.append(row)

    filings = pd.DataFrame(filing_rows).sort_values(["year", "ticker"]).reset_index(drop=True)
    candidate_sentences = (
        pd.DataFrame(sentence_rows).sort_values(["year", "ticker", "sentence_idx"]).reset_index(drop=True)
    )
    if dropped_rows:
        dropped = pd.DataFrame(dropped_rows).sort_values(["year", "ticker", "reason"]).reset_index(drop=True)
    else:
        dropped = pd.DataFrame(columns=["ticker", "year", "reason", "text_path"])

    filings.to_csv(filings_cache, index=False)
    candidate_sentences.to_csv(sentences_cache, index=False)
    dropped.to_csv(dropped_cache, index=False)
    return filings, candidate_sentences, dropped


def score_sentence_cache(
    sentence_df: pd.DataFrame,
    finbert_model: str,
    batch_size: int,
    max_lengths: Sequence[int],
    cache_dir: Path,
    progress_log_every_batches: int = 50,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    score_cache = cache_dir / "sentence_scores.jsonl"
    if score_cache.exists():
        scored = pd.read_json(score_cache, lines=True)
        if "sentence" not in scored.columns:
            scored = pd.DataFrame(columns=["sentence"])
    else:
        scored = pd.DataFrame(columns=["sentence"])

    uniq = sorted(set(sentence_df["sentence"].astype(str)))
    score_df = pd.DataFrame({"sentence": uniq}).merge(scored, on="sentence", how="left")
    for max_len in max_lengths:
        pol_name = f"polarity_f{max_len}"
        pos_name = f"pos_hits_f{max_len}"
        neg_name = f"neg_hits_f{max_len}"
        if {pol_name, pos_name, neg_name}.issubset(score_df.columns) and score_df[pol_name].notna().all():
            continue
        scorer = SentenceScorer(
            method="finbert",
            finbert_model=finbert_model,
            batch_size=int(batch_size),
            finbert_max_length=int(max_len),
        )
        rows: List[Dict[str, float]] = []
        total_batches = max(1, math.ceil(len(uniq) / max(1, int(batch_size))))
        batch_heartbeat_every = max(0, int(progress_log_every_batches))
        score_start = time.perf_counter()
        for batch_idx, i in enumerate(range(0, len(uniq), int(batch_size)), start=1):
            batch = uniq[i : i + int(batch_size)]
            rows.extend(scorer.score_many(batch))
            if batch_heartbeat_every and (
                batch_idx == 1 or batch_idx % batch_heartbeat_every == 0 or batch_idx == total_batches
            ):
                elapsed = time.perf_counter() - score_start
                avg_per_batch = elapsed / float(batch_idx)
                batches_left = max(0, total_batches - batch_idx)
                eta = avg_per_batch * batches_left
                scored_sentences = min(len(uniq), batch_idx * int(batch_size))
                print(
                    f"[SCORE_CACHE_PROGRESS] max_len={max_len} | "
                    f"batches={batch_idx}/{total_batches} | "
                    f"sentences={scored_sentences}/{len(uniq)} | "
                    f"elapsed={format_duration(elapsed)} | "
                    f"eta={format_duration(eta)}",
                    flush=True,
                )
        score_df[pol_name] = [float(r["polarity"]) for r in rows]
        score_df[pos_name] = [float(r["pos_hits"]) for r in rows]
        score_df[neg_name] = [float(r["neg_hits"]) for r in rows]
        score_df.to_json(score_cache, orient="records", lines=True)

    return score_df


def compute_config_sentiment(
    cfg: T100FinBERTConfig,
    filings_df: pd.DataFrame,
    sentences_df: pd.DataFrame,
) -> pd.DataFrame:
    hit_col = f"hits__{cfg.substantive_key}"
    pol_col = f"polarity_f{cfg.finbert_max_length}"
    pos_col = f"pos_hits_f{cfg.finbert_max_length}"
    neg_col = f"neg_hits_f{cfg.finbert_max_length}"

    cand = sentences_df[
        (pd.to_numeric(sentences_df[hit_col], errors="coerce") >= int(cfg.min_transition_term_hits))
        & (pd.to_numeric(sentences_df["char_len"], errors="coerce") >= int(cfg.min_sentence_chars))
    ].copy()

    raw_counts = (
        cand.groupby(["ticker", "year"], as_index=False)
        .size()
        .rename(columns={"size": "n_transition_sentences_raw"})
    )

    if cfg.business_filter_on:
        filtered = cand[cand["filter_reason"] != "ok"].copy()
        filtered_counts = (
            filtered.pivot_table(
                index=["ticker", "year"],
                columns="filter_reason",
                values="sentence_idx",
                aggfunc="count",
                fill_value=0,
            )
            .reset_index()
        )
        filtered_counts.columns = [
            str(c) if isinstance(c, str) else c for c in filtered_counts.columns
        ]
        risk_cols = [c for c in filtered_counts.columns if c in {"risk_header", "risk_boilerplate"}]
        acct_cols = [c for c in filtered_counts.columns if c == "accounting_mechanics"]
        noise_cols = [c for c in filtered_counts.columns if c not in {"ticker", "year"} | set(risk_cols) | set(acct_cols)]
        filtered_counts["n_transition_filtered_risk"] = filtered_counts[risk_cols].sum(axis=1) if risk_cols else 0
        filtered_counts["n_transition_filtered_accounting"] = filtered_counts[acct_cols].sum(axis=1) if acct_cols else 0
        filtered_counts["n_transition_filtered_noise"] = filtered_counts[noise_cols].sum(axis=1) if noise_cols else 0
        filtered_counts = filtered_counts[
            [
                "ticker",
                "year",
                "n_transition_filtered_risk",
                "n_transition_filtered_accounting",
                "n_transition_filtered_noise",
            ]
        ]
        cand = cand[cand["filter_reason"] == "ok"].copy()
    else:
        filtered_counts = pd.DataFrame(
            columns=[
                "ticker",
                "year",
                "n_transition_filtered_risk",
                "n_transition_filtered_accounting",
                "n_transition_filtered_noise",
            ]
        )

    repeat_counts_rows: List[Dict[str, object]] = []
    if cfg.drop_cross_year_boilerplate:
        keep_mask = pd.Series(False, index=cand.index)
        prior_by_ticker: Dict[str, List[Tuple[str, set[str]]]] = {}
        for idx, row in cand.sort_values(["ticker", "year", "sentence_idx"]).iterrows():
            ticker = str(row["ticker"])
            norm = str(row.get("repeat_norm", "") or "")
            token_set = set(str(row.get("repeat_tokens", "") or "").split())
            prior = prior_by_ticker.setdefault(ticker, [])
            if is_cross_year_repeat(norm, token_set, prior, float(cfg.repeat_jaccard_threshold)):
                repeat_counts_rows.append({"ticker": ticker, "year": int(row["year"]), "n_transition_filtered_repeat": 1})
                continue
            keep_mask.loc[idx] = True
            if norm:
                prior.append((norm, token_set))
        cand = cand[keep_mask].copy()

    repeat_counts = (
        pd.DataFrame(repeat_counts_rows)
        .groupby(["ticker", "year"], as_index=False)["n_transition_filtered_repeat"]
        .sum()
        if repeat_counts_rows
        else pd.DataFrame(columns=["ticker", "year", "n_transition_filtered_repeat"])
    )

    agg_rows: List[Dict[str, object]] = []
    if not cand.empty:
        for (ticker, year), d in cand.groupby(["ticker", "year"]):
            pol = pd.to_numeric(d[pol_col], errors="coerce").dropna().to_numpy(dtype=float)
            pos = pd.to_numeric(d[pos_col], errors="coerce").dropna().to_numpy(dtype=float)
            neg = pd.to_numeric(d[neg_col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(pol) > 0:
                pos_share = float(np.mean(pol > 0.05))
                neg_share = float(np.mean(pol < -0.05))
                stance = pos_share - neg_share
                mean_pol = float(np.mean(pol))
                median_pol = float(np.median(pol))
            else:
                pos_share = np.nan
                neg_share = np.nan
                stance = np.nan
                mean_pol = np.nan
                median_pol = np.nan
            agg_rows.append(
                {
                    "ticker": ticker,
                    "year": int(year),
                    "n_transition_sentences": int(len(d)),
                    "transition_sentiment_mean": mean_pol,
                    "transition_sentiment_median": median_pol,
                    "transition_pos_share": pos_share,
                    "transition_neg_share": neg_share,
                    "transition_stance_index": stance,
                    "transition_pos_hits_total": float(np.sum(pos)) if len(pos) else np.nan,
                    "transition_neg_hits_total": float(np.sum(neg)) if len(neg) else np.nan,
                }
            )
    agg = pd.DataFrame(agg_rows)

    base = filings_df.copy()
    out = base.merge(raw_counts, on=["ticker", "year"], how="left")
    out = out.merge(filtered_counts, on=["ticker", "year"], how="left")
    out = out.merge(repeat_counts, on=["ticker", "year"], how="left")
    out = out.merge(agg, on=["ticker", "year"], how="left")
    for col in [
        "n_transition_sentences_raw",
        "n_transition_filtered_risk",
        "n_transition_filtered_accounting",
        "n_transition_filtered_noise",
        "n_transition_filtered_repeat",
        "n_transition_sentences",
    ]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    out["scorer"] = "finbert"
    return out.sort_values(["year", "ticker"]).reset_index(drop=True)


def compute_coverage_stats(sent_df: pd.DataFrame, universe_n: int) -> Dict[str, float]:
    out: Dict[str, float] = {
        "rows": float(len(sent_df)),
        "unique_tickers": float(sent_df["ticker"].nunique()) if "ticker" in sent_df.columns else 0.0,
    }
    year_counts = sent_df.groupby("year").size() if "year" in sent_df.columns and not sent_df.empty else pd.Series(dtype=float)
    out["min_year_rows"] = float(year_counts.min()) if not year_counts.empty else 0.0
    out["max_year_rows"] = float(year_counts.max()) if not year_counts.empty else 0.0
    out["mean_year_rows"] = float(year_counts.mean()) if not year_counts.empty else 0.0
    out["mean_year_coverage_pct"] = float((year_counts / max(universe_n, 1) * 100.0).mean()) if not year_counts.empty else 0.0
    stance = pd.to_numeric(sent_df.get("transition_stance_index"), errors="coerce")
    out["stance_non_null_pct"] = float(stance.notna().mean() * 100.0) if len(sent_df) else 0.0
    return out


def write_config_outputs(run_dir: Path, sent_df: pd.DataFrame, parser_dropped: pd.DataFrame, cfg: T100FinBERTConfig) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    sent_df.to_csv(run_dir / "sentiment.csv", index=False)
    parser_dropped.to_csv(run_dir / "dropped.csv", index=False)
    marker = {
        "status": "complete",
        "config_id": cfg.config_id,
        "substantive_file": cfg.substantive_file,
        "min_sentence_chars": cfg.min_sentence_chars,
        "business_filter_on": cfg.business_filter_on,
        "drop_cross_year_boilerplate": cfg.drop_cross_year_boilerplate,
        "repeat_jaccard_threshold": cfg.repeat_jaccard_threshold,
        "min_transition_term_hits": cfg.min_transition_term_hits,
        "finbert_max_length": cfg.finbert_max_length,
    }
    (run_dir / "_extract_complete.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings(
        "ignore",
        message="Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated.*",
        category=FutureWarning,
    )
    ap = argparse.ArgumentParser(description="Search cached FinBERT settings on the final thesis universe")
    ap.add_argument("--universe", choices=["r1000", "t100"], default="r1000")
    ap.add_argument("--tickers-file", type=Path, default=None)
    ap.add_argument("--cache-manifest", type=Path, default=Path("data/cache/edgar_html/cache_manifest.csv"))
    ap.add_argument("--text-cache-dir", type=Path, default=Path("data/cache/edgar_text"))
    ap.add_argument("--years", type=str, default=DEFAULTS["years"])
    ap.add_argument("--eval-years", type=str, default=DEFAULTS["eval_years"])
    ap.add_argument("--returns-file", type=Path, default=None)
    ap.add_argument("--min-months-forward", type=int, default=DEFAULTS["min_months_forward"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--finbert-model", type=str, default=DEFAULTS["finbert_model"])
    ap.add_argument(
        "--substantive-files",
        type=str,
        default=DEFAULTS["substantive_files"],
    )
    ap.add_argument("--char-grid", type=str, default=DEFAULTS["char_grid"])
    ap.add_argument("--min-transition-term-hits-grid", type=str, default=DEFAULTS["min_transition_term_hits_grid"])
    ap.add_argument("--repeat-jaccard-grid", type=str, default=DEFAULTS["repeat_jaccard_grid"])
    ap.add_argument("--finbert-max-length-grid", type=str, default=DEFAULTS["finbert_max_length_grid"])
    ap.add_argument("--business-filter-modes", type=str, default=DEFAULTS["business_filter_modes"])
    ap.add_argument("--repeat-modes", type=str, default=DEFAULTS["repeat_modes"])
    ap.add_argument("--missing-score-policy", choices=["drop", "zero", "both"], default=DEFAULTS["missing_score_policy"])
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--max-filings", type=int, default=0)
    ap.add_argument("--progress-log-every-filings", type=int, default=500)
    ap.add_argument("--progress-log-every-batches", type=int, default=50)
    ap.add_argument("--progress-log-every-configs", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    args = ap.parse_args()

    universe_defaults = DEFAULTS[args.universe]
    if args.tickers_file is None:
        args.tickers_file = universe_defaults["tickers_file"]
    if args.returns_file is None:
        args.returns_file = universe_defaults["returns_file"]
    if args.out_dir is None:
        args.out_dir = universe_defaults["out_dir"]

    substantive_files = parse_csv_list(args.substantive_files)
    for sf in substantive_files:
        if not Path(sf).exists():
            raise FileNotFoundError(f"Missing substantive file: {sf}")
    char_grid = parse_int_grid(args.char_grid)
    min_hit_grid = parse_int_grid(args.min_transition_term_hits_grid)
    repeat_grid = parse_float_grid(args.repeat_jaccard_grid)
    finbert_len_grid = parse_int_grid(args.finbert_max_length_grid)
    filter_modes = parse_bool_modes(args.business_filter_modes)
    repeat_modes = parse_bool_modes(args.repeat_modes)

    cfgs = build_grid(
        substantive_files=substantive_files,
        char_grid=char_grid,
        min_transition_term_hits_grid=min_hit_grid,
        business_filter_modes=filter_modes,
        repeat_modes=repeat_modes,
        repeat_jaccard_grid=repeat_grid,
        finbert_max_length_grid=finbert_len_grid,
    )
    if args.max_runs > 0:
        cfgs = cfgs[: int(args.max_runs)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    filings_df, base_sentence_df, parser_dropped = build_base_cache(
        tickers_file=args.tickers_file,
        cache_manifest=args.cache_manifest,
        text_cache_dir=args.text_cache_dir,
        years_raw=args.years,
        term_files=substantive_files,
        min_sentence_chars_floor=min(char_grid),
        cache_dir=args.out_dir / "_cache",
        max_filings=int(args.max_filings),
        progress_log_every_filings=int(args.progress_log_every_filings),
    )
    score_df = score_sentence_cache(
        sentence_df=base_sentence_df,
        finbert_model=args.finbert_model,
        batch_size=int(args.batch_size),
        max_lengths=finbert_len_grid,
        cache_dir=args.out_dir / "_cache",
        progress_log_every_batches=int(args.progress_log_every_batches),
    )
    sentence_df = base_sentence_df.merge(score_df, on="sentence", how="left")

    eval_years = parse_years(args.eval_years)
    fwd = compute_forward_1y(args.returns_file, eval_years, min_months=int(args.min_months_forward))
    if fwd.empty:
        raise RuntimeError("No forward returns available for requested evaluation years.")
    universe_n = len(parse_tickers(args.tickers_file))

    config_rows: List[Dict[str, object]] = []
    model_frames: List[pd.DataFrame] = []
    best_rows: List[Dict[str, object]] = []

    total_cfgs = len(cfgs)
    config_heartbeat_every = max(0, int(args.progress_log_every_configs))
    config_start = time.perf_counter()
    for idx, cfg in enumerate(tqdm(cfgs, desc=f"Evaluating cached {args.universe} FinBERT configs"), start=1):
        if config_heartbeat_every and (idx == 1 or idx % config_heartbeat_every == 0 or idx == total_cfgs):
            elapsed = time.perf_counter() - config_start
            avg_per_cfg = elapsed / float(idx)
            cfgs_left = max(0, total_cfgs - idx)
            eta = avg_per_cfg * cfgs_left
            print(
                f"[CONFIG_PROGRESS] {idx}/{total_cfgs} | {cfg.config_id} | "
                f"elapsed={format_duration(elapsed)} | "
                f"avg_per_config={avg_per_cfg:.1f}s | "
                f"configs_left={cfgs_left} | eta={format_duration(eta)}",
                flush=True,
            )
        run_dir = runs_dir / cfg.config_id
        sent_path = run_dir / "sentiment.csv"
        if sent_path.exists() and not args.force:
            sent_df = pd.read_csv(sent_path)
        else:
            sent_df = compute_config_sentiment(cfg, filings_df, sentence_df)
            write_config_outputs(run_dir, sent_df, parser_dropped, cfg)

        cov = compute_coverage_stats(sent_df, universe_n=universe_n)
        cfg_row: Dict[str, object] = {
            "config_id": cfg.config_id,
            "substantive_file": cfg.substantive_file,
            "min_sentence_chars": cfg.min_sentence_chars,
            "business_filter_on": cfg.business_filter_on,
            "drop_cross_year_boilerplate": cfg.drop_cross_year_boilerplate,
            "repeat_jaccard_threshold": cfg.repeat_jaccard_threshold,
            "min_transition_term_hits": cfg.min_transition_term_hits,
            "finbert_max_length": cfg.finbert_max_length,
            "sentiment_rows": int(cov["rows"]),
            "unique_tickers": int(cov["unique_tickers"]),
            "mean_year_rows": round(float(cov["mean_year_rows"]), 3),
            "mean_year_coverage_pct": round(float(cov["mean_year_coverage_pct"]), 3),
            "stance_non_null_pct": round(float(cov["stance_non_null_pct"]), 3),
            "parser_dropped_rows": int(len(parser_dropped)),
        }
        config_rows.append(cfg_row)

        res = evaluate_sentiment_file(sent_df, fwd, missing_score_policy=args.missing_score_policy)
        if res.empty:
            continue
        res = res.copy()
        res.insert(0, "config_id", cfg.config_id)
        res["sentiment_rows"] = int(cov["rows"])
        res["mean_year_coverage_pct"] = round(float(cov["mean_year_coverage_pct"]), 3)
        res["stance_non_null_pct"] = round(float(cov["stance_non_null_pct"]), 3)
        res["parser_dropped_rows"] = int(len(parser_dropped))
        model_frames.append(res)
        best_rows.append(res.sort_values(["p_value", "adj_r2"], ascending=[True, False]).iloc[0].to_dict())

    run_cfg_df = pd.DataFrame(config_rows).sort_values("config_id")
    model_df = (
        pd.concat(model_frames, ignore_index=True).sort_values(["p_value", "adj_r2"], ascending=[True, False])
        if model_frames
        else pd.DataFrame()
    )
    best_df = (
        pd.DataFrame(best_rows).sort_values(["p_value", "adj_r2"], ascending=[True, False])
        if best_rows
        else pd.DataFrame()
    )

    run_cfg_df.to_csv(args.out_dir / "run_configs.csv", index=False)
    model_df.to_csv(args.out_dir / "model_results.csv", index=False)
    best_df.to_csv(args.out_dir / "best_result_per_config.csv", index=False)
    parser_dropped.to_csv(args.out_dir / "parser_dropped.csv", index=False)
    if not best_df.empty:
        best_df.head(50).to_csv(args.out_dir / "top50_models.csv", index=False)
        best = best_df.iloc[0]
        print(
            "[BEST] "
            f"config_id={best['config_id']} | "
            f"score={best['score_name']} | "
            f"transform={best['transform']} | "
            f"missing={best['missing_score_policy']} | "
            f"p={best['p_value']:.6g} | "
            f"t={best['t_stat']:.3f} | "
            f"beta={best['beta']:.6f} | "
            f"N={int(best['N'])}"
        )
    print(f"[OUT] {args.out_dir / 'run_configs.csv'} ({len(run_cfg_df)} rows)")
    print(f"[OUT] {args.out_dir / 'model_results.csv'} ({len(model_df)} rows)")
    print(f"[OUT] {args.out_dir / 'best_result_per_config.csv'} ({len(best_df)} rows)")


if __name__ == "__main__":
    main()
