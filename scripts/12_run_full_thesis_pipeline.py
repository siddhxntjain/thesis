#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / '.venv' / 'bin' / 'python')

TLS_SHARED_SPEC = ROOT / 'data/processed/search/tls/shared/selected_shared_spec.csv'
R1000_SENTIMENT_RESULTS = ROOT / 'data/processed/search/sentiment/r1000/model_results.csv'
R1000_SENTIMENT_RUNS = ROOT / 'data/processed/search/sentiment/r1000/runs'
T100_SENTIMENT_RESULTS = ROOT / 'data/processed/search/sentiment/t100/model_results.csv'
T100_SENTIMENT_RUNS = ROOT / 'data/processed/search/sentiment/t100/runs'

R1000_CURATED = ROOT / 'data/curated/r1000/final_within_year'
T100_CURATED = ROOT / 'data/curated/t100/final_within_year_shared'
FINAL_OUTPUT = ROOT / 'data/outputs/final/within_year'
FACTOR_OUTPUT = ROOT / 'data/outputs/factors/final'
PORTFOLIO_OUTPUT = ROOT / 'data/outputs/portfolios/final'

YEARS = '2015-2024'
TLS_LOCK = ('3.25', '1.0', '10')
R1000_SENTIMENT_LOCK = {
    'config_id': 'finbert_core_c30_h1_filter_off_repeat_on_j088_f256',
    'score_name': 'transition_sentiment_median',
    'transform': 'winsor_5_95',
    'missing_policy': 'drop',
}
T100_SENTIMENT_FILE = T100_SENTIMENT_RUNS / R1000_SENTIMENT_LOCK['config_id'] / 'sentiment.csv'


def run(cmd: list[str]) -> None:
    print('[RUN]', ' '.join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_locks() -> None:
    run([PYTHON, 'scripts/03_search_tls_lock.py', '--universe', 'r1000'])
    run([PYTHON, 'scripts/03_search_tls_lock.py', '--universe', 't100'])
    run([PYTHON, 'scripts/04_select_tls_lock.py'])
    run([PYTHON, 'scripts/05_search_sentiment_lock.py', '--universe', 'r1000'])
    run([PYTHON, 'scripts/05_search_sentiment_lock.py', '--universe', 't100'])
    run([PYTHON, 'scripts/06_validate_sentiment_lock.py'])


def run_curated() -> None:
    run([
        PYTHON, 'scripts/07_build_canonical_data.py',
        '--universe-file', 'data/raw/universe/tickers.txt',
        '--years', YEARS,
        '--feature-cache-dir', 'data/processed/feature_cache/r1000',
        '--sentiment-file', str(R1000_SENTIMENT_RUNS / R1000_SENTIMENT_LOCK['config_id'] / 'sentiment.csv'),
        '--sentiment-primary-col', R1000_SENTIMENT_LOCK['score_name'],
        '--returns-file', 'data/raw/returns/daily_ret_10y_full_r1000.csv',
        '--curated-root', str(R1000_CURATED),
        '--tls-sw', TLS_LOCK[0], '--tls-pw', TLS_LOCK[1], '--tls-cw', TLS_LOCK[2],
    ])
    run([
        PYTHON, 'scripts/07_build_canonical_data.py',
        '--universe-file', 'data/raw/universe/transition_100_tickers.txt',
        '--years', YEARS,
        '--feature-cache-dir', 'data/processed/feature_cache/t100',
        '--sentiment-file', str(T100_SENTIMENT_FILE),
        '--sentiment-primary-col', R1000_SENTIMENT_LOCK['score_name'],
        '--returns-file', 'data/raw/returns/daily_ret_10y.csv',
        '--curated-root', str(T100_CURATED),
        '--tls-sw', TLS_LOCK[0], '--tls-pw', TLS_LOCK[1], '--tls-cw', TLS_LOCK[2],
    ])


def run_analysis() -> None:
    run([
        PYTHON, 'scripts/08_run_characteristic_analysis.py',
        '--r1000-curated-root', str(R1000_CURATED),
        '--t100-curated-root', str(T100_CURATED),
        '--out-dir', str(FINAL_OUTPUT),
        '--r1000-sentiment-file', str(R1000_SENTIMENT_RUNS / R1000_SENTIMENT_LOCK['config_id'] / 'sentiment.csv'),
        '--t100-sentiment-file', str(T100_SENTIMENT_FILE),
        '--r1000-sentiment-primary-col', R1000_SENTIMENT_LOCK['score_name'],
        '--t100-sentiment-primary-col', R1000_SENTIMENT_LOCK['score_name'],
        '--r1000-sentiment-score-col', R1000_SENTIMENT_LOCK['score_name'],
        '--t100-sentiment-score-col', R1000_SENTIMENT_LOCK['score_name'],
        '--r1000-sentiment-missing-policy', R1000_SENTIMENT_LOCK['missing_policy'],
        '--t100-sentiment-missing-policy', R1000_SENTIMENT_LOCK['missing_policy'],
        '--r1000-sentiment-transform', R1000_SENTIMENT_LOCK['transform'],
        '--t100-sentiment-transform', R1000_SENTIMENT_LOCK['transform'],
        '--r1000-tls-sw', TLS_LOCK[0], '--r1000-tls-pw', TLS_LOCK[1], '--r1000-tls-cw', TLS_LOCK[2],
    ])


def run_factors() -> None:
    run([PYTHON, 'scripts/09_run_factor_lab.py', '--out-root', str(FACTOR_OUTPUT)])


def run_portfolios() -> None:
    run([PYTHON, 'scripts/10_run_combo_portfolios.py', '--out-dir', str(PORTFOLIO_OUTPUT)])


def run_assets() -> None:
    run([PYTHON, 'scripts/11_export_thesis_assets.py'])


def main() -> None:
    ap = argparse.ArgumentParser(description='Run the cleaned thesis pipeline in canonical stage order')
    ap.add_argument('--stage', choices=['locks', 'curated', 'analysis', 'factors', 'portfolios', 'assets', 'all'], default='all')
    args = ap.parse_args()

    if args.stage in {'locks', 'all'}:
        run_locks()
    if args.stage in {'curated', 'all'}:
        run_curated()
    if args.stage in {'analysis', 'all'}:
        run_analysis()
    if args.stage in {'factors', 'all'}:
        run_factors()
    if args.stage in {'portfolios', 'all'}:
        run_portfolios()
    if args.stage in {'assets', 'all'}:
        run_assets()


if __name__ == '__main__':
    main()
