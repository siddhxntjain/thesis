#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from thesis_pipeline.canonical import _load_returns_monthly
from thesis_pipeline.chart_style import (
    PALETTE,
    apply_style,
    percent_formatter,
    save_figure,
    style_axes,
    tidy_legend,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / 'data/outputs/portfolios/final'

DEFAULT_UNIVERSE_PANELS = {
    't100': ROOT / 'data/curated/t100/final_within_year_shared/ticker_year_panel.csv',
    'r1000': ROOT / 'data/curated/r1000/final_within_year/ticker_year_panel.csv',
}
DEFAULT_RETURNS_FILE = ROOT / 'data/raw/returns/daily_ret_10y_full_r1000.csv'
DEFAULT_BENCH_FILE = ROOT / 'data/curated/r1000/final_within_year/benchmark_factors_monthly.csv'
CUTS = [0.10, 0.25, 1.0 / 3.0]
SIGNALS = {
    'tls': {'col': 'tls_score', 'label': 'TLS', 'sign': {'t100': 1.0, 'r1000': 1.0}},
    'sentiment': {'col': 'year_sentiment', 'label': 'Sentiment', 'sign': {'t100': -1.0, 'r1000': -1.0}},
    'co2': {'col': 'co2_scaled', 'label': 'CO2', 'sign': {'t100': 1.0, 'r1000': 1.0}},
}


def load_panel(panel_path: Path) -> pd.DataFrame:
    panel = pd.read_csv(panel_path)
    panel['ticker'] = panel['ticker'].astype(str).str.upper().str.strip()
    panel['signal_year'] = pd.to_numeric(panel['signal_year'], errors='coerce').astype(int)
    panel = panel[panel['has_complete_1y'] == True].copy()
    panel = panel.dropna(subset=['ret_1y'])
    return panel


def load_bench(bench_file: Path) -> pd.DataFrame:
    ff = pd.read_csv(bench_file)
    ff['month_end'] = pd.to_datetime(ff['month_end'], errors='coerce')
    for col in ['rf', 'mkt_rf', 'smb', 'hml', 'rmw', 'cma']:
        ff[col] = pd.to_numeric(ff[col], errors='coerce')
    return ff.dropna(subset=['month_end', 'rf', 'mkt_rf', 'smb', 'hml'])


def combo_id(keys: Iterable[str]) -> str:
    return '+'.join(keys)


def all_combos() -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    names = list(SIGNALS.keys())
    for r in range(1, len(names) + 1):
        out.extend(combinations(names, r))
    return out


def standardize(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors='coerce')
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (x - x.mean()) / sd


def build_membership(panel: pd.DataFrame, universe: str, combo: tuple[str, ...], cutoff: float) -> pd.DataFrame:
    req_cols = [SIGNALS[k]['col'] for k in combo]
    rows: list[pd.DataFrame] = []
    for year, g in panel.groupby('signal_year'):
        d = g[['ticker', 'signal_year', 'ret_1y'] + req_cols].copy()
        d = d.dropna(subset=req_cols).copy()
        if d.empty:
            continue
        signed_components = []
        for key in combo:
            col = SIGNALS[key]['col']
            z = standardize(d[col])
            sign = SIGNALS[key]['sign'][universe]
            d[f'z_{key}'] = sign * z
            signed_components.append(f'z_{key}')
        d = d.dropna(subset=signed_components).copy()
        if d.empty:
            continue
        d['combo_score'] = d[signed_components].mean(axis=1)
        d = d.sort_values('combo_score')
        universe_n = len(d)
        bucket_n = max(1, int(np.floor(universe_n * cutoff)))
        if bucket_n * 2 > universe_n:
            continue
        d['side'] = 'MID'
        d.iloc[:bucket_n, d.columns.get_loc('side')] = 'SHORT'
        d.iloc[-bucket_n:, d.columns.get_loc('side')] = 'LONG'
        d['universe_n'] = universe_n
        d['combo_id'] = combo_id(combo)
        d['cutoff'] = cutoff
        rows.append(d)
    if not rows:
        return pd.DataFrame(columns=['ticker', 'signal_year', 'ret_1y', 'combo_score', 'side', 'universe_n', 'combo_id', 'cutoff'])
    return pd.concat(rows, ignore_index=True)


def build_monthly_portfolio(membership: pd.DataFrame, returns_m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    month_rows: list[dict] = []
    annual_rows: list[dict] = []
    for year, g in membership.groupby('signal_year'):
        start = pd.Timestamp(f'{int(year)}-12-31')
        end = pd.Timestamp(f'{int(year) + 1}-12-31')
        hold = returns_m[(returns_m['date'] > start) & (returns_m['date'] <= end)].copy()
        if hold.empty:
            continue
        long_t = set(g.loc[g['side'] == 'LONG', 'ticker'])
        short_t = set(g.loc[g['side'] == 'SHORT', 'ticker'])
        eligible_t = set(g['ticker'])
        if not long_t or not short_t:
            continue
        long_m = hold[hold['ticker'].isin(long_t)].groupby('month_end', as_index=False)['ret'].mean().rename(columns={'ret': 'long_ret'})
        short_m = hold[hold['ticker'].isin(short_t)].groupby('month_end', as_index=False)['ret'].mean().rename(columns={'ret': 'short_ret'})
        basket_m = hold[hold['ticker'].isin(eligible_t)].groupby('month_end', as_index=False)['ret'].mean().rename(columns={'ret': 'basket_ret'})
        m = long_m.merge(short_m, on='month_end', how='inner').merge(basket_m, on='month_end', how='inner')
        if m.empty:
            continue
        m['ls_ret'] = m['long_ret'] - m['short_ret']
        m['signal_year'] = int(year)
        m['hold_end_year'] = int(year) + 1
        m['n_long'] = int(len(long_t))
        m['n_short'] = int(len(short_t))
        m['universe_n'] = int(g['universe_n'].iloc[0])
        month_rows.extend(m.to_dict('records'))
        annual_rows.append({
            'signal_year': int(year),
            'hold_end_year': int(year) + 1,
            'months': int(m['month_end'].nunique()),
            'n_long': int(len(long_t)),
            'n_short': int(len(short_t)),
            'universe_n': int(g['universe_n'].iloc[0]),
            'long_ret_1y': float(np.prod(1.0 + m['long_ret'].to_numpy(dtype=float)) - 1.0),
            'short_ret_1y': float(np.prod(1.0 + m['short_ret'].to_numpy(dtype=float)) - 1.0),
            'ls_ret_1y': float(np.prod(1.0 + m['ls_ret'].to_numpy(dtype=float)) - 1.0),
            'basket_ret_1y': float(np.prod(1.0 + m['basket_ret'].to_numpy(dtype=float)) - 1.0),
        })
    monthly = pd.DataFrame(month_rows)
    annual = pd.DataFrame(annual_rows).sort_values('signal_year').reset_index(drop=True) if annual_rows else pd.DataFrame()
    return monthly, annual


def anchored_path(df: pd.DataFrame, year_col: str, ret_col: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    g = df[[year_col, ret_col]].copy().sort_values(year_col)
    start = int(g[year_col].min())
    anchor = pd.DataFrame([{year_col: start, ret_col: 0.0}])
    g = pd.concat([anchor, g], ignore_index=True)
    g['cum_ret'] = (1.0 + g[ret_col]).cumprod() - 1.0
    return g


def fit_alpha_models(monthly: pd.DataFrame, bench: pd.DataFrame) -> list[dict]:
    if monthly.empty:
        return []
    x = monthly.merge(bench, on='month_end', how='inner').dropna(subset=['ls_ret', 'rf', 'mkt_rf', 'smb', 'hml'])
    if x.empty:
        return []
    x['ls_excess'] = x['ls_ret'] - x['rf']
    specs = [
        ('capm', ['mkt_rf']),
        ('ff3', ['mkt_rf', 'smb', 'hml']),
    ]
    if x[['rmw', 'cma']].notna().all().all():
        specs.append(('ff5', ['mkt_rf', 'smb', 'hml', 'rmw', 'cma']))
    rows = []
    for name, factors in specs:
        y = x['ls_excess']
        X = sm.add_constant(x[factors])
        m = sm.OLS(y, X).fit()
        rows.append({
            'model': name,
            'n_months': int(m.nobs),
            'alpha_monthly': float(m.params['const']),
            'alpha_t': float(m.tvalues['const']),
            'alpha_p': float(m.pvalues['const']),
            'r2': float(m.rsquared),
            'adj_r2': float(m.rsquared_adj),
        })
    return rows


def summarize_combo(universe: str, combo: tuple[str, ...], cutoff: float, membership: pd.DataFrame, monthly: pd.DataFrame, annual: pd.DataFrame, alpha_rows: list[dict]) -> tuple[list[dict], pd.DataFrame]:
    combo_name = combo_id(combo)
    summary_rows: list[dict] = []
    if annual.empty:
        return summary_rows, annual
    ls = annual['ls_ret_1y'].to_numpy(dtype=float)
    if len(ls) >= 2 and np.std(ls, ddof=1) > 0:
        annual_t = float(np.mean(ls) / (np.std(ls, ddof=1) / np.sqrt(len(ls))))
        annual_p = float(stats.ttest_1samp(ls, 0.0, nan_policy='omit').pvalue)
    else:
        annual_t = np.nan
        annual_p = np.nan
    annual = annual.copy()
    annual['cum_ls_ret'] = (1.0 + annual['ls_ret_1y']).cumprod() - 1.0
    annual['cum_basket_ret'] = (1.0 + annual['basket_ret_1y']).cumprod() - 1.0
    common = {
        'universe': universe,
        'combo_id': combo_name,
        'cutoff': cutoff,
        'n_years': int(len(annual)),
        'start_signal_year': int(annual['signal_year'].min()),
        'end_hold_year': int(annual['hold_end_year'].max()),
        'mean_ls_ret_1y': float(annual['ls_ret_1y'].mean()),
        'mean_basket_ret_1y': float(annual['basket_ret_1y'].mean()),
        'annual_t': annual_t,
        'annual_p': annual_p,
        'final_cum_ls_ret': float(annual['cum_ls_ret'].iloc[-1]),
        'final_cum_basket_ret': float(annual['cum_basket_ret'].iloc[-1]),
        'avg_universe_n': float(annual['universe_n'].mean()),
        'avg_n_long': float(annual['n_long'].mean()),
        'avg_n_short': float(annual['n_short'].mean()),
    }
    if alpha_rows:
        for row in alpha_rows:
            out = common.copy()
            out.update(row)
            summary_rows.append(out)
    else:
        out = common.copy()
        out.update({'model': 'none', 'n_months': 0, 'alpha_monthly': np.nan, 'alpha_t': np.nan, 'alpha_p': np.nan, 'r2': np.nan, 'adj_r2': np.nan})
        summary_rows.append(out)
    return summary_rows, annual


def plot_best(summary: pd.DataFrame, annual_detail: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty or annual_detail.empty:
        return
    best_rows = []
    ff3 = summary[summary['model'] == 'ff3'].copy()
    for universe, g in ff3.groupby('universe'):
        g_pos = g[(g['mean_ls_ret_1y'] > 0) & (g['alpha_monthly'] > 0)].copy()
        target = g_pos if not g_pos.empty else g
        g2 = target.sort_values(['alpha_p', 'annual_p', 'final_cum_ls_ret'], ascending=[True, True, False])
        if not g2.empty:
            best_rows.append(g2.iloc[0])
    if not best_rows:
        return
    fig, axes = plt.subplots(1, len(best_rows), figsize=(7 * len(best_rows), 4.5), squeeze=False)
    apply_style()
    for ax, row in zip(axes[0], best_rows):
        d = annual_detail[
            (annual_detail['universe'] == row['universe'])
            & (annual_detail['combo_id'] == row['combo_id'])
            & (np.isclose(annual_detail['cutoff'], row['cutoff']))
        ].copy()
        if d.empty:
            continue
        ls = anchored_path(d.rename(columns={'hold_end_year': 'year'}), 'year', 'ls_ret_1y')
        basket = anchored_path(d.rename(columns={'hold_end_year': 'year'}), 'year', 'basket_ret_1y')
        ax.plot(ls['year'], ls['cum_ret'], marker='o', color=PALETTE['sentiment'], linewidth=2.4, label='Long-short')
        ax.plot(basket['year'], basket['cum_ret'], marker='s', color=PALETTE['universe'], linewidth=2.4, label='Matched basket')
        ax.axhline(0.0, color='#666666', linewidth=1.0)
        style_axes(ax)
        ax.yaxis.set_major_formatter(percent_formatter())
        ax.set_xlabel('Year')
        ax.set_ylabel('Cumulative return')
        tidy_legend(ax, loc='best')
    fig.tight_layout()
    save_figure(fig, out_dir / 'best_combination_portfolios.png')


def plot_heatmaps(summary: pd.DataFrame, out_dir: Path) -> None:
    ff3 = summary[summary['model'] == 'ff3'].copy()
    if ff3.empty:
        return
    for metric, fname, title in [
        ('mean_ls_ret_1y', 'combination_mean_return_heatmap.png', 'Mean annual long-short return'),
        ('alpha_p', 'combination_alpha_p_heatmap.png', 'FF3 alpha p-value'),
    ]:
        piv = ff3.pivot_table(index='combo_id', columns=['universe', 'cutoff'], values=metric, aggfunc='first')
        if piv.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(piv))))
        apply_style()
        data = piv.to_numpy(dtype=float)
        im = ax.imshow(data, cmap='RdYlGn_r' if metric == 'alpha_p' else 'RdYlGn', aspect='auto')
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{u}-{int(round(c*100))}%" for u, c in piv.columns], rotation=45, ha='right')
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isfinite(v):
                    text = f'{v:.2%}' if metric == 'mean_ls_ret_1y' else f'{v:.3f}'
                    ax.text(j, i, text, ha='center', va='center', fontsize=8, color='black')
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        save_figure(fig, out_dir / fname)


def main() -> None:
    ap = argparse.ArgumentParser(description='Build long-short portfolios across all final signal combinations.')
    ap.add_argument('--t100-panel', type=Path, default=DEFAULT_UNIVERSE_PANELS['t100'])
    ap.add_argument('--r1000-panel', type=Path, default=DEFAULT_UNIVERSE_PANELS['r1000'])
    ap.add_argument('--returns-file', type=Path, default=DEFAULT_RETURNS_FILE)
    ap.add_argument('--bench-file', type=Path, default=DEFAULT_BENCH_FILE)
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    returns_m = _load_returns_monthly(args.returns_file)
    bench = load_bench(args.bench_file)

    membership_rows = []
    monthly_rows = []
    annual_rows = []
    summary_rows = []

    universe_panels = {'t100': args.t100_panel, 'r1000': args.r1000_panel}
    for universe in ['t100', 'r1000']:
        panel = load_panel(universe_panels[universe])
        returns_sub = returns_m[returns_m['ticker'].isin(set(panel['ticker']))].copy()
        for combo in all_combos():
            for cutoff in CUTS:
                membership = build_membership(panel, universe, combo, cutoff)
                if membership.empty:
                    continue
                membership['universe'] = universe
                membership_rows.append(membership)
                monthly, annual = build_monthly_portfolio(membership, returns_sub)
                if not monthly.empty:
                    monthly['universe'] = universe
                    monthly['combo_id'] = combo_id(combo)
                    monthly['cutoff'] = cutoff
                    monthly_rows.append(monthly)
                if not annual.empty:
                    annual['universe'] = universe
                    annual['combo_id'] = combo_id(combo)
                    annual['cutoff'] = cutoff
                    annual_rows.append(annual)
                alpha_rows = fit_alpha_models(monthly, bench)
                rows, annual_enriched = summarize_combo(universe, combo, cutoff, membership, monthly, annual, alpha_rows)
                summary_rows.extend(rows)
                if not annual_enriched.empty:
                    annual_enriched['universe'] = universe
                    annual_enriched['combo_id'] = combo_id(combo)
                    annual_enriched['cutoff'] = cutoff
                    annual_rows[-1] = annual_enriched

    membership_df = pd.concat(membership_rows, ignore_index=True) if membership_rows else pd.DataFrame()
    monthly_df = pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()
    annual_df = pd.concat(annual_rows, ignore_index=True) if annual_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)

    membership_df.to_csv(out_dir / 'long_short_membership.csv', index=False)
    monthly_df.to_csv(out_dir / 'long_short_monthly.csv', index=False)
    annual_df.to_csv(out_dir / 'long_short_annual.csv', index=False)
    summary_df.to_csv(out_dir / 'long_short_summary.csv', index=False)

    if not summary_df.empty:
        ff3 = summary_df[summary_df['model'] == 'ff3'].copy()
        best = ff3.sort_values(['universe', 'alpha_p', 'annual_p', 'final_cum_ls_ret'])
        best.groupby('universe', as_index=False).first().to_csv(out_dir / 'best_by_universe.csv', index=False)
        best_pos = ff3[(ff3['mean_ls_ret_1y'] > 0) & (ff3['alpha_monthly'] > 0)].copy()
        if not best_pos.empty:
            best_pos = best_pos.sort_values(['universe', 'alpha_p', 'annual_p', 'final_cum_ls_ret'])
            best_pos.groupby('universe', as_index=False).first().to_csv(out_dir / 'best_positive_by_universe.csv', index=False)
    plot_best(summary_df, annual_df, out_dir)
    plot_heatmaps(summary_df, out_dir)

    lines = [
        '# Final Long-Short Combination Portfolios',
        '',
        'This run builds annual-rebalanced long-short portfolios on the final optimized 10-year panels for both universes.',
        '',
        '- Universes: `t100`, `r1000`',
        '- Signals: `tls`, `sentiment`, `co2`',
        '- Combinations: all non-empty subsets of those three signals',
        '- Cutoffs: `10%`, `25%`, `33%`',
        '- Portfolio score: average of signed within-year z-scores, where the sign follows the final cross-sectional return relation',
        '- Benchmarks: matched-sample equal-weight basket, plus CAPM / FF3 / FF5 alpha where available',
        '',
    ]
    if not summary_df.empty:
        ff3 = summary_df[summary_df['model'] == 'ff3'].copy()
        best_overall = ff3.sort_values(['universe', 'alpha_p', 'annual_p', 'final_cum_ls_ret'])
        best_positive = ff3[(ff3['mean_ls_ret_1y'] > 0) & (ff3['alpha_monthly'] > 0)].copy()
        for universe in sorted(ff3['universe'].unique()):
            g_all = best_overall[best_overall['universe'] == universe]
            if not g_all.empty:
                row = g_all.iloc[0]
                lines.extend([
                    f'## {universe.upper()} strongest overall spread',
                    f"- Combo: `{row['combo_id']}` at `{int(round(row['cutoff'] * 100))}%` tails",
                    f"- Mean annual long-short return: `{row['mean_ls_ret_1y']:.2%}`",
                    f"- Annual spread p-value: `{row['annual_p']:.3f}`",
                    f"- FF3 alpha monthly: `{row['alpha_monthly']:.4f}` (`p={row['alpha_p']:.3f}`)",
                    f"- Final cumulative long-short return: `{row['final_cum_ls_ret']:.2%}` vs matched basket `{row['final_cum_basket_ret']:.2%}`",
                    '',
                ])
            g_pos = best_positive[best_positive['universe'] == universe]
            if g_pos.empty:
                continue
            row = g_pos.sort_values(['alpha_p', 'annual_p', 'final_cum_ls_ret'], ascending=[True, True, False]).iloc[0]
            lines.extend([
                f'## {universe.upper()} best positive strategy',
                f"- Combo: `{row['combo_id']}` at `{int(round(row['cutoff'] * 100))}%` tails",
                f"- Mean annual long-short return: `{row['mean_ls_ret_1y']:.2%}`",
                f"- Annual spread p-value: `{row['annual_p']:.3f}`",
                f"- FF3 alpha monthly: `{row['alpha_monthly']:.4f}` (`p={row['alpha_p']:.3f}`)",
                f"- Final cumulative long-short return: `{row['final_cum_ls_ret']:.2%}` vs matched basket `{row['final_cum_basket_ret']:.2%}`",
                '',
            ])
    (out_dir / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('[OUT]', out_dir / 'long_short_summary.csv')
    print('[OUT]', out_dir / 'summary.md')
    print('[OUT]', out_dir / 'best_combination_portfolios.png')


if __name__ == '__main__':
    main()
