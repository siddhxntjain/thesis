# Thesis Pipeline

This repository contains the active code used to reproduce the final within-year thesis pipeline for text-based energy-transition signals in U.S. equities. The supported surface is intentionally small: runnable entrypoints live in `scripts/`, shared logic lives in `src/thesis_pipeline/`, and the data tree is organized by lifecycle stage. The pipeline is designed so another researcher can identify the correct scripts, inputs, and outputs without needing to reverse-engineer historical experiments. `Thesis Draft/` is read-only thesis source material and is not a write target for the active pipeline. `archive/` is local-only historical material and is not part of the publishable repo surface.

## Top-Level Layout

- `CLAUDE.md`
  - repo-specific working notes and conventions for agentic development
- `README.md`
  - repo map, pipeline order, and file/function descriptions
- `.gitignore`
  - GitHub-safe ignore rules for licensed data, caches, generated outputs, and local-only folders
- `scripts/`
  - supported command-line entrypoints for each pipeline stage
- `src/thesis_pipeline/`
  - reusable package code used by the entrypoints
- `data/`
  - raw inputs, SEC caches, processed search outputs, curated panels, and final outputs
- `assets/`
  - tracked static diagrams and authored term lists needed by the active pipeline
- `Thesis Draft/`
  - LaTeX thesis source; read-only from the pipeline perspective

## Active Python Package

All active reusable logic lives in `src/thesis_pipeline/`.

- `__init__.py`
  - shared constants for the final thesis defaults, including the locked TLS and sentiment specs
- `canonical.py`
  - builds the canonical ticker-year thesis panels by merging TLS, sentiment, returns, factors, sector metadata, assets, and emissions
- `chart_style.py`
  - shared plotting colors, formatters, and save helpers used by figure-producing scripts
- `edgar_io.py`
  - EDGAR HTML/text parsing and text-extraction helpers used by TLS and sentiment pipelines
- `tls.py`
  - TLS feature-cache construction helpers and term-window scoring utilities
- `sentiment.py`
  - transition-sentence extraction, FinBERT/lexicon sentiment helpers, and one-year evaluation utilities
- `regressions.py`
  - shared regression machinery for pooled models, yearly models, sector interactions, and control grids
- `factors.py`
  - factor construction, factor benchmarking, and portfolio/factor summary helpers

## Active Scripts

These are the only supported entrypoints.

1. `scripts/01_prepare_edgar_cache.py`
   - repairs and normalizes the EDGAR cache
   - parses cached filing HTML into plain text
   - repairs zero-byte text cache files and normalizes manifest paths

2. `scripts/02_build_feature_cache.py`
   - builds TLS feature-cache files for one or more years
   - supports both `r1000` and `t100`
   - outputs feature caches used by the TLS lock search

3. `scripts/03_search_tls_lock.py`
   - runs the TLS hyperparameter search
   - supports `--universe r1000` and `--universe t100`
   - writes search results into `data/processed/search/tls/`

4. `scripts/04_select_tls_lock.py`
   - compares the universe-specific TLS searches
   - writes the shared selected TLS lock into `data/processed/search/tls/shared/`

5. `scripts/05_search_sentiment_lock.py`
   - runs the locked sentiment search over the final thesis configuration grid
   - supports `--universe r1000` and `--universe t100`
   - writes search-stage sentiment outputs into `data/processed/search/sentiment/`

6. `scripts/06_validate_sentiment_lock.py`
   - validates the shortlisted Russell 1000 sentiment candidates on holdout and expanding-window splits
   - writes holdout and CV summaries for the final lock decision

7. `scripts/07_build_canonical_data.py`
   - builds a canonical curated panel for a thesis universe
   - used after the lock search when a final ticker-year panel is needed for downstream regressions

8. `scripts/08_run_characteristic_analysis.py`
   - runs the main characteristic layer of the thesis
   - includes TLS-only regressions, sentiment-only regressions, joint horse races, sector interactions, control survival, and leave-year-out robustness

9. `scripts/09_run_factor_lab.py`
   - builds the custom factor returns and evaluates them against CAPM, FF3, and FF5
   - writes factor-model summaries and strategy summaries

10. `scripts/10_run_combo_portfolios.py`
    - builds the final within-year signal-combination portfolios
    - outputs combined long-short portfolio summaries and diagnostics

11. `scripts/11_export_thesis_assets.py`
    - consolidates the frozen thesis-facing figures and tables into `data/outputs/thesis_assets/`
    - validates that the expected thesis figure filenames exist in the stable export tree
    - does not read from or write into `Thesis Draft/`

12. `scripts/12_run_full_thesis_pipeline.py`
    - orchestrates the complete pipeline in canonical stage order
    - supported interface: `--stage {locks,curated,analysis,factors,portfolios,assets,all}`

## Canonical Run Order

The supported end-to-end command is:

```bash
./.venv/bin/python scripts/12_run_full_thesis_pipeline.py --stage all
```

Equivalent stage order:

1. Prepare EDGAR cache
2. Build feature caches
3. Search TLS lock
4. Select shared TLS lock
5. Search sentiment lock
6. Validate Russell 1000 sentiment lock
7. Build canonical curated panels
8. Run characteristic analysis
9. Run factor lab
10. Run combination portfolios
11. Export thesis assets

## Data Layout

### `data/raw/`
Repo-local external inputs.

- `data/raw/assets/asset_data.csv`
  - firm assets panel
- `data/raw/emissions/co2data.csv`
  - firm emissions panel
- `data/raw/factors/`
  - cleaned Fama-French files and related factor inputs
- `data/raw/metadata/`
  - sector and auxiliary firm metadata
- `data/raw/returns/`
  - daily and monthly return files for the supported universes
- `data/raw/tls_exports/`
  - older raw TLS export artifacts retained as source material
- `data/raw/universe/`
  - ticker lists for the Russell 1000 and Transition-100 universes
- `data/raw/esg_scores/`
  - third-party ESG score inputs

### `data/cache/`
SEC EDGAR cache.

- `data/cache/edgar_api/`
  - SEC company-ticker and API-side cache files
- `data/cache/edgar_html/`
  - cached filing HTML plus manifest
- `data/cache/edgar_text/`
  - parsed filing text cache

### `data/processed/`
Search-stage and intermediate outputs that can be regenerated.

- `data/processed/feature_cache/r1000/`
  - Russell 1000 TLS feature caches
- `data/processed/feature_cache/t100/`
  - Transition-100 TLS feature caches
- `data/processed/search/tls/r1000/`
  - Russell 1000 TLS search outputs
- `data/processed/search/tls/t100/`
  - Transition-100 TLS search outputs
- `data/processed/search/tls/shared/`
  - selected shared TLS lock
- `data/processed/search/sentiment/r1000/`
  - Russell 1000 sentiment search outputs and holdout validation
- `data/processed/search/sentiment/t100/`
  - Transition-100 sentiment search outputs
- `data/processed/sweeps/`
  - retained search runbooks and sweep notes
- `data/processed/terms/`
  - ranked or derived term artifacts
- `data/processed/transition_basket/`
  - universe-selection support material

### `data/curated/`
Final curated thesis panels used by the analysis scripts.

- `data/curated/r1000/final_within_year/`
  - canonical Russell 1000 panel
- `data/curated/t100/final_within_year_shared/`
  - canonical Transition-100 panel built under the shared TLS lock

### `data/outputs/`
Frozen final outputs used by the analysis scripts and thesis export layer.

- `data/outputs/final/within_year/`
  - final characteristic-analysis outputs
- `data/outputs/factors/final/`
  - final factor-lab outputs
- `data/outputs/portfolios/final/`
  - final signal-combination portfolio outputs
- `data/outputs/thesis_assets/characteristics/`
  - thesis-facing figures and tables for the characteristic chapters
- `data/outputs/thesis_assets/locked_sentiment/`
  - locked-sentiment diagnostics
- `data/outputs/thesis_assets/appendix/`
  - appendix-ready exported tables and exhibits

## Static Assets

The tracked `assets/` folder keeps only reusable repo-native files that are still part of the supported surface.

- `fig_return_alignment_timeline.{png,mmd,svg}`
  - return-alignment timeline diagram
- `fig_tls_vs_sentiment_comparison.{png,mmd,svg}`
  - TLS-versus-sentiment comparison graphic
- `terms/`
  - authored term lists used by the TLS and sentiment search/build scripts

## Defaults and Where To Change Them

- Global locked thesis defaults are defined in:
  - `src/thesis_pipeline/__init__.py`
- Script-level execution defaults are defined near the top of each file in `scripts/`
- The supported factor cutoffs are defined in:
  - `src/thesis_pipeline/__init__.py`
  - `src/thesis_pipeline/factors.py`
- The end-to-end orchestration order is defined in:
  - `scripts/12_run_full_thesis_pipeline.py`

## Structural Standards

The active surface is intended to satisfy:

- `./.venv/bin/python -m compileall scripts src/thesis_pipeline`
- every active script supports `--help`
- no active script imports from `archive/`
- no active script contains absolute `/Users/...` paths
- no active script writes into `Thesis Draft/`
