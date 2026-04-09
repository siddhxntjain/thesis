# Thesis Repo Guide

This repository contains the cleaned active pipeline for the thesis codebase. The supported surface is intentionally small:

- `scripts/` contains the runnable entrypoints
- `src/thesis_pipeline/` contains shared package code
- `assets/` contains tracked non-sensitive static assets, including the term lists used by TLS and sentiment
- `data/` contains local raw inputs, caches, processed outputs, curated panels, and frozen thesis outputs
- `archive/` contains superseded material and is not part of the active pipeline
- `Thesis Draft/` is read-only thesis source material and is not an output destination

## Supported Pipeline

Use `scripts/12_run_full_thesis_pipeline.py` as the top-level orchestrator. The supported stage order is:

1. prepare EDGAR cache
2. build TLS feature caches
3. search TLS lock
4. select shared TLS lock
5. search sentiment lock
6. validate the Russell 1000 sentiment lock
7. build canonical curated panels
8. run characteristic analysis
9. run factor lab
10. run combination portfolios
11. export thesis assets

## Locked Defaults

The source of truth for the locked thesis defaults is `src/thesis_pipeline/__init__.py`.

Key defaults:
- years: `2015-2024`
- shared TLS lock: `(3.25, 1.0, 10)`
- Russell 1000 sentiment lock:
  - `finbert_core_c30_h1_filter_off_repeat_on_j088_f256`
  - `transition_sentiment_median`
  - `winsor_5_95`
  - `drop`
- factor cutoffs: `10%`, `25%`, `33%`

## File Ownership

- Do not add new active scripts at repo root.
- Add reusable code under `src/thesis_pipeline/`.
- Add new runnable stages under `scripts/`.
- Do not write into `Thesis Draft/` from pipeline code.
- Do not depend on files in `archive/`.
- Do not put tracked inputs under `data/raw/`; Git ignores that tree.
- Safe tracked authored resources such as term lists belong under `assets/`.

## Publication Standard

The GitHub version of this repo should contain code, docs, and safe static assets only. Raw data, SEC caches, processed outputs, curated panels, thesis drafts, and archived experiments are local-only.
