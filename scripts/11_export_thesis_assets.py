#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEFAULTS = {
    "characteristics_src": Path("data/outputs/thesis_assets/characteristics"),
    "locked_sentiment_src": Path("data/outputs/thesis_assets/locked_sentiment"),
    "appendix_src": Path("data/outputs/thesis_assets/appendix"),
    "characteristics_dst": Path("data/outputs/thesis_assets/characteristics"),
    "locked_sentiment_dst": Path("data/outputs/thesis_assets/locked_sentiment"),
    "appendix_dst": Path("data/outputs/thesis_assets/appendix"),
}

EXPECTED_FIGURES = {
    "characteristics": [
        "fig_combined_signal_horserace_t100_vs_r1000.png",
        "fig_joint_characteristic_yearly_paths.png",
        "fig_joint_tls_sentiment_scatter_r1000new.png",
        "fig_r1000_custom_factors_ttm_vs_ff5.png",
        "fig_r1000_custom_factors_vs_ff5.png",
        "fig_r1000_factor_exposure_growth.png",
        "fig_r1000_tls_yearly_paths.png",
        "fig_return_alignment_timeline.png",
        "fig_sentiment_yearly_paths.png",
        "fig_tls_distribution_locked_spec.png",
        "fig_tls_robustness_neighborhood.png",
        "fig_tls_substantive_vs_boilerplate.png",
        "fig_tls_vs_sentiment_comparison.png",
    ],
    "locked_sentiment": [
        "fig_locked_sentiment_diagnostics_panel.png",
    ],
}

STATIC_FIGURE_SOURCES = {
    "fig_return_alignment_timeline.png": Path("assets/fig_return_alignment_timeline.png"),
    "fig_tls_vs_sentiment_comparison.png": Path("assets/fig_tls_vs_sentiment_comparison.png"),
}


def mirror_tree(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    if src.resolve() == dst.resolve():
        return sum(1 for path in src.rglob('*') if path.is_file())
    copied = 0
    for path in src.rglob('*'):
        if path.is_dir():
            continue
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def ensure_expected_figures(asset_root: Path) -> int:
    copied = 0
    for bucket, names in EXPECTED_FIGURES.items():
        bucket_dir = asset_root / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            target = bucket_dir / name
            if target.exists():
                continue
            src = STATIC_FIGURE_SOURCES.get(name)
            if src is None or not src.exists():
                raise FileNotFoundError(
                    f"Missing expected figure '{name}' in {bucket_dir} and no static source is configured."
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            copied += 1
    return copied


def main() -> None:
    ap = argparse.ArgumentParser(description="Export frozen thesis assets into one stable output tree")
    ap.add_argument('--characteristics-src', type=Path, default=DEFAULTS['characteristics_src'])
    ap.add_argument('--locked-sentiment-src', type=Path, default=DEFAULTS['locked_sentiment_src'])
    ap.add_argument('--appendix-src', type=Path, default=DEFAULTS['appendix_src'])
    ap.add_argument('--characteristics-dst', type=Path, default=DEFAULTS['characteristics_dst'])
    ap.add_argument('--locked-sentiment-dst', type=Path, default=DEFAULTS['locked_sentiment_dst'])
    ap.add_argument('--appendix-dst', type=Path, default=DEFAULTS['appendix_dst'])
    args = ap.parse_args()

    copied = {
        'characteristics': mirror_tree(args.characteristics_src, args.characteristics_dst),
        'locked_sentiment': mirror_tree(args.locked_sentiment_src, args.locked_sentiment_dst),
        'appendix': mirror_tree(args.appendix_src, args.appendix_dst),
        'expected_figures': ensure_expected_figures(args.characteristics_dst.parent),
    }
    for key, count in copied.items():
        print(f"[OUT] {key}: {count} files")


if __name__ == '__main__':
    main()
