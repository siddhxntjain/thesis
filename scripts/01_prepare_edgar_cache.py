#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from thesis_pipeline.edgar_io import html_to_text

DEFAULTS = {
    "cache_dir": Path("data/cache/edgar_html"),
    "text_cache_dir": Path("data/cache/edgar_text"),
    "manifest_path": Path("data/cache/edgar_html/cache_manifest.csv"),
}


def normalize_manifest_paths(manifest_path: Path, cache_dir: Path) -> int:
    if not manifest_path.exists():
        return 0
    df = pd.read_csv(manifest_path)
    if "cache_path" not in df.columns:
        return 0
    changed = 0
    new_paths = []
    for value in df["cache_path"]:
        if pd.isna(value):
            new_paths.append(value)
            continue
        raw = str(value).replace("\\", "/")
        out = raw
        if raw.startswith("cached_edgar/"):
            out = str(cache_dir / raw.split("cached_edgar/", 1)[1])
        elif "/cached_edgar/" in raw:
            out = str(cache_dir / raw.split("/cached_edgar/", 1)[1])
        if out != raw:
            changed += 1
        new_paths.append(out)
    df["cache_path"] = new_paths
    if changed:
        df.to_csv(manifest_path, index=False)
    return changed


def parse_missing_html(cache_dir: Path, text_cache_dir: Path, force: bool) -> tuple[int, int]:
    html_files = list(cache_dir.rglob("*.htm")) + list(cache_dir.rglob("*.html"))
    written = 0
    skipped = 0
    for html_path in html_files:
        rel = html_path.relative_to(cache_dir)
        out_path = text_cache_dir / rel.with_suffix(".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not force:
            skipped += 1
            continue
        if html_path.stat().st_size == 0:
            skipped += 1
            continue
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        out_path.write_text(html_to_text(html), encoding="utf-8")
        written += 1
    return written, skipped


def repair_zero_byte_text(text_cache_dir: Path) -> int:
    repaired = 0
    for txt_path in text_cache_dir.rglob("*.txt"):
        if txt_path.exists() and txt_path.stat().st_size == 0:
            txt_path.unlink()
            repaired += 1
    return repaired


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize cached EDGAR paths and build missing text extracts")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULTS["cache_dir"])
    ap.add_argument("--text-cache-dir", type=Path, default=DEFAULTS["text_cache_dir"])
    ap.add_argument("--manifest-path", type=Path, default=DEFAULTS["manifest_path"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    args.text_cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_changed = normalize_manifest_paths(args.manifest_path, args.cache_dir)
    repaired_zero_byte = repair_zero_byte_text(args.text_cache_dir)
    written, skipped = parse_missing_html(args.cache_dir, args.text_cache_dir, force=args.force)

    summary = {
        "manifest_paths_normalized": manifest_changed,
        "zero_byte_text_removed": repaired_zero_byte,
        "text_files_written": written,
        "text_files_skipped": skipped,
    }
    summary_path = args.text_cache_dir / "prepare_edgar_cache_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OUT] {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
