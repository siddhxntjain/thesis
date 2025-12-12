#!/usr/bin/env python3
"""
edgar_cleaning.py

Merges TLS scores from edgar.py output with additional company metadata.
Creates cleaned_edgar.csv while preserving raw data in tls_scores_all.csv.
"""

import pandas as pd
import sys

def zscore(x: pd.Series) -> pd.Series:
    """Standardize a vector to mean 0, std 1 (adds small epsilon to avoid divide-by-zero)."""
    return (x - x.mean()) / (x.std(ddof=0) + 1e-12)

def normalize_ticker(ticker):
    """
    Normalize ticker format for matching.
    Converts dots to hyphens (e.g., BRK.A -> BRK-A, BRK.B -> BRK-B)
    """
    if pd.isna(ticker):
        return ticker
    return str(ticker).replace('.', '-')

def main():
    print("Loading TLS scores from tls_scores_all.csv...")
    try:
        tls_df = pd.read_csv('tls_scores_all.csv')
        print(f"  ✓ Loaded {len(tls_df)} rows from TLS scores")
    except FileNotFoundError:
        print("ERROR: tls_scores_all.csv not found")
        sys.exit(1)

    print("\nLoading additional ticker data from ticker_addtl_data.csv...")
    try:
        # Read with encoding to handle BOM if present
        addtl_df = pd.read_csv('ticker_addtl_data.csv', encoding='utf-8-sig')
        print(f"  ✓ Loaded {len(addtl_df)} rows from additional data")
    except FileNotFoundError:
        print("ERROR: ticker_addtl_data.csv not found")
        sys.exit(1)

    # Select and rename columns from additional data
    print("\nPreparing merge columns...")
    addtl_df = addtl_df[['filename', 'value', 'description', 'sector', 'industry']].copy()
    addtl_df.rename(columns={
        'filename': 'ticker_merge',
        'value': 'market_cap'
    }, inplace=True)

    # Normalize tickers for matching
    tls_df['ticker_normalized'] = tls_df['ticker'].apply(normalize_ticker)
    addtl_df['ticker_normalized'] = addtl_df['ticker_merge'].apply(normalize_ticker)

    # Perform left join to preserve all TLS data
    print("\nMerging datasets...")
    merged_df = tls_df.merge(
        addtl_df[['ticker_normalized', 'market_cap', 'description', 'sector', 'industry']],
        on='ticker_normalized',
        how='left'
    )

    # Drop the temporary normalized ticker column
    merged_df.drop(columns=['ticker_normalized'], inplace=True)

    # Drop all z-score columns (will be recalculated after cleaning)
    print("\nDropping z-score columns...")
    z_score_cols = [col for col in merged_df.columns if col.endswith('_z')]
    if z_score_cols:
        merged_df.drop(columns=z_score_cols, inplace=True)
        print(f"  ✓ Dropped {len(z_score_cols)} z-score columns")

    # Drop rows where all base scores are 0 (erroneous data)
    print("\nRemoving erroneous rows (all base scores = 0)...")
    initial_count = len(merged_df)
    merged_df = merged_df[
        ~((merged_df['substantive_base'] == 0) &
          (merged_df['boilerplate_base'] == 0) &
          (merged_df['tls_base'] == 0))
    ]
    dropped_count = initial_count - len(merged_df)
    if dropped_count > 0:
        print(f"  ✓ Dropped {dropped_count} rows with all-zero base scores")
    else:
        print(f"  ✓ No erroneous rows found")

    # Report merge statistics
    matched = merged_df['market_cap'].notna().sum()
    unmatched = merged_df['market_cap'].isna().sum()
    print(f"  ✓ Successfully merged {matched} rows")
    if unmatched > 0:
        print(f"  ⚠ {unmatched} rows without matching ticker data")

    # Get unique tickers that didn't match
    unmatched_tickers = merged_df[merged_df['market_cap'].isna()]['ticker'].unique()
    if len(unmatched_tickers) > 0:
        print(f"\n  Unmatched tickers ({len(unmatched_tickers)} unique):")
        print(f"    {', '.join(sorted(unmatched_tickers)[:20])}")
        if len(unmatched_tickers) > 20:
            print(f"    ... and {len(unmatched_tickers) - 20} more")

    # Compute z-scores for all metrics (after cleaning)
    print("\nComputing z-scores for all metrics...")

    # Base metrics z-scores
    merged_df["substantive_base_z"] = zscore(merged_df["substantive_base"])
    merged_df["boilerplate_base_z"] = zscore(merged_df["boilerplate_base"])
    merged_df["tls_base_z"] = zscore(merged_df["tls_base"])

    # Section-weighted z-scores
    merged_df["substantive_section_z"] = zscore(merged_df["substantive_section"])
    merged_df["boilerplate_section_z"] = zscore(merged_df["boilerplate_section"])
    merged_df["tls_section_z"] = zscore(merged_df["tls_section"])

    # Proximity-weighted z-scores (1.5x and 2x)
    for suffix in ["prox15", "prox20"]:
        merged_df[f"substantive_{suffix}_z"] = zscore(merged_df[f"substantive_{suffix}"])
        merged_df[f"boilerplate_{suffix}_z"] = zscore(merged_df[f"boilerplate_{suffix}"])
        merged_df[f"tls_{suffix}_z"] = zscore(merged_df[f"tls_{suffix}"])

    # Full weighting z-scores (section + proximity at 1.5x and 2x)
    for suffix in ["full15", "full20"]:
        merged_df[f"substantive_{suffix}_z"] = zscore(merged_df[f"substantive_{suffix}"])
        merged_df[f"boilerplate_{suffix}_z"] = zscore(merged_df[f"boilerplate_{suffix}"])
        merged_df[f"tls_{suffix}_z"] = zscore(merged_df[f"tls_{suffix}"])

    print(f"  ✓ Added 18 z-score columns")

    # Save cleaned data
    print("\nSaving cleaned data to cleaned_edgar.csv...")
    merged_df.to_csv('cleaned_edgar.csv', index=False)
    print(f"  ✓ Saved {len(merged_df)} rows to cleaned_edgar.csv")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total rows in output:        {len(merged_df)}")
    print(f"Unique tickers:              {merged_df['ticker'].nunique()}")
    print(f"Rows with market cap data:   {matched} ({matched/len(merged_df)*100:.1f}%)")
    print(f"Unique sectors:              {merged_df['sector'].nunique()}")
    print(f"Unique industries:           {merged_df['industry'].nunique()}")
    print("\nTop 5 sectors by filing count:")
    print(merged_df['sector'].value_counts().head())
    print("\n✓ Data cleaning complete!")

if __name__ == "__main__":
    main()
