#!/usr/bin/env python3
"""
edgar_visualization.py

Comprehensive visualization suite for TLS (Transition Language Score) analysis.
Creates publication-quality plots analyzing scoring methodologies, sectors, industries, and market cap.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Create output directory
OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load and prepare the cleaned EDGAR data."""
    print("Loading cleaned_edgar.csv...")
    try:
        df = pd.read_csv('cleaned_edgar.csv')
        print(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")

        # Parse filing_date and filter for 2023-2025 only
        df['filing_date'] = pd.to_datetime(df['filing_date'])
        df['year'] = df['filing_date'].dt.year

        initial_count = len(df)
        df = df[df['year'].isin([2023, 2024, 2025])].copy()
        filtered_count = len(df)

        print(f"  ✓ Filtered to 2023-2025 filings only: {filtered_count} rows ({initial_count - filtered_count} removed)")

        # Create market cap buckets
        df = create_market_cap_buckets(df)

        return df
    except FileNotFoundError:
        print("ERROR: cleaned_edgar.csv not found. Run edgar_cleaning.py first.")
        sys.exit(1)

def create_market_cap_buckets(df):
    """Create 4 market cap buckets: Small, Medium, Large, Mega."""
    # Filter out rows without market cap data
    df_with_mc = df[df['market_cap'].notna()].copy()

    # Calculate quartiles for buckets
    quartiles = df_with_mc['market_cap'].quantile([0.25, 0.5, 0.75])

    def bucket(mc):
        if pd.isna(mc):
            return 'Unknown'
        elif mc < quartiles[0.25]:
            return 'Small'
        elif mc < quartiles[0.5]:
            return 'Medium'
        elif mc < quartiles[0.75]:
            return 'Large'
        else:
            return 'Mega'

    df['market_cap_bucket'] = df['market_cap'].apply(bucket)

    print(f"\nMarket cap buckets created:")
    print(f"  Small:  < ${quartiles[0.25]:,.0f}M")
    print(f"  Medium: ${quartiles[0.25]:,.0f}M - ${quartiles[0.5]:,.0f}M")
    print(f"  Large:  ${quartiles[0.5]:,.0f}M - ${quartiles[0.75]:,.0f}M")
    print(f"  Mega:   > ${quartiles[0.75]:,.0f}M")

    return df

def plot_methodology_distributions(df):
    """Plot distributions for all 6 TLS methodologies."""
    print("\n[1] Plotting TLS methodology distributions...")

    methodologies = ['base', 'section', 'prox15', 'prox20', 'full15', 'full20']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, method in enumerate(methodologies):
        col = f'tls_{method}'
        data = df[col].dropna()

        # Histogram with KDE
        axes[idx].hist(data, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)

        # Add KDE
        try:
            from scipy import stats
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            axes[idx].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        except:
            pass

        # Add vertical line at mean
        mean_val = data.mean()
        axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')

        axes[idx].set_title(f'TLS Distribution: {method.upper()}', fontweight='bold')
        axes[idx].set_xlabel('TLS Score')
        axes[idx].set_ylabel('Density')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_tls_distributions.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / '01_tls_distributions.png'}")
    plt.close()

def plot_substantive_vs_boilerplate(df):
    """Plot substantive vs boilerplate scores across methodologies."""
    print("\n[2] Plotting substantive vs boilerplate comparison...")

    methodologies = ['base', 'section', 'prox15', 'prox20', 'full15', 'full20']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, method in enumerate(methodologies):
        sub_col = f'substantive_{method}'
        boil_col = f'boilerplate_{method}'

        # Scatter plot
        axes[idx].scatter(df[boil_col], df[sub_col], alpha=0.3, s=20, color='steelblue')

        # Add diagonal line (where substantive = boilerplate)
        max_val = max(df[sub_col].max(), df[boil_col].max())
        axes[idx].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Equal (TLS=0)')

        axes[idx].set_title(f'{method.upper()}: Substantive vs Boilerplate', fontweight='bold')
        axes[idx].set_xlabel('Boilerplate Score')
        axes[idx].set_ylabel('Substantive Score')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_substantive_vs_boilerplate.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / '02_substantive_vs_boilerplate.png'}")
    plt.close()

def plot_sector_analysis(df):
    """Plot average TLS scores by sector."""
    print("\n[3] Plotting sector analysis...")

    # Get unique sectors dynamically
    sectors = df['sector'].dropna().unique()
    print(f"  Analyzing {len(sectors)} unique sectors")

    # Calculate average TLS_base by sector
    sector_stats = df.groupby('sector').agg({
        'tls_base': ['mean', 'median', 'std', 'count']
    }).round(3)
    sector_stats.columns = ['mean', 'median', 'std', 'count']
    sector_stats = sector_stats.sort_values('mean', ascending=False)

    # Bar plot with error bars
    fig, ax = plt.subplots(figsize=(14, 8))

    x_pos = np.arange(len(sector_stats))
    bars = ax.bar(x_pos, sector_stats['mean'], yerr=sector_stats['std'],
                   alpha=0.8, capsize=5, color='steelblue', edgecolor='black')

    # Color bars by positive/negative
    for i, bar in enumerate(bars):
        if sector_stats['mean'].iloc[i] < 0:
            bar.set_color('coral')

    ax.set_xlabel('Sector', fontweight='bold')
    ax.set_ylabel('Average TLS Score (Base)', fontweight='bold')
    ax.set_title('Average TLS Score by Sector (with Std Dev)', fontweight='bold', fontsize=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sector_stats.index, rotation=45, ha='right')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')

    # Add sample size annotations
    for i, (idx, row) in enumerate(sector_stats.iterrows()):
        ax.text(i, row['mean'] + row['std'] + 0.1, f"n={int(row['count'])}",
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_sector_avg_tls.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / '03_sector_avg_tls.png'}")
    plt.close()

def plot_sector_boxplots(df):
    """Box plots of TLS distributions by sector."""
    print("\n[4] Plotting sector box plots...")

    sectors = df['sector'].dropna().unique()
    df_filtered = df[df['sector'].notna()].copy()

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create box plot
    sector_order = df_filtered.groupby('sector')['tls_base'].median().sort_values(ascending=False).index

    bp = ax.boxplot([df_filtered[df_filtered['sector'] == s]['tls_base'].dropna() for s in sector_order],
                     labels=sector_order, patch_artist=True, showfliers=False)

    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax.set_xlabel('Sector', fontweight='bold')
    ax.set_ylabel('TLS Score (Base)', fontweight='bold')
    ax.set_title('TLS Score Distribution by Sector', fontweight='bold', fontsize=16)
    ax.set_xticklabels(sector_order, rotation=45, ha='right')
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_sector_boxplots.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / '04_sector_boxplots.png'}")
    plt.close()

def plot_industry_analysis(df, top_n=20):
    """Plot average TLS scores for top N industries."""
    print(f"\n[5] Plotting industry analysis (top {top_n})...")

    # Get unique industries dynamically
    industries = df['industry'].dropna().unique()
    print(f"  Analyzing {len(industries)} unique industries")

    # Calculate average TLS_base by industry (only industries with 5+ companies)
    industry_stats = df.groupby('industry').agg({
        'tls_base': ['mean', 'median', 'std', 'count']
    })
    industry_stats.columns = ['mean', 'median', 'std', 'count']
    industry_stats = industry_stats[industry_stats['count'] >= 5]  # Filter small samples
    industry_stats = industry_stats.sort_values('mean', ascending=False).head(top_n)

    # Horizontal bar plot
    fig, ax = plt.subplots(figsize=(12, 10))

    y_pos = np.arange(len(industry_stats))
    bars = ax.barh(y_pos, industry_stats['mean'], xerr=industry_stats['std'],
                    alpha=0.8, capsize=4, color='steelblue', edgecolor='black')

    # Color bars by positive/negative
    for i, bar in enumerate(bars):
        if industry_stats['mean'].iloc[i] < 0:
            bar.set_color('coral')
        else:
            bar.set_color('lightgreen')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(industry_stats.index)
    ax.set_xlabel('Average TLS Score (Base)', fontweight='bold')
    ax.set_ylabel('Industry', fontweight='bold')
    ax.set_title(f'Top {top_n} Industries by Average TLS Score (min 5 companies)',
                 fontweight='bold', fontsize=16)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    # Add sample size annotations
    for i, (idx, row) in enumerate(industry_stats.iterrows()):
        ax.text(row['mean'] + row['std'] + 0.1, i, f"n={int(row['count'])}",
                ha='left', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_industry_top20.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / '05_industry_top20.png'}")
    plt.close()

def plot_market_cap_analysis(df):
    """Plot TLS scores by market cap buckets."""
    print("\n[6] Plotting market cap analysis...")

    df_filtered = df[df['market_cap_bucket'] != 'Unknown'].copy()

    # Define bucket order
    bucket_order = ['Small', 'Medium', 'Large', 'Mega']

    # Bar plot with error bars
    fig, ax = plt.subplots(figsize=(12, 8))

    bucket_stats = df_filtered.groupby('market_cap_bucket').agg({
        'tls_base': ['mean', 'median', 'std', 'count']
    })
    bucket_stats.columns = ['mean', 'median', 'std', 'count']
    bucket_stats = bucket_stats.reindex(bucket_order)

    x_pos = np.arange(len(bucket_stats))
    bars = ax.bar(x_pos, bucket_stats['mean'], yerr=bucket_stats['std'],
                   alpha=0.8, capsize=8, color=['#FF9999', '#FFCC99', '#99CCFF', '#99FF99'],
                   edgecolor='black')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(bucket_stats.index)
    ax.set_xlabel('Market Cap Bucket', fontweight='bold')
    ax.set_ylabel('Average TLS Score (Base)', fontweight='bold')
    ax.set_title('Average TLS Score by Market Cap Bucket', fontweight='bold', fontsize=16)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')

    # Add sample size annotations
    for i, (idx, row) in enumerate(bucket_stats.iterrows()):
        ax.text(i, row['mean'] + row['std'] + 0.1, f"n={int(row['count'])}",
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_market_cap_avg.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / '06_market_cap_avg.png'}")
    plt.close()

def plot_market_cap_boxplots(df):
    """Box plots of TLS by market cap bucket."""
    print("\n[7] Plotting market cap box plots...")

    df_filtered = df[df['market_cap_bucket'] != 'Unknown'].copy()
    bucket_order = ['Small', 'Medium', 'Large', 'Mega']

    fig, ax = plt.subplots(figsize=(12, 8))

    bp = ax.boxplot([df_filtered[df_filtered['market_cap_bucket'] == b]['tls_base'].dropna()
                      for b in bucket_order],
                     labels=bucket_order, patch_artist=True)

    # Color boxes
    colors = ['#FF9999', '#FFCC99', '#99CCFF', '#99FF99']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Market Cap Bucket', fontweight='bold')
    ax.set_ylabel('TLS Score (Base)', fontweight='bold')
    ax.set_title('TLS Score Distribution by Market Cap Bucket', fontweight='bold', fontsize=16)
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_market_cap_boxplots.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / '07_market_cap_boxplots.png'}")
    plt.close()

def plot_market_cap_scatter(df):
    """Scatter plot of market cap vs TLS score."""
    print("\n[8] Plotting market cap scatter...")

    df_filtered = df[df['market_cap'].notna()].copy()

    fig, ax = plt.subplots(figsize=(14, 8))

    # Log scale for market cap
    scatter = ax.scatter(df_filtered['market_cap'], df_filtered['tls_base'],
                         alpha=0.5, s=30, c=df_filtered['tls_base'],
                         cmap='RdYlGn', edgecolors='black', linewidth=0.5)

    ax.set_xscale('log')
    ax.set_xlabel('Market Cap ($M, log scale)', fontweight='bold')
    ax.set_ylabel('TLS Score (Base)', fontweight='bold')
    ax.set_title('Market Cap vs TLS Score', fontweight='bold', fontsize=16)
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('TLS Score', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_market_cap_scatter.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / '08_market_cap_scatter.png'}")
    plt.close()

def plot_methodology_correlation(df):
    """Correlation heatmap between different TLS methodologies."""
    print("\n[9] Plotting methodology correlation heatmap...")

    methodologies = ['tls_base', 'tls_section', 'tls_prox15', 'tls_prox20', 'tls_full15', 'tls_full20']

    corr_matrix = df[methodologies].corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=0, vmax=1, ax=ax)

    ax.set_title('Correlation Between TLS Methodologies', fontweight='bold', fontsize=16)

    # Rename labels for readability
    labels = ['Base', 'Section', 'Prox 1.5x', 'Prox 2.0x', 'Full 1.5x', 'Full 2.0x']
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '09_methodology_correlation.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / '09_methodology_correlation.png'}")
    plt.close()

def plot_top_bottom_companies(df, n=20):
    """Plot top and bottom N companies by TLS score."""
    print(f"\n[10] Plotting top/bottom {n} companies...")

    # Get most recent filing per company
    df_latest = df.sort_values('filing_date').groupby('ticker').tail(1)

    # Top N
    top_n = df_latest.nlargest(n, 'tls_base')[['ticker', 'tls_base', 'sector']]

    fig, ax = plt.subplots(figsize=(12, 10))

    y_pos = np.arange(len(top_n))
    bars = ax.barh(y_pos, top_n['tls_base'], alpha=0.8, edgecolor='black')

    # Color by sector
    sectors = top_n['sector'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(sectors)))
    sector_color_map = dict(zip(sectors, colors))

    for i, (idx, row) in enumerate(top_n.iterrows()):
        if pd.notna(row['sector']):
            bars[i].set_color(sector_color_map[row['sector']])
        else:
            bars[i].set_color('gray')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_n['ticker'])
    ax.set_xlabel('TLS Score (Base)', fontweight='bold')
    ax.set_ylabel('Company Ticker', fontweight='bold')
    ax.set_title(f'Top {n} Companies by TLS Score', fontweight='bold', fontsize=16)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '10_top_companies.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / '10_top_companies.png'}")
    plt.close()

    # Bottom N
    bottom_n = df_latest.nsmallest(n, 'tls_base')[['ticker', 'tls_base', 'sector']]

    fig, ax = plt.subplots(figsize=(12, 10))

    y_pos = np.arange(len(bottom_n))
    bars = ax.barh(y_pos, bottom_n['tls_base'], alpha=0.8, edgecolor='black', color='coral')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(bottom_n['ticker'])
    ax.set_xlabel('TLS Score (Base)', fontweight='bold')
    ax.set_ylabel('Company Ticker', fontweight='bold')
    ax.set_title(f'Bottom {n} Companies by TLS Score', fontweight='bold', fontsize=16)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '11_bottom_companies.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / '11_bottom_companies.png'}")
    plt.close()

def plot_sector_by_methodology(df):
    """Compare TLS scores across sectors for different methodologies."""
    print("\n[11] Plotting sector comparison across methodologies...")

    methodologies = ['base', 'section', 'prox15', 'prox20', 'full15', 'full20']

    # Calculate mean TLS by sector for each methodology
    sector_means = {}
    for method in methodologies:
        sector_means[method] = df.groupby('sector')[f'tls_{method}'].mean()

    sector_df = pd.DataFrame(sector_means)
    sector_df = sector_df.sort_values('base', ascending=False)

    fig, ax = plt.subplots(figsize=(14, 10))

    x = np.arange(len(sector_df))
    width = 0.13

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, method in enumerate(methodologies):
        offset = (i - 2.5) * width
        ax.bar(x + offset, sector_df[method], width, label=method.upper(),
               alpha=0.8, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(sector_df.index, rotation=45, ha='right')
    ax.set_xlabel('Sector', fontweight='bold')
    ax.set_ylabel('Average TLS Score', fontweight='bold')
    ax.set_title('Sector Scores Across All TLS Methodologies', fontweight='bold', fontsize=16)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '12_sector_methodology_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / '12_sector_methodology_comparison.png'}")
    plt.close()

def plot_substantive_boilerplate_by_sector(df):
    """Plot average substantive vs boilerplate scores by sector."""
    print("\n[12] Plotting substantive vs boilerplate by sector...")

    sector_stats = df.groupby('sector').agg({
        'substantive_base': 'mean',
        'boilerplate_base': 'mean',
        'tls_base': 'mean'
    }).sort_values('tls_base', ascending=False)

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(sector_stats))
    width = 0.35

    bars1 = ax.bar(x - width/2, sector_stats['substantive_base'], width,
                   label='Substantive', alpha=0.8, color='lightgreen', edgecolor='black')
    bars2 = ax.bar(x + width/2, sector_stats['boilerplate_base'], width,
                   label='Boilerplate', alpha=0.8, color='coral', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(sector_stats.index, rotation=45, ha='right')
    ax.set_xlabel('Sector', fontweight='bold')
    ax.set_ylabel('Average Score (per 10k words)', fontweight='bold')
    ax.set_title('Substantive vs Boilerplate Language by Sector', fontweight='bold', fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '13_sector_substantive_vs_boilerplate.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / '13_sector_substantive_vs_boilerplate.png'}")
    plt.close()

def plot_tls_over_time_by_sector(df):
    """Plot average TLS score over time (2023-2025) by sector."""
    print("\n[14] Plotting TLS score over time by sector...")

    # Filter for years 2023-2025 and calculate mean TLS score by sector and year
    df_filtered = df[df['year'].isin([2023, 2024, 2025])]
    sector_yearly_tls = df_filtered.groupby(['sector', 'year'])['tls_base'].mean().unstack()

    # Sort sectors by their 2025 TLS score for better visualization
    sector_yearly_tls = sector_yearly_tls.sort_values(2025, ascending=False)

    fig, ax = plt.subplots(figsize=(16, 10))

    x = np.arange(len(sector_yearly_tls.index))
    width = 0.25

    # Define colors for each year
    colors = {2023: 'coral', 2024: 'lightblue', 2025: 'lightgreen'}

    # Create grouped bars for each year
    ax.bar(x - width, sector_yearly_tls[2023], width, label='2023', color=colors[2023], edgecolor='black')
    ax.bar(x, sector_yearly_tls[2024], width, label='2024', color=colors[2024], edgecolor='black')
    ax.bar(x + width, sector_yearly_tls[2025], width, label='2025', color=colors[2025], edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(sector_yearly_tls.index, rotation=45, ha='right')
    ax.set_xlabel('Sector', fontweight='bold')
    ax.set_ylabel('Average TLS Score (Base)', fontweight='bold')
    ax.set_title('Average TLS Score by Sector (2023-2025)', fontweight='bold', fontsize=16)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(title='Year')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '14_tls_over_time_by_sector.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / '14_tls_over_time_by_sector.png'}")
    plt.close()

def plot_tls_full15_distribution(df):
    """Publication-quality distribution plot for TLS FULL 15 methodology."""
    print("\n[15] Plotting TLS FULL 15 distribution for paper...")

    data = df['tls_full15'].dropna()

    # Create figure with larger size for publication quality and white background
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.set_facecolor('white')

    # Create histogram
    n, bins, patches = ax.hist(data, bins=60, alpha=0.7, color='steelblue',
                                edgecolor='black', density=True, linewidth=1.2)

    # Calculate and display statistics
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()

    # Add vertical lines for mean and median
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2.5,
               label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='darkgreen', linestyle='-.', linewidth=2.5,
               label=f'Median: {median_val:.2f}')

    # Add shaded area for ±1 standard deviation
    ax.axvspan(mean_val - std_val, mean_val + std_val, alpha=0.2, color='yellow',
               label=f'±1 Std Dev ({std_val:.2f})')

    # Styling
    ax.set_xlabel('TLS Score (FULL 15 Methodology)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Density', fontweight='bold', fontsize=14)
    ax.set_title('Distribution of Transition Language Score (TLS FULL 15)\nSection Weight: 1.5x | Proximity Weight: 1.5x',
                 fontweight='bold', fontsize=16, pad=20)
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.4, linestyle='--', color='gray')

    # Set spine colors to black for clean LaTeX-style look
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '15_tls_full15_distribution_paper.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {OUTPUT_DIR / '15_tls_full15_distribution_paper.png'}")
    plt.close()

def generate_top_bottom_firms_by_sector(df):
    """Generate a list of top 10 and bottom 10 firms by sector."""
    output = []
    
    # Get the most recent filing for each company
    df_latest = df.sort_values('filing_date').groupby('ticker').tail(1)

    sectors = df_latest['sector'].dropna().unique()
    
    for sector in sorted(sectors):
        output.append("\n" + "="*80)
        output.append(f"SECTOR: {sector.upper()}")
        output.append("="*80)

        sector_df = df_latest[df_latest['sector'] == sector]

        top_10 = sector_df.nlargest(10, 'tls_base')
        bottom_10 = sector_df.nsmallest(10, 'tls_base')

        output.append("\n--- Top 10 Firms (by most recent TLS Score) ---")
        if top_10.empty:
            output.append("No firms found.")
        else:
            for _, row in top_10.iterrows():
                output.append(f"{row['description']:40s}: {row['tls_base']:8.3f} (Filing Date: {row['filing_date'].strftime('%Y-%m-%d')})")

        output.append("\n--- Bottom 10 Firms (by most recent TLS Score) ---")
        if bottom_10.empty:
            output.append("No firms found.")
        else:
            for _, row in bottom_10.iterrows():
                output.append(f"{row['description']:40s}: {row['tls_base']:8.3f} (Filing Date: {row['filing_date'].strftime('%Y-%m-%d')})")
    
    return "\n".join(output)

def generate_summary_statistics(df):
    """Generate and save summary statistics."""
    print("\n[16] Generating summary statistics...")

    summary = []

    # Overall statistics
    summary.append("="*80)
    summary.append("EDGAR TLS ANALYSIS - SUMMARY STATISTICS")
    summary.append("="*80)
    summary.append(f"\nTotal filings analyzed: {len(df)}")
    summary.append(f"Unique companies: {df['ticker'].nunique()}")
    summary.append(f"Unique sectors: {df['sector'].nunique()}")
    summary.append(f"Unique industries: {df['industry'].nunique()}")

    # TLS score statistics (base methodology)
    summary.append("\n" + "-"*80)
    summary.append("TLS SCORE STATISTICS (BASE METHODOLOGY)")
    summary.append("-"*80)
    summary.append(f"Mean: {df['tls_base'].mean():.3f}")
    summary.append(f"Median: {df['tls_base'].median():.3f}")
    summary.append(f"Std Dev: {df['tls_base'].std():.3f}")
    summary.append(f"Min: {df['tls_base'].min():.3f}")
    summary.append(f"Max: {df['tls_base'].max():.3f}")

    # Top sectors
    summary.append("\n" + "-"*80)
    summary.append("TOP 5 SECTORS BY AVERAGE TLS SCORE")
    summary.append("-"*80)
    top_sectors = df.groupby('sector')['tls_base'].mean().sort_values(ascending=False).head(5)
    for sector, score in top_sectors.items():
        count = len(df[df['sector'] == sector])
        summary.append(f"{sector:30s}: {score:7.3f} (n={count})")

    # Top industries
    summary.append("\n" + "-"*80)
    summary.append("TOP 10 INDUSTRIES BY AVERAGE TLS SCORE (min 5 companies)")
    summary.append("-"*80)
    industry_stats = df.groupby('industry').agg({'tls_base': ['mean', 'count']})
    industry_stats.columns = ['mean', 'count']
    top_industries = industry_stats[industry_stats['count'] >= 5].sort_values('mean', ascending=False).head(10)
    for industry, row in top_industries.iterrows():
        summary.append(f"{industry:40s}: {row['mean']:7.3f} (n={int(row['count'])})")

    # Market cap analysis
    summary.append("\n" + "-"*80)
    summary.append("MARKET CAP BUCKET ANALYSIS")
    summary.append("-"*80)
    bucket_order = ['Small', 'Medium', 'Large', 'Mega']
    bucket_stats = df[df['market_cap_bucket'] != 'Unknown'].groupby('market_cap_bucket').agg({
        'tls_base': ['mean', 'median', 'count']
    })
    bucket_stats.columns = ['mean', 'median', 'count']
    bucket_stats = bucket_stats.reindex(bucket_order)
    for bucket, row in bucket_stats.iterrows():
        summary.append(f"{bucket:10s}: Mean={row['mean']:7.3f}, Median={row['median']:7.3f} (n={int(row['count'])})")

    # Methodology correlation
    summary.append("\n" + "-"*80)
    summary.append("METHODOLOGY CORRELATIONS (with base)")
    summary.append("-"*80)
    for method in ['section', 'prox15', 'prox20', 'full15', 'full20']:
        corr = df['tls_base'].corr(df[f'tls_{method}'])
        summary.append(f"tls_base vs tls_{method:8s}: {corr:.4f}")

    # Generate and append top/bottom firms by sector
    summary.append(generate_top_bottom_firms_by_sector(df))

    summary.append("\n" + "="*80)

    # Save to file
    summary_text = "\n".join(summary)
    (OUTPUT_DIR / 'summary_statistics.txt').write_text(summary_text)
    print(f"  ✓ Saved: {OUTPUT_DIR / 'summary_statistics.txt'}")

    # Also print to console
    print("\n" + summary_text)

def main():
    """Run all visualizations."""
    print("="*80)
    print("EDGAR TLS VISUALIZATION SUITE")
    print("="*80)

    # Load data
    df = load_data()

    # Generate all visualizations
    plot_methodology_distributions(df)
    plot_substantive_vs_boilerplate(df)
    plot_sector_analysis(df)
    plot_sector_boxplots(df)
    plot_industry_analysis(df, top_n=20)
    plot_market_cap_analysis(df)
    plot_market_cap_boxplots(df)
    plot_market_cap_scatter(df)
    plot_methodology_correlation(df)
    plot_top_bottom_companies(df, n=20)
    plot_sector_by_methodology(df)
    plot_substantive_boilerplate_by_sector(df)
    plot_tls_over_time_by_sector(df)
    plot_tls_full15_distribution(df)
    generate_summary_statistics(df)

    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS COMPLETE")
    print("="*80)
    print(f"All outputs saved to: {OUTPUT_DIR.absolute()}/")
    print(f"Total visualizations: 15 images + 1 summary statistics file")

if __name__ == "__main__":
    main()
