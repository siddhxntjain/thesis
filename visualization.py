#!/usr/bin/env python3
"""
Visualization suite for TLS (Transition Language Score) analysis
Generates comprehensive charts exploring the dataset from multiple angles
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_data(csv_path: str) -> pd.DataFrame:
    """Load TLS scores and add derived columns"""
    df = pd.read_csv(csv_path)
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    df['year'] = df['filing_date'].dt.year

    # Add sector classification (you'll need to expand this)
    sector_map = {
        # Clean Energy
        'TSLA': 'Clean Energy', 'NEE': 'Clean Energy', 'BEP': 'Clean Energy',
        'ENPH': 'Clean Energy', 'SEDG': 'Clean Energy', 'RUN': 'Clean Energy',
        'FSLR': 'Clean Energy', 'PLUG': 'Clean Energy', 'BE': 'Clean Energy',
        'CHPT': 'Clean Energy',

        # Oil & Gas
        'XOM': 'Oil & Gas', 'CVX': 'Oil & Gas', 'COP': 'Oil & Gas',
        'SLB': 'Oil & Gas', 'OXY': 'Oil & Gas', 'BP': 'Oil & Gas',
        'EQNR': 'Oil & Gas',

        # Refining
        'MPC': 'Refining', 'PSX': 'Refining', 'VLO': 'Refining',
    }
    df['sector'] = df['ticker'].map(sector_map).fillna('Other')

    return df

# ============================================================================
# DISTRIBUTION ANALYSIS
# ============================================================================

def plot_tls_distributions(df: pd.DataFrame, output_dir: Path):
    """Compare distributions across all 6 methodologies"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    methodologies = ['base', 'section', 'prox15', 'prox20', 'full15', 'full20']

    for i, method in enumerate(methodologies):
        col = f'tls_{method}'
        axes[i].hist(df[col], bins=50, edgecolor='black', alpha=0.7)
        axes[i].axvline(df[col].median(), color='red', linestyle='--',
                       label=f'Median: {df[col].median():.1f}')
        axes[i].axvline(0, color='black', linestyle='-', linewidth=2)
        axes[i].set_xlabel('TLS Score')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'TLS Distribution: {method.upper()}')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'tls_distributions_all_methods.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_substantive_vs_boilerplate(df: pd.DataFrame, output_dir: Path):
    """Scatter: substantive vs boilerplate for each methodology"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    methodologies = ['base', 'section', 'prox15', 'prox20', 'full15', 'full20']

    for i, method in enumerate(methodologies):
        sub_col = f'substantive_{method}'
        bp_col = f'boilerplate_{method}'

        # Color by TLS (substantive - boilerplate)
        tls_col = f'tls_{method}'
        scatter = axes[i].scatter(df[sub_col], df[bp_col],
                                 c=df[tls_col], cmap='RdYlGn',
                                 alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        axes[i].plot([0, df[sub_col].max()], [0, df[sub_col].max()],
                     'k--', alpha=0.5, label='Equal line')
        axes[i].set_xlabel('Substantive Score')
        axes[i].set_ylabel('Boilerplate Score')
        axes[i].set_title(f'{method.upper()}: Substantive vs Boilerplate')
        axes[i].legend()

        plt.colorbar(scatter, ax=axes[i], label='TLS')

    plt.tight_layout()
    plt.savefig(output_dir / 'substantive_vs_boilerplate_all_methods.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# METHODOLOGY COMPARISON
# ============================================================================

def plot_methodology_correlation_matrix(df: pd.DataFrame, output_dir: Path):
    """Correlation heatmap: how similar are the 6 TLS variants?"""
    tls_cols = [f'tls_{m}' for m in ['base', 'section', 'prox15', 'prox20', 'full15', 'full20']]
    corr = df[tls_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm',
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Between TLS Methodologies', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'methodology_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_ranking_consistency(df: pd.DataFrame, output_dir: Path):
    """How consistent are company rankings across methodologies?"""
    methodologies = ['base', 'section', 'prox15', 'prox20', 'full15', 'full20']

    # Get top 20 companies by each methodology
    top_n = 20
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, method in enumerate(methodologies):
        col = f'tls_{method}'
        top_companies = df.nlargest(top_n, col)[['ticker', col]]

        axes[i].barh(range(len(top_companies)), top_companies[col].values)
        axes[i].set_yticks(range(len(top_companies)))
        axes[i].set_yticklabels(top_companies['ticker'].values)
        axes[i].set_xlabel('TLS Score')
        axes[i].set_title(f'Top {top_n} Companies: {method.upper()}')
        axes[i].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'top_companies_all_methods.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_weighting_impact(df: pd.DataFrame, output_dir: Path):
    """How do section and proximity weights change scores?"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Base vs Section (effect of section weighting)
    axes[0, 0].scatter(df['tls_base'], df['tls_section'], alpha=0.5, s=30)
    axes[0, 0].plot([df['tls_base'].min(), df['tls_base'].max()],
                    [df['tls_base'].min(), df['tls_base'].max()], 'r--')
    axes[0, 0].set_xlabel('TLS Base (no weights)')
    axes[0, 0].set_ylabel('TLS Section (1.5x section weight)')
    axes[0, 0].set_title('Impact of Section Weighting')

    # 2. Base vs Prox15 (effect of proximity weighting)
    axes[0, 1].scatter(df['tls_base'], df['tls_prox15'], alpha=0.5, s=30)
    axes[0, 1].plot([df['tls_base'].min(), df['tls_base'].max()],
                    [df['tls_base'].min(), df['tls_base'].max()], 'r--')
    axes[0, 1].set_xlabel('TLS Base')
    axes[0, 1].set_ylabel('TLS Prox15 (1.5x proximity weight)')
    axes[0, 1].set_title('Impact of Proximity Weighting (1.5x)')

    # 3. Prox15 vs Prox20 (comparing proximity multipliers)
    axes[1, 0].scatter(df['tls_prox15'], df['tls_prox20'], alpha=0.5, s=30)
    axes[1, 0].plot([df['tls_prox15'].min(), df['tls_prox15'].max()],
                    [df['tls_prox15'].min(), df['tls_prox15'].max()], 'r--')
    axes[1, 0].set_xlabel('TLS Prox15 (1.5x)')
    axes[1, 0].set_ylabel('TLS Prox20 (2.0x)')
    axes[1, 0].set_title('Proximity Multiplier Comparison')

    # 4. Full15 vs Full20 (combined effects)
    axes[1, 1].scatter(df['tls_full15'], df['tls_full20'], alpha=0.5, s=30)
    axes[1, 1].plot([df['tls_full15'].min(), df['tls_full15'].max()],
                    [df['tls_full15'].min(), df['tls_full15'].max()], 'r--')
    axes[1, 1].set_xlabel('TLS Full15 (section 1.5x + prox 1.5x)')
    axes[1, 1].set_ylabel('TLS Full20 (section 1.5x + prox 2.0x)')
    axes[1, 1].set_title('Full Weighting Comparison')

    plt.tight_layout()
    plt.savefig(output_dir / 'weighting_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# SECTOR ANALYSIS
# ============================================================================

def plot_sector_comparison(df: pd.DataFrame, output_dir: Path):
    """Box plots comparing TLS distributions by sector"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    methodologies = ['base', 'section', 'prox15', 'prox20', 'full15', 'full20']

    for i, method in enumerate(methodologies):
        col = f'tls_{method}'
        df.boxplot(column=col, by='sector', ax=axes[i])
        axes[i].set_xlabel('Sector')
        axes[i].set_ylabel('TLS Score')
        axes[i].set_title(f'TLS by Sector: {method.upper()}')
        plt.sca(axes[i])
        plt.xticks(rotation=45, ha='right')

    plt.suptitle('')  # Remove default title
    plt.tight_layout()
    plt.savefig(output_dir / 'sector_comparison_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_sector_means(df: pd.DataFrame, output_dir: Path):
    """Bar chart: mean TLS by sector for each methodology"""
    methodologies = ['base', 'section', 'prox15', 'prox20', 'full15', 'full20']

    sector_means = df.groupby('sector')[[f'tls_{m}' for m in methodologies]].mean()

    ax = sector_means.plot(kind='bar', figsize=(12, 8), width=0.8)
    ax.set_xlabel('Sector', fontsize=12)
    ax.set_ylabel('Mean TLS Score', fontsize=12)
    ax.set_title('Mean TLS by Sector Across Methodologies', fontsize=14, fontweight='bold')
    ax.legend(title='Methodology', labels=[m.upper() for m in methodologies])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'sector_mean_tls_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# TEMPORAL ANALYSIS
# ============================================================================

def plot_temporal_trends(df: pd.DataFrame, output_dir: Path):
    """Time series: TLS evolution over years"""
    if df['year'].nunique() < 2:
        print("Skipping temporal analysis - need multiple years of data")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    methodologies = ['base', 'section', 'prox15', 'prox20', 'full15', 'full20']

    for i, method in enumerate(methodologies):
        col = f'tls_{method}'
        yearly = df.groupby('year')[col].agg(['mean', 'median', 'std'])

        axes[i].plot(yearly.index, yearly['mean'], marker='o', label='Mean', linewidth=2)
        axes[i].plot(yearly.index, yearly['median'], marker='s', label='Median', linewidth=2)
        axes[i].fill_between(yearly.index,
                            yearly['mean'] - yearly['std'],
                            yearly['mean'] + yearly['std'],
                            alpha=0.3, label='±1 Std Dev')
        axes[i].set_xlabel('Year')
        axes[i].set_ylabel('TLS Score')
        axes[i].set_title(f'Temporal Trend: {method.upper()}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_trends_all_methods.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_sector_evolution(df: pd.DataFrame, output_dir: Path):
    """How do sectors evolve over time?"""
    if df['year'].nunique() < 2:
        return

    # Use tls_full20 as the primary metric
    sector_yearly = df.groupby(['year', 'sector'])['tls_full20'].mean().unstack()

    ax = sector_yearly.plot(kind='line', marker='o', figsize=(12, 8), linewidth=2)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Mean TLS (Full20)', fontsize=12)
    ax.set_title('Sector TLS Evolution Over Time', fontsize=14, fontweight='bold')
    ax.legend(title='Sector')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'sector_evolution_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# COMPONENT ANALYSIS
# ============================================================================

def plot_component_breakdown(df: pd.DataFrame, output_dir: Path):
    """Stacked bar: substantive + boilerplate for top companies"""
    top_n = 30
    top_companies = df.nlargest(top_n, 'tls_full20')[['ticker', 'substantive_full20', 'boilerplate_full20']]

    fig, ax = plt.subplots(figsize=(14, 10))

    x = range(len(top_companies))
    ax.barh(x, top_companies['substantive_full20'], label='Substantive', color='green', alpha=0.8)
    ax.barh(x, -top_companies['boilerplate_full20'], label='Boilerplate', color='red', alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(top_companies['ticker'])
    ax.set_xlabel('Score (per 10k words)', fontsize=12)
    ax.set_title(f'Top {top_n} Companies: Substantive vs Boilerplate Breakdown',
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax.legend()
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'component_breakdown_top_companies.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_greenwashing_detection(df: pd.DataFrame, output_dir: Path):
    """Identify potential greenwashing: high boilerplate, low substantive"""
    plt.figure(figsize=(12, 10))

    scatter = plt.scatter(df['substantive_full20'], df['boilerplate_full20'],
                         c=df['tls_full20'], cmap='RdYlGn',
                         alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

    # Annotate top greenwashers (high boilerplate, negative TLS)
    greenwashers = df[df['tls_full20'] < 0].nlargest(10, 'boilerplate_full20')
    for _, row in greenwashers.iterrows():
        plt.annotate(row['ticker'],
                    (row['substantive_full20'], row['boilerplate_full20']),
                    fontsize=8, alpha=0.7)

    # Annotate top transition leaders (high substantive, low boilerplate)
    leaders = df.nlargest(10, 'tls_full20')
    for _, row in leaders.iterrows():
        plt.annotate(row['ticker'],
                    (row['substantive_full20'], row['boilerplate_full20']),
                    fontsize=8, alpha=0.7, color='darkgreen', fontweight='bold')

    plt.xlabel('Substantive Score', fontsize=12)
    plt.ylabel('Boilerplate Score', fontsize=12)
    plt.title('Greenwashing Detection: Substantive vs Boilerplate', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='TLS (Full20)')
    plt.grid(True, alpha=0.3)

    # Draw quadrant lines at medians
    plt.axvline(df['substantive_full20'].median(), color='gray', linestyle='--', alpha=0.5)
    plt.axhline(df['boilerplate_full20'].median(), color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'greenwashing_detection.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# STATISTICAL SUMMARY
# ============================================================================

def generate_summary_statistics(df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive summary stats table"""
    methodologies = ['base', 'section', 'prox15', 'prox20', 'full15', 'full20']

    summary_data = []
    for method in methodologies:
        sub_col = f'substantive_{method}'
        bp_col = f'boilerplate_{method}'
        tls_col = f'tls_{method}'

        summary_data.append({
            'Methodology': method.upper(),
            'Substantive Mean': df[sub_col].mean(),
            'Substantive Std': df[sub_col].std(),
            'Boilerplate Mean': df[bp_col].mean(),
            'Boilerplate Std': df[bp_col].std(),
            'TLS Mean': df[tls_col].mean(),
            'TLS Std': df[tls_col].std(),
            'TLS Median': df[tls_col].median(),
            'TLS Min': df[tls_col].min(),
            'TLS Max': df[tls_col].max(),
            'Negative TLS %': (df[tls_col] < 0).mean() * 100
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False, float_format='%.2f')

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(csv_path: str, output_dir: str = "visualizations"):
    """Run all visualizations"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Loading data from {csv_path}...")
    df = load_data(csv_path)
    print(f"Loaded {len(df)} filings from {df['ticker'].nunique()} unique companies")

    print("\nGenerating visualizations...")

    print("  - Distribution analysis...")
    plot_tls_distributions(df, output_path)
    plot_substantive_vs_boilerplate(df, output_path)

    print("  - Methodology comparison...")
    plot_methodology_correlation_matrix(df, output_path)
    plot_ranking_consistency(df, output_path)
    plot_weighting_impact(df, output_path)

    print("  - Sector analysis...")
    plot_sector_comparison(df, output_path)
    plot_sector_means(df, output_path)

    print("  - Temporal analysis...")
    plot_temporal_trends(df, output_path)
    plot_sector_evolution(df, output_path)

    print("  - Component analysis...")
    plot_component_breakdown(df, output_path)
    plot_greenwashing_detection(df, output_path)

    print("  - Summary statistics...")
    generate_summary_statistics(df, output_path)

    print(f"\n✅ All visualizations saved to {output_path}/")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate TLS analysis visualizations")
    parser.add_argument("csv_file", type=str, help="Path to TLS scores CSV file")
    parser.add_argument("--output-dir", type=str, default="visualizations",
                       help="Output directory for plots (default: visualizations/)")

    args = parser.parse_args()
    main(args.csv_file, args.output_dir)
