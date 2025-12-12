"""
Comprehensive Factor Model Regression Analysis
==============================================
Analyzes ESG and TLS impact on 1-year and 5-year returns
Stratified by sector and market cap with extensive visualizations

Author: Thesis Project
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import warnings
import os
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 10

# Create outputs directory
os.makedirs('outputs', exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_regressions(df, group_name, return_period):
    """
    Run all three regression models on a given dataframe.

    Returns dict with results or None if insufficient sample size.
    """
    if len(df) < 10:
        return None

    # Calculate z-scores within this subset
    df_subset = df.copy()
    df_subset['esg_z'] = stats.zscore(df_subset['msci_esg_score'])
    df_subset['tls_z'] = stats.zscore(df_subset['tls_full15'])

    y = df_subset['AnnualReturn']

    # Model 1: ESG-only
    X1 = sm.add_constant(df_subset['esg_z'])
    model1 = sm.OLS(y, X1).fit()

    # Model 2: TLS-only
    X2 = sm.add_constant(df_subset['tls_z'])
    model2 = sm.OLS(y, X2).fit()

    # Model 3: ESG + TLS
    X3 = sm.add_constant(df_subset[['esg_z', 'tls_z']])
    model3 = sm.OLS(y, X3).fit()

    return {
        'Period': return_period,
        'Group': group_name,
        'N': len(df_subset),
        'Mean_Return': df_subset['AnnualReturn'].mean(),
        'Std_Return': df_subset['AnnualReturn'].std(),
        'ESG_R2': model1.rsquared,
        'ESG_AdjR2': model1.rsquared_adj,
        'ESG_coef': model1.params['esg_z'],
        'ESG_tstat': model1.tvalues['esg_z'],
        'ESG_pval': model1.pvalues['esg_z'],
        'TLS_R2': model2.rsquared,
        'TLS_AdjR2': model2.rsquared_adj,
        'TLS_coef': model2.params['tls_z'],
        'TLS_tstat': model2.tvalues['tls_z'],
        'TLS_pval': model2.pvalues['tls_z'],
        'Combined_R2': model3.rsquared,
        'Combined_AdjR2': model3.rsquared_adj,
        'Combined_ESG_coef': model3.params['esg_z'],
        'Combined_ESG_tstat': model3.tvalues['esg_z'],
        'Combined_ESG_pval': model3.pvalues['esg_z'],
        'Combined_TLS_coef': model3.params['tls_z'],
        'Combined_TLS_tstat': model3.tvalues['tls_z'],
        'Combined_TLS_pval': model3.pvalues['tls_z'],
        'model1': model1,
        'model2': model2,
        'model3': model3
    }

def format_pvalue(p):
    """Format p-value with significance stars."""
    if p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.10:
        return '*'
    else:
        return ''

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("=" * 80)
print("COMPREHENSIVE FACTOR MODEL REGRESSION ANALYSIS")
print("=" * 80)
print("\nLoading datasets...")

edgar_df = pd.read_csv('../cleaned_edgar.csv')
esg_df = pd.read_csv('../esg_scores/msci_esg_numerical.csv')
returns_1y_df = pd.read_csv('../returns/1y_ret.csv')
returns_5y_df = pd.read_csv('../returns/5y_ret.csv')

print(f"  EDGAR: {len(edgar_df)} rows")
print(f"  ESG: {len(esg_df)} tickers")
print(f"  1Y Returns: {len(returns_1y_df)} tickers")
print(f"  5Y Returns: {len(returns_5y_df)} tickers")

# ============================================================================
# 2. PREPARE BASE DATA
# ============================================================================

print("\nPreparing data...")

# Convert filing_date to datetime and get most recent filing per ticker
edgar_df['filing_date'] = pd.to_datetime(edgar_df['filing_date'])
edgar_df_sorted = edgar_df.sort_values(['ticker', 'filing_date'], ascending=[True, False])
edgar_recent = edgar_df_sorted.groupby('ticker').first().reset_index()

edgar_clean = edgar_recent[['ticker', 'tls_full15', 'filing_date', 'sector', 'market_cap']].copy()

# Standardize column names
esg_df.columns = ['ticker', 'msci_esg_score']
returns_1y_df.columns = ['ticker', 'AnnualReturn']
returns_5y_df.columns = ['ticker', 'AnnualReturn']

# ============================================================================
# 3. CREATE DATASETS FOR 1Y AND 5Y
# ============================================================================

print("Merging datasets...")

# 1-year returns dataset
merged_1y = edgar_clean.merge(esg_df, on='ticker', how='inner')
merged_1y = merged_1y.merge(returns_1y_df, on='ticker', how='inner')
df_1y = merged_1y.dropna(subset=['msci_esg_score', 'tls_full15', 'AnnualReturn']).copy()

# 5-year returns dataset
merged_5y = edgar_clean.merge(esg_df, on='ticker', how='inner')
merged_5y = merged_5y.merge(returns_5y_df, on='ticker', how='inner')
df_5y = merged_5y.dropna(subset=['msci_esg_score', 'tls_full15', 'AnnualReturn']).copy()

print(f"  1Y sample: {len(df_1y)} companies")
print(f"  5Y sample: {len(df_5y)} companies")

# ============================================================================
# 4. CREATE MARKET CAP BUCKETS (QUARTILE-BASED)
# ============================================================================

print("\nCreating market cap buckets (quartile-based)...")

# Calculate quartiles from 1Y dataset (using all non-null market caps)
quartiles_1y = df_1y['market_cap'].dropna().quantile([0.25, 0.5, 0.75])
quartiles_5y = df_5y['market_cap'].dropna().quantile([0.25, 0.5, 0.75])

print(f"  1Y Market Cap Quartiles:")
print(f"    25th: ${quartiles_1y[0.25]/1e9:.2f}B")
print(f"    50th: ${quartiles_1y[0.5]/1e9:.2f}B")
print(f"    75th: ${quartiles_1y[0.75]/1e9:.2f}B")

print(f"  5Y Market Cap Quartiles:")
print(f"    25th: ${quartiles_5y[0.25]/1e9:.2f}B")
print(f"    50th: ${quartiles_5y[0.5]/1e9:.2f}B")
print(f"    75th: ${quartiles_5y[0.75]/1e9:.2f}B")

def bucket_1y(mc):
    if pd.isna(mc):
        return 'Unknown'
    elif mc < quartiles_1y[0.25]:
        return 'Small'
    elif mc < quartiles_1y[0.5]:
        return 'Medium'
    elif mc < quartiles_1y[0.75]:
        return 'Large'
    else:
        return 'Mega'

def bucket_5y(mc):
    if pd.isna(mc):
        return 'Unknown'
    elif mc < quartiles_5y[0.25]:
        return 'Small'
    elif mc < quartiles_5y[0.5]:
        return 'Medium'
    elif mc < quartiles_5y[0.75]:
        return 'Large'
    else:
        return 'Mega'

df_1y['market_cap_bucket'] = df_1y['market_cap'].apply(bucket_1y)
df_5y['market_cap_bucket'] = df_5y['market_cap'].apply(bucket_5y)

print(f"\n  1Y Market cap distribution:")
print(df_1y['market_cap_bucket'].value_counts().sort_index())
print(f"\n  5Y Market cap distribution:")
print(df_5y['market_cap_bucket'].value_counts().sort_index())

# ============================================================================
# 5. RUN ALL REGRESSIONS
# ============================================================================

print("\nRunning regressions...")

results_list = []

# Overall regressions
print("  Overall (1Y)...")
result = run_regressions(df_1y, "Overall", "1Y")
if result:
    results_list.append(result)

print("  Overall (5Y)...")
result = run_regressions(df_5y, "Overall", "5Y")
if result:
    results_list.append(result)

# Sector regressions
sectors = sorted(df_1y['sector'].dropna().unique())
for sector in sectors:
    print(f"  {sector} (1Y)...")
    sector_df = df_1y[df_1y['sector'] == sector].copy()
    result = run_regressions(sector_df, sector, "1Y")
    if result:
        results_list.append(result)

    print(f"  {sector} (5Y)...")
    sector_df = df_5y[df_5y['sector'] == sector].copy()
    result = run_regressions(sector_df, sector, "5Y")
    if result:
        results_list.append(result)

# Market cap bucket regressions
bucket_order = ['Small', 'Medium', 'Large', 'Mega']

for bucket in bucket_order:
    print(f"  {bucket} (1Y)...")
    bucket_df = df_1y[df_1y['market_cap_bucket'] == bucket].copy()
    result = run_regressions(bucket_df, bucket, "1Y")
    if result:
        results_list.append(result)

    print(f"  {bucket} (5Y)...")
    bucket_df = df_5y[df_5y['market_cap_bucket'] == bucket].copy()
    result = run_regressions(bucket_df, bucket, "5Y")
    if result:
        results_list.append(result)

# ============================================================================
# 6. CREATE SUMMARY DATAFRAME
# ============================================================================

print("\nCreating summary tables...")

summary_df = pd.DataFrame(results_list)

# Drop model objects for CSV export
summary_export = summary_df.drop(columns=['model1', 'model2', 'model3'], errors='ignore')
summary_export.to_csv('outputs/regression_summary_complete.csv', index=False)

# ============================================================================
# 7. GENERATE COMPREHENSIVE VISUALIZATIONS
# ============================================================================

print("\nGenerating visualizations...")

# -----------------------------------------------------------------------------
# PLOT 1: R² Comparison - 1Y vs 5Y (Overall)
# -----------------------------------------------------------------------------
print("  1. Overall R² comparison (1Y vs 5Y)...")

overall_results = summary_df[summary_df['Group'] == 'Overall'].copy()

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(2)
width = 0.25

periods = overall_results['Period'].values
esg_r2 = overall_results['ESG_R2'].values
tls_r2 = overall_results['TLS_R2'].values
combined_r2 = overall_results['Combined_R2'].values

ax.bar(x - width, esg_r2, width, label='ESG-only', alpha=0.8, color='green')
ax.bar(x, tls_r2, width, label='TLS-only', alpha=0.8, color='blue')
ax.bar(x + width, combined_r2, width, label='ESG + TLS', alpha=0.8, color='purple')

ax.set_ylabel('R²', fontweight='bold')
ax.set_title('Model Performance: 1-Year vs 5-Year Returns (Overall Sample)', fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(periods)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('outputs/01_overall_r2_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# PLOT 2: Coefficient Comparison - 1Y vs 5Y (Overall)
# -----------------------------------------------------------------------------
print("  2. Overall coefficient comparison (1Y vs 5Y)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ESG coefficients
axes[0].bar(periods, overall_results['ESG_coef'].values, alpha=0.7, color='green', edgecolor='black')
axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0].set_ylabel('ESG Coefficient', fontweight='bold')
axes[0].set_title('ESG Impact on Returns\n(1Y vs 5Y)', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# TLS coefficients
axes[1].bar(periods, overall_results['TLS_coef'].values, alpha=0.7, color='blue', edgecolor='black')
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1].set_ylabel('TLS Coefficient', fontweight='bold')
axes[1].set_title('TLS Impact on Returns\n(1Y vs 5Y)', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/02_overall_coefficient_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# PLOT 3: R² by Sector (1Y)
# -----------------------------------------------------------------------------
print("  3. R² by sector (1Y)...")

sector_1y = summary_df[(summary_df['Period'] == '1Y') & (summary_df['Group'] != 'Overall') &
                       (~summary_df['Group'].isin(['Small', 'Medium', 'Large', 'Mega']))].copy()

if len(sector_1y) > 0:
    sector_1y = sector_1y.sort_values('Combined_R2', ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(sector_1y) * 0.4)))
    y_pos = np.arange(len(sector_1y))
    width = 0.25

    ax.barh(y_pos - width, sector_1y['ESG_R2'], width, label='ESG-only', alpha=0.8, color='green')
    ax.barh(y_pos, sector_1y['TLS_R2'], width, label='TLS-only', alpha=0.8, color='blue')
    ax.barh(y_pos + width, sector_1y['Combined_R2'], width, label='ESG + TLS', alpha=0.8, color='purple')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sector_1y['Group'])
    ax.set_xlabel('R²', fontweight='bold')
    ax.set_title('Model Performance by Sector (1-Year Returns)', fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('outputs/03_sector_r2_1y.png', dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# PLOT 4: R² by Sector (5Y)
# -----------------------------------------------------------------------------
print("  4. R² by sector (5Y)...")

sector_5y = summary_df[(summary_df['Period'] == '5Y') & (summary_df['Group'] != 'Overall') &
                       (~summary_df['Group'].isin(['Small', 'Medium', 'Large', 'Mega']))].copy()

if len(sector_5y) > 0:
    sector_5y = sector_5y.sort_values('Combined_R2', ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(sector_5y) * 0.4)))
    y_pos = np.arange(len(sector_5y))
    width = 0.25

    ax.barh(y_pos - width, sector_5y['ESG_R2'], width, label='ESG-only', alpha=0.8, color='green')
    ax.barh(y_pos, sector_5y['TLS_R2'], width, label='TLS-only', alpha=0.8, color='blue')
    ax.barh(y_pos + width, sector_5y['Combined_R2'], width, label='ESG + TLS', alpha=0.8, color='purple')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sector_5y['Group'])
    ax.set_xlabel('R²', fontweight='bold')
    ax.set_title('Model Performance by Sector (5-Year Returns)', fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('outputs/04_sector_r2_5y.png', dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# PLOT 5: ESG Coefficients by Sector (1Y vs 5Y Comparison)
# -----------------------------------------------------------------------------
print("  5. ESG coefficients by sector (1Y vs 5Y)...")

# Merge 1Y and 5Y sector results
sector_comparison = sector_1y[['Group', 'ESG_coef', 'ESG_pval']].merge(
    sector_5y[['Group', 'ESG_coef', 'ESG_pval']],
    on='Group', suffixes=('_1Y', '_5Y'), how='outer'
).fillna(0)

if len(sector_comparison) > 0:
    sector_comparison = sector_comparison.sort_values('ESG_coef_5Y', ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(sector_comparison) * 0.4)))
    y_pos = np.arange(len(sector_comparison))
    width = 0.35

    # Color by significance
    colors_1y = ['green' if p < 0.05 else 'lightgray' for p in sector_comparison['ESG_pval_1Y']]
    colors_5y = ['darkgreen' if p < 0.05 else 'lightgray' for p in sector_comparison['ESG_pval_5Y']]

    ax.barh(y_pos - width/2, sector_comparison['ESG_coef_1Y'], width,
            label='1Y Returns', alpha=0.8, color=colors_1y, edgecolor='black', linewidth=0.5)
    ax.barh(y_pos + width/2, sector_comparison['ESG_coef_5Y'], width,
            label='5Y Returns', alpha=0.8, color=colors_5y, edgecolor='black', linewidth=0.5)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sector_comparison['Group'])
    ax.set_xlabel('ESG Coefficient', fontweight='bold')
    ax.set_title('ESG Impact by Sector: 1Y vs 5Y Returns\n(Dark = p<0.05, Light = not significant)',
                 fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('outputs/05_sector_esg_coef_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# PLOT 6: TLS Coefficients by Sector (1Y vs 5Y Comparison)
# -----------------------------------------------------------------------------
print("  6. TLS coefficients by sector (1Y vs 5Y)...")

sector_comparison_tls = sector_1y[['Group', 'TLS_coef', 'TLS_pval']].merge(
    sector_5y[['Group', 'TLS_coef', 'TLS_pval']],
    on='Group', suffixes=('_1Y', '_5Y'), how='outer'
).fillna(0)

if len(sector_comparison_tls) > 0:
    sector_comparison_tls = sector_comparison_tls.sort_values('TLS_coef_5Y', ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(sector_comparison_tls) * 0.4)))
    y_pos = np.arange(len(sector_comparison_tls))
    width = 0.35

    # Color by significance
    colors_1y = ['blue' if p < 0.05 else 'lightgray' for p in sector_comparison_tls['TLS_pval_1Y']]
    colors_5y = ['darkblue' if p < 0.05 else 'lightgray' for p in sector_comparison_tls['TLS_pval_5Y']]

    ax.barh(y_pos - width/2, sector_comparison_tls['TLS_coef_1Y'], width,
            label='1Y Returns', alpha=0.8, color=colors_1y, edgecolor='black', linewidth=0.5)
    ax.barh(y_pos + width/2, sector_comparison_tls['TLS_coef_5Y'], width,
            label='5Y Returns', alpha=0.8, color=colors_5y, edgecolor='black', linewidth=0.5)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sector_comparison_tls['Group'])
    ax.set_xlabel('TLS Coefficient', fontweight='bold')
    ax.set_title('TLS Impact by Sector: 1Y vs 5Y Returns\n(Dark = p<0.05, Light = not significant)',
                 fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('outputs/06_sector_tls_coef_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# PLOT 7: Market Cap R² Comparison (1Y)
# -----------------------------------------------------------------------------
print("  7. Market cap R² (1Y)...")

mktcap_1y = summary_df[(summary_df['Period'] == '1Y') &
                       (summary_df['Group'].isin(['Small', 'Medium', 'Large', 'Mega']))].copy()

if len(mktcap_1y) > 0:
    # Order by market cap size
    mktcap_1y['Order'] = mktcap_1y['Group'].map({
        'Small': 1,
        'Medium': 2,
        'Large': 3,
        'Mega': 4
    })
    mktcap_1y = mktcap_1y.sort_values('Order')

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(mktcap_1y))
    width = 0.25

    ax.bar(x - width, mktcap_1y['ESG_R2'], width, label='ESG-only', alpha=0.8, color='green')
    ax.bar(x, mktcap_1y['TLS_R2'], width, label='TLS-only', alpha=0.8, color='blue')
    ax.bar(x + width, mktcap_1y['Combined_R2'], width, label='ESG + TLS', alpha=0.8, color='purple')

    ax.set_xticks(x)
    ax.set_xticklabels(mktcap_1y['Group'], rotation=20, ha='right')
    ax.set_ylabel('R²', fontweight='bold')
    ax.set_title('Model Performance by Market Cap (1-Year Returns)', fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('outputs/07_mktcap_r2_1y.png', dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# PLOT 8: Market Cap R² Comparison (5Y)
# -----------------------------------------------------------------------------
print("  8. Market cap R² (5Y)...")

mktcap_5y = summary_df[(summary_df['Period'] == '5Y') &
                       (summary_df['Group'].isin(['Small', 'Medium', 'Large', 'Mega']))].copy()

if len(mktcap_5y) > 0:
    # Order by market cap size
    mktcap_5y['Order'] = mktcap_5y['Group'].map({
        'Small': 1,
        'Medium': 2,
        'Large': 3,
        'Mega': 4
    })
    mktcap_5y = mktcap_5y.sort_values('Order')

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(mktcap_5y))
    width = 0.25

    ax.bar(x - width, mktcap_5y['ESG_R2'], width, label='ESG-only', alpha=0.8, color='green')
    ax.bar(x, mktcap_5y['TLS_R2'], width, label='TLS-only', alpha=0.8, color='blue')
    ax.bar(x + width, mktcap_5y['Combined_R2'], width, label='ESG + TLS', alpha=0.8, color='purple')

    ax.set_xticks(x)
    ax.set_xticklabels(mktcap_5y['Group'], rotation=20, ha='right')
    ax.set_ylabel('R²', fontweight='bold')
    ax.set_title('Model Performance by Market Cap (5-Year Returns)', fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('outputs/08_mktcap_r2_5y.png', dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# PLOT 9: Market Cap Coefficient Comparison (ESG)
# -----------------------------------------------------------------------------
print("  9. Market cap ESG coefficients (1Y vs 5Y)...")

if len(mktcap_1y) > 0 and len(mktcap_5y) > 0:
    mktcap_esg = mktcap_1y[['Group', 'ESG_coef', 'ESG_pval']].merge(
        mktcap_5y[['Group', 'ESG_coef', 'ESG_pval']],
        on='Group', suffixes=('_1Y', '_5Y')
    )

    mktcap_esg['Order'] = mktcap_esg['Group'].map({
        'Small': 1,
        'Medium': 2,
        'Large': 3,
        'Mega': 4
    })
    mktcap_esg = mktcap_esg.sort_values('Order')

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(mktcap_esg))
    width = 0.35

    colors_1y = ['green' if p < 0.05 else 'lightgray' for p in mktcap_esg['ESG_pval_1Y']]
    colors_5y = ['darkgreen' if p < 0.05 else 'lightgray' for p in mktcap_esg['ESG_pval_5Y']]

    ax.bar(x - width/2, mktcap_esg['ESG_coef_1Y'], width,
           label='1Y Returns', alpha=0.8, color=colors_1y, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, mktcap_esg['ESG_coef_5Y'], width,
           label='5Y Returns', alpha=0.8, color=colors_5y, edgecolor='black', linewidth=0.5)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(mktcap_esg['Group'], rotation=20, ha='right')
    ax.set_ylabel('ESG Coefficient', fontweight='bold')
    ax.set_title('ESG Impact by Market Cap: 1Y vs 5Y\n(Dark = p<0.05)', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('outputs/09_mktcap_esg_coef_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# PLOT 10: Market Cap Coefficient Comparison (TLS)
# -----------------------------------------------------------------------------
print("  10. Market cap TLS coefficients (1Y vs 5Y)...")

if len(mktcap_1y) > 0 and len(mktcap_5y) > 0:
    mktcap_tls = mktcap_1y[['Group', 'TLS_coef', 'TLS_pval']].merge(
        mktcap_5y[['Group', 'TLS_coef', 'TLS_pval']],
        on='Group', suffixes=('_1Y', '_5Y')
    )

    mktcap_tls['Order'] = mktcap_tls['Group'].map({
        'Small': 1,
        'Medium': 2,
        'Large': 3,
        'Mega': 4
    })
    mktcap_tls = mktcap_tls.sort_values('Order')

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(mktcap_tls))
    width = 0.35

    colors_1y = ['blue' if p < 0.05 else 'lightgray' for p in mktcap_tls['TLS_pval_1Y']]
    colors_5y = ['darkblue' if p < 0.05 else 'lightgray' for p in mktcap_tls['TLS_pval_5Y']]

    ax.bar(x - width/2, mktcap_tls['TLS_coef_1Y'], width,
           label='1Y Returns', alpha=0.8, color=colors_1y, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, mktcap_tls['TLS_coef_5Y'], width,
           label='5Y Returns', alpha=0.8, color=colors_5y, edgecolor='black', linewidth=0.5)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(mktcap_tls['Group'], rotation=20, ha='right')
    ax.set_ylabel('TLS Coefficient', fontweight='bold')
    ax.set_title('TLS Impact by Market Cap: 1Y vs 5Y\n(Dark = p<0.05)', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('outputs/10_mktcap_tls_coef_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# PLOT 11: Heatmap - R² by Sector and Period
# -----------------------------------------------------------------------------
print("  11. R² heatmap by sector...")

sector_all = summary_df[(summary_df['Group'] != 'Overall') &
                        (~summary_df['Group'].isin(['Small', 'Medium', 'Large', 'Mega']))].copy()

if len(sector_all) > 0:
    # Create pivot tables for each model
    pivot_esg = sector_all.pivot_table(values='ESG_R2', index='Group', columns='Period')
    pivot_tls = sector_all.pivot_table(values='TLS_R2', index='Group', columns='Period')
    pivot_combined = sector_all.pivot_table(values='Combined_R2', index='Group', columns='Period')

    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(sector_all['Group'].unique()) * 0.5)))

    sns.heatmap(pivot_esg, annot=True, fmt='.4f', cmap='Greens', ax=axes[0],
                cbar_kws={'label': 'R²'}, vmin=0, vmax=0.15)
    axes[0].set_title('ESG-only Model R²', fontweight='bold')
    axes[0].set_ylabel('Sector', fontweight='bold')

    sns.heatmap(pivot_tls, annot=True, fmt='.4f', cmap='Blues', ax=axes[1],
                cbar_kws={'label': 'R²'}, vmin=0, vmax=0.15)
    axes[1].set_title('TLS-only Model R²', fontweight='bold')
    axes[1].set_ylabel('')

    sns.heatmap(pivot_combined, annot=True, fmt='.4f', cmap='Purples', ax=axes[2],
                cbar_kws={'label': 'R²'}, vmin=0, vmax=0.15)
    axes[2].set_title('Combined Model R²', fontweight='bold')
    axes[2].set_ylabel('')

    plt.tight_layout()
    plt.savefig('outputs/11_sector_r2_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# PLOT 12: Scatterplot - TLS vs Returns (1Y Overall)
# -----------------------------------------------------------------------------
print("  12. Scatterplot TLS vs returns (1Y)...")

df_1y['esg_z'] = stats.zscore(df_1y['msci_esg_score'])
df_1y['tls_z'] = stats.zscore(df_1y['tls_full15'])

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(df_1y['tls_z'], df_1y['AnnualReturn'], alpha=0.4, s=30, edgecolors='k', linewidths=0.3)

# Add regression line
z = np.polyfit(df_1y['tls_z'], df_1y['AnnualReturn'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_1y['tls_z'].min(), df_1y['tls_z'].max(), 100)
ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Fit: y={z[0]:.4f}x+{z[1]:.4f}')

ax.set_xlabel('TLS (z-score)', fontweight='bold')
ax.set_ylabel('1-Year Return', fontweight='bold')
ax.set_title('1-Year Stock Returns vs. Transition Language Score', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/12_scatter_tls_1y.png', dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# PLOT 13: Scatterplot - ESG vs Returns (1Y Overall)
# -----------------------------------------------------------------------------
print("  13. Scatterplot ESG vs returns (1Y)...")

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(df_1y['esg_z'], df_1y['AnnualReturn'], alpha=0.4, s=30, edgecolors='k',
           linewidths=0.3, color='green')

# Add regression line
z = np.polyfit(df_1y['esg_z'], df_1y['AnnualReturn'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_1y['esg_z'].min(), df_1y['esg_z'].max(), 100)
ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Fit: y={z[0]:.4f}x+{z[1]:.4f}')

ax.set_xlabel('MSCI ESG Score (z-score)', fontweight='bold')
ax.set_ylabel('1-Year Return', fontweight='bold')
ax.set_title('1-Year Stock Returns vs. MSCI ESG Score', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/13_scatter_esg_1y.png', dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# PLOT 14: Scatterplot - TLS vs Returns (5Y Overall)
# -----------------------------------------------------------------------------
print("  14. Scatterplot TLS vs returns (5Y)...")

df_5y['esg_z'] = stats.zscore(df_5y['msci_esg_score'])
df_5y['tls_z'] = stats.zscore(df_5y['tls_full15'])

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(df_5y['tls_z'], df_5y['AnnualReturn'], alpha=0.4, s=30, edgecolors='k', linewidths=0.3)

# Add regression line
z = np.polyfit(df_5y['tls_z'], df_5y['AnnualReturn'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_5y['tls_z'].min(), df_5y['tls_z'].max(), 100)
ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Fit: y={z[0]:.4f}x+{z[1]:.4f}')

ax.set_xlabel('TLS (z-score)', fontweight='bold')
ax.set_ylabel('5-Year Return', fontweight='bold')
ax.set_title('5-Year Stock Returns vs. Transition Language Score', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/14_scatter_tls_5y.png', dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# PLOT 15: Scatterplot - ESG vs Returns (5Y Overall)
# -----------------------------------------------------------------------------
print("  15. Scatterplot ESG vs returns (5Y)...")

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(df_5y['esg_z'], df_5y['AnnualReturn'], alpha=0.4, s=30, edgecolors='k',
           linewidths=0.3, color='green')

# Add regression line
z = np.polyfit(df_5y['esg_z'], df_5y['AnnualReturn'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_5y['esg_z'].min(), df_5y['esg_z'].max(), 100)
ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Fit: y={z[0]:.4f}x+{z[1]:.4f}')

ax.set_xlabel('MSCI ESG Score (z-score)', fontweight='bold')
ax.set_ylabel('5-Year Return', fontweight='bold')
ax.set_title('5-Year Stock Returns vs. MSCI ESG Score', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/15_scatter_esg_5y.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 8. WRITE DETAILED RESULTS TO TEXT FILE
# ============================================================================

print("\nWriting detailed results to file...")

with open('outputs/regression_results_detailed.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("COMPREHENSIVE FACTOR MODEL REGRESSION ANALYSIS\n")
    f.write("=" * 80 + "\n")
    f.write(f"\nAnalysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"1-Year Sample Size: {len(df_1y)} companies\n")
    f.write(f"5-Year Sample Size: {len(df_5y)} companies\n\n")

    # Write summary table
    f.write("=" * 80 + "\n")
    f.write("SUMMARY TABLE: ALL REGRESSIONS\n")
    f.write("=" * 80 + "\n\n")

    summary_table = summary_export[['Period', 'Group', 'N', 'ESG_R2', 'ESG_coef', 'ESG_pval',
                                     'TLS_R2', 'TLS_coef', 'TLS_pval',
                                     'Combined_R2', 'Combined_ESG_coef', 'Combined_TLS_coef']]
    f.write(summary_table.to_string(index=False))
    f.write("\n\n*** p<0.01, ** p<0.05, * p<0.10\n\n")

    # Detailed model outputs for Overall only
    f.write("=" * 80 + "\n")
    f.write("DETAILED REGRESSION OUTPUT: OVERALL SAMPLE\n")
    f.write("=" * 80 + "\n\n")

    for result in results_list:
        if result['Group'] == 'Overall':
            f.write(f"\n{'-'*80}\n")
            f.write(f"{result['Period']} RETURNS - OVERALL SAMPLE (N={result['N']})\n")
            f.write(f"{'-'*80}\n\n")

            f.write("Model 1: ESG-only\n")
            f.write(str(result['model1'].summary()))
            f.write("\n\n")

            f.write("Model 2: TLS-only\n")
            f.write(str(result['model2'].summary()))
            f.write("\n\n")

            f.write("Model 3: ESG + TLS\n")
            f.write(str(result['model3'].summary()))
            f.write("\n\n")

# ============================================================================
# 9. FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nOutputs saved to: factor_models/outputs/")
print(f"  - regression_summary_complete.csv (all results)")
print(f"  - regression_results_detailed.txt (detailed output)")
print(f"  - 15 visualization PNG files")
print(f"\nTotal regressions performed: {len(results_list)}")
print("=" * 80)
