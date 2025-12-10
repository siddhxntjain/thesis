import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Hardcode your CSV path here ---
CSV_PATH = "tls_scoresp2.csv"   # change to your full path if needed

# --- Load ---
df = pd.read_csv(CSV_PATH, parse_dates=["filing_date"])

# --- Basic info ---
print("\n=== Basic Info ===")
print(df.info())
print("\nRows:", len(df), " | Columns:", list(df.columns))

# --- Descriptive statistics ---
print("\n=== Descriptive Stats (key columns) ===")
cols = ["tokens", "substantive_per10k", "boilerplate_per10k", "tls_raw", "tls_z"]
print(df[cols].describe().round(2))

# --- Latest filing per ticker ---
latest = (
    df.sort_values("filing_date")
      .groupby("ticker")
      .tail(1)
      .sort_values("tls_z", ascending=False)
)

print("\n=== Latest Filing per Ticker (ranked by tls_z) ===")
print(latest[["ticker", "filing_date", "tls_z", "tls_raw",
              "substantive_per10k", "boilerplate_per10k"]]
      .to_string(index=False, justify="center", col_space=12))

# --- Mean TLS by ticker ---
mean = (
    df.groupby("ticker")[cols]
      .mean()
      .sort_values("tls_z", ascending=False)
      .reset_index()
)

print("\n=== Mean TLS z-score by Ticker ===")
print(mean[["ticker", "tls_z", "tls_raw", "substantive_per10k", "boilerplate_per10k"]]
      .round(2)
      .to_string(index=False, justify="center", col_space=12))

# --- Ranking: Most Substantive Tokens ---
sub_rank = (
    df.groupby("ticker")["substantive_per10k"]
      .mean()
      .sort_values(ascending=False)
      .reset_index()
)

print("\n=== Ranking by Most Substantive Tokens (average per 10k words) ===")
print(sub_rank.head(10).to_string(index=False, justify="center", col_space=12))

# --- Simple highlights ---
print("\n=== Top 5 'Green' Firms by TLS ===")
print(mean.head(5)[["ticker","tls_z"]].to_string(index=False))

print("\n=== Bottom 5 'Brown' Firms by TLS ===")
print(mean.tail(5)[["ticker","tls_z"]].to_string(index=False))

# ==============================
#            PLOTS
# ==============================

# Helper to save and show each plot
def save_show(fig, filename, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.show()

# 1) Distribution of tls_z across all filings
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.hist(df["tls_z"].dropna(), bins=30)
ax1.set_title("Distribution of TLS z-scores")
ax1.set_xlabel("tls_z")
ax1.set_ylabel("Count")
save_show(fig1, "tls_z_distribution.png")

# 2) Substantive vs Boilerplate scatter
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(df["boilerplate_per10k"], df["substantive_per10k"], alpha=0.6)
ax2.set_title("Substantive vs Boilerplate (per 10k words)")
ax2.set_xlabel("boilerplate_per10k")
ax2.set_ylabel("substantive_per10k")
save_show(fig2, "substantive_vs_boilerplate_scatter.png")

# 3) Top 10 tickers by mean tls_z (bar chart)
top10_tls = mean.head(10)
fig3 = plt.figure(figsize=(10, 5))
ax3 = fig3.add_subplot(111)
ax3.bar(top10_tls["ticker"], top10_tls["tls_z"])
ax3.set_title("Top 10 tickers by mean TLS z-score")
ax3.set_xlabel("Ticker")
ax3.set_ylabel("Mean tls_z")
ax3.tick_params(axis="x", rotation=45)
plt.xticks(ha="right")  # sets horizontal alignment
save_show(fig3, "top10_tlsz_bar.png")

# 4) Top 10 tickers by mean substantive_per10k (bar chart)
top10_sub = sub_rank.head(10)
fig4 = plt.figure(figsize=(10, 5))
ax4 = fig4.add_subplot(111)
ax4.bar(top10_sub["ticker"], top10_sub["substantive_per10k"])
ax4.set_title("Top 10 tickers by substantive content (avg per 10k words)")
ax4.set_xlabel("Ticker")
ax4.set_ylabel("Mean substantive_per10k")
ax4.tick_params(axis="x", rotation=45)
plt.xticks(ha="right")  # sets horizontal alignment
save_show(fig4, "top10_substantive_bar.png")

# 5) Time trend of average tls_z by month
df_month = (
    df.dropna(subset=["filing_date"])
      .assign(month=lambda d: d["filing_date"].dt.to_period("M").dt.to_timestamp())
      .groupby("month")["tls_z"]
      .mean()
      .reset_index()
      .sort_values("month")
)

if len(df_month) > 0:
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    ax5.plot(df_month["month"], df_month["tls_z"], marker="o")
    ax5.set_title("Average TLS z-score over time")
    ax5.set_xlabel("Month")
    ax5.set_ylabel("Mean tls_z")
    plt.xticks(rotation=45, ha="right")
    save_show(fig5, "tlsz_over_time.png")

# 6) Latest filing per ticker ranked by tls_z (bar chart)
latest_sorted = latest.sort_values("tls_z", ascending=False).head(20)
fig6 = plt.figure(figsize=(10, 6))
ax6 = fig6.add_subplot(111)
ax6.bar(latest_sorted["ticker"], latest_sorted["tls_z"])
ax6.set_title("Latest filing per ticker ranked by tls_z")
ax6.set_xlabel("Ticker")
ax6.set_ylabel("Latest tls_z")
ax6.tick_params(axis="x", rotation=45)
plt.xticks(ha="right")  # sets horizontal alignment
save_show(fig6, "latest_tlsz_bar.png")

print("\nSaved figures:")
print(" - tls_z_distribution.png")
print(" - substantive_vs_boilerplate_scatter.png")
print(" - top10_tlsz_bar.png")
print(" - top10_substantive_bar.png")
print(" - tlsz_over_time.png")
print(" - latest_tlsz_bar.png")