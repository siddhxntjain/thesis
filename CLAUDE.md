# Thesis Project: Measuring Energy Transition Exposure Through Corporate Disclosures

## Background & Motivation

The transition to a net-zero energy system has been called "arguably the largest engineering project undertaken by humankind." Achieving deep decarbonization will require unprecedented technological deployment across every sector—power generation, transportation, heavy industry. The Energy Transitions Commission estimates reaching net-zero by 2050 will demand ~$3.5 trillion in investment per year ($110 trillion total over 2021-2050).

**The Problem**: Investors seeking to support and profit from this transition face a critical challenge—**greenwashing**. ESG jargon and unsupported claims obscure which companies are truly accelerating decarbonization versus those merely engaging in performative sustainability theater. A 2023 PWC survey found 94% of investors "believe corporate reporting on sustainability performance contains unsupported claims." This misrepresentation leads to capital misallocation and erosion of trust.

## Research Objective

This project develops an **objective, data-driven transition exposure factor** using multiple data sources to measure a firm's genuine engagement with the energy transition. The factor will combine:

- **Textual analysis** of SEC 10-K filings (this codebase)
- **Financial data** (capex allocation, R&D spending, asset composition)
- **Operational metrics** (emissions intensity, renewable capacity, energy mix)
- **Market data** (stock returns, correlations with climate policy events)
- **Other quantitative signals** (patents, supply chain composition, etc.)

This codebase focuses on the **textual analysis component**—extracting transition language signals from mandatory SEC filings to distinguish substantive action from ESG boilerplate.

## This Component: Transition Language Score (TLS)

### What This Codebase Does

We parse 10-K filings for ~10,000 public companies and compute:

**TLS = (Substantive Energy-Transition Language) - (ESG Boilerplate)**

Both components are normalized per 10,000 words for comparability across firm size.

### Lexicon Construction

**Substantive Terms** (~80 terms):
- Concrete technologies: solar, wind, battery, hydrogen, geothermal, nuclear, CCUS
- Specific metrics: MW, GW, GWh, LCOE, capacity factors
- Action-oriented: electrification, decarbonization, retrofit, grid modernization
- Market mechanisms: PPAs, offtake agreements, IRA tax credits (45Q, 45V, 48C)
- Deployment indicators: inverter, microgrid, interconnection, BESS

**Boilerplate Terms** (~60 terms):
- Generic ESG frameworks: GRI, SASB, TCFD, CDP
- Vague commitments: sustainability values, climate leadership, stakeholder engagement
- Strategy buzzwords: ESG roadmap, low-carbon ambitions, sustainability initiatives
- Disclosure jargon: materiality assessment, ESG metrics, climate governance

### Weighting Methodologies

To test robustness and optimize signal extraction, we compute **6 variations** of TLS:

1. **Base**: Raw term counts (no weights)
2. **Section**: 1.5x weight for matches in material 10-K sections (Items 1, 1A, 7, 7A)
3. **Prox15**: 1.5x weight when terms appear near quantified mentions (±120 chars from "$500M", "2 GW", etc.)
4. **Prox20**: 2.0x weight for proximity to numbers (stronger quantification signal)
5. **Full15**: Section (1.5x) + Proximity (1.5x) = max 2.25x
6. **Full20**: Section (1.5x) + Proximity (2.0x) = max 3.0x

**Rationale for proximity weighting**: Terms like "solar" near "$500 million capex" or "2 GW capacity" indicate concrete action, not vague aspiration.

### Output Dataset

For each company-filing:
- **42 columns total**
- 6 identifiers (ticker, CIK, filing date, document)
- 18 raw metrics (6 methodologies × 3: substantive, boilerplate, TLS)
- 18 z-scores (for cross-company comparison)

## Integration into Broader Factor Model

**TLS is ONE input signal** that will be combined with other quantitative measures to construct the final transition exposure factor:

### Other Factor Components (Outside This Codebase)
- **Capex allocation**: % of capital expenditure toward low-carbon projects
- **Revenue exposure**: % revenue from clean energy products/services
- **Emissions trajectory**: Year-over-year change in Scope 1, 2, 3 emissions intensity
- **Asset composition**: Renewable vs. fossil fuel generation capacity (for utilities/energy)
- **Patent filings**: Clean energy technology patents
- **Supply chain**: Upstream/downstream transition exposure
- **Market signals**: Stock return sensitivity to climate policy news

### Factor Construction Process
1. **Compute TLS scores** (this codebase) → text-based signal
2. **Merge with financial data** → capex, emissions, revenues
3. **Normalize & weight components** → combine into composite transition score
4. **Rank companies** → form quintiles based on composite score
5. **Build long-short portfolio** → Long Q5 (high transition), Short Q1 (low transition)
6. **Test factor performance** → Analyze returns, correlations, alpha vs. Fama-French factors

### Why Textual Analysis Matters

While financial/operational data is critical, **language analysis adds unique value**:

- **Forward-looking**: Capex is backward-looking; language signals future intent
- **Qualitative context**: Numbers miss the "why" and "how" of strategy
- **Greenwashing detection**: TLS explicitly measures gap between talk and quantification
- **Early signal**: Language shifts may precede financial statement changes
- **Comprehensive**: Captures R&D, partnerships, strategy beyond just capex

**TLS is not the full factor—it's a complementary signal** that captures information orthogonal to purely financial metrics.

## Expected Results & Validation

### Hypothesis
Companies with high TLS (especially proximity-weighted variants) should exhibit:
- **Higher transition-related capex** (verifiable from cash flow statements)
- **Measurable emissions reductions** (Scope 1, 2, 3 trends)
- **Stock returns correlated with climate policy events** (IRA passage, carbon pricing)
- **Actual deployment of clean energy capacity** (vs. aspirational targets)

### Validation Strategy
1. **Correlation with financial data**: Regress TLS against disclosed renewable/low-carbon capex
2. **Emissions intensity**: Compare TLS rankings with carbon intensity trends
3. **Incremental information**: Does TLS add explanatory power beyond capex/emissions alone?
4. **Cross-validation**: Do high-TLS companies also score high on other transition metrics?

## Implementation

### Data Pipeline
1. **Ticker mapping**: 10,000+ public companies from SEC EDGAR
2. **10-K downloads**: Most recent 1-3 filings per company via EDGAR API
3. **Text extraction**: Parse HTML, remove boilerplate (headers, tables, exhibits)
4. **Lexicon matching**: Regex-based term detection with context awareness
5. **Weighting**: Apply section/proximity multipliers
6. **Normalization**: Scale by document length (per 10k words)
7. **Z-scoring**: Cross-sectional standardization for comparison

### File Structure
```
thesis/
├── edgar.py                          # Main NLP pipeline
├── substantive_terms.txt             # Transition lexicon (~80 terms)
├── boilerplate_terms.txt             # ESG jargon (~60 terms)
├── tickers.txt                       # All public company tickers
├── edgar_cache/company_tickers.json  # SEC ticker-to-CIK mapping
├── METHODOLOGY_VARIATIONS.md         # Technical documentation
└── CLAUDE.md                         # This file
```

### Runtime
- ~5 seconds per ticker (SEC rate limits)
- ~14 hours for full dataset (10k companies × 3 filings)

## Next Steps (This Component)

1. **Complete data collection**: Process all 10k companies × 3 filings
2. **Validate TLS scores**: Compare against capex, emissions
3. **Optimize methodology**: Test which of 6 weighting variants maximizes signal
4. **Merge with financial data**: Combine TLS with other factor components
5. **Temporal analysis**: Track TLS evolution 2020-2025 (post-IRA)

## Academic Context

This work contributes to:
- **Corporate climate disclosure** (Ilhan et al., 2021; Krueger et al., 2020): Using text to measure transition risk
- **Greenwashing detection** (Lyon & Montgomery, 2015; Delmas & Burbano, 2011): Quantifying substantive vs. symbolic action
- **Textual analysis in finance** (Loughran & McDonald, 2011): Domain-specific lexicons for asset pricing
- **ESG ratings divergence** (Berg et al., 2022; Christensen et al., 2022): Alternative to proprietary ratings

## Key Takeaway

This codebase generates the **textual analysis component** of a broader transition exposure factor. By analyzing language in mandatory SEC filings, we extract signals about companies' transition commitments that complement financial and operational metrics. The final factor will integrate TLS with capex, emissions, revenue exposure, and other quantitative data to provide a comprehensive measure of which firms are genuinely driving decarbonization.

---

*For technical implementation details, see `METHODOLOGY_VARIATIONS.md`*
