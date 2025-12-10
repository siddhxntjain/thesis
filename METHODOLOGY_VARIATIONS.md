# EDGAR NLP Methodology Variations

## Overview
The updated `edgar.py` now computes **18 different scoring variations** from each filing (plus z-scores), allowing you to test different NLP methodologies without re-downloading the filings.

## Recent Changes

### Key Improvements
1. **Removed Arbitrary 20k Character Limit**: Section weighting now captures ENTIRE sections (from header until next Item header)
2. **Added Proximity Multiplier Variations**: Tests two proximity multipliers (1.5x and 2.0x) to compare moderate vs strong emphasis on quantified mentions
3. **Removed Legacy Columns**: Cleaned up output schema with consistent naming (`_base`, `_section`, `_prox15`, `_prox20`, `_full15`, `_full20`)
4. **Moved Lexicons to External Files**:
   - `substantive_terms.txt`: Energy-transition language terms (default)
   - `boilerplate_terms.txt`: Generic ESG language terms (default)
   - Easy to modify terms without editing code

## Dataset Structure

### Core Metrics (3 types):
1. **substantive**: Count of energy-transition language (renewable, solar, wind, hydrogen, battery, etc.)
2. **boilerplate**: Count of generic ESG language (ESG, CSR, sustainability values, etc.)
3. **tls** (Transition Language Score): substantive - boilerplate

### Weighting Methodologies (6 variations):

#### 1. **Base** (No Weighting)
- Pure word counts normalized per 10,000 words
- No section weights, no proximity boosts
- Columns: `substantive_base`, `boilerplate_base`, `tls_base`
- **Use case**: Baseline comparison, simplest approach

#### 2. **Section-Weighted**
- Applies 1.5x multiplier to matches in key 10-K sections:
  - Item 1: Business
  - Item 1A: Risk Factors
  - Item 7: MD&A
  - Item 7A: Quantitative Market Risk
- Weight applied to ENTIRE section (from header until next Item header or end of document)
- No arbitrary character limits - captures full section content
- Columns: `substantive_section`, `boilerplate_section`, `tls_section`
- **Use case**: Emphasizes material operational disclosures

#### 3. **Proximity-Weighted (1.5x multiplier)**
- Applies 1.5x multiplier when substantive terms appear near quantified mentions
- Searches ±120 characters (~8-10 words) for patterns like:
  - "$1.2B", "500 MW", "2 GWh", "300 million capex"
- Only applies to substantive terms (not boilerplate)
- Columns: `substantive_prox15`, `boilerplate_prox15`, `tls_prox15`
- **Use case**: Moderate emphasis on concrete, quantified commitments

#### 4. **Proximity-Weighted (2.0x multiplier)**
- Same as above but with stronger 2.0x multiplier for numeric proximity
- Columns: `substantive_prox20`, `boilerplate_prox20`, `tls_prox20`
- **Use case**: Strong emphasis on quantified commitments vs vague language

#### 5. **Full Weighting (Section 1.5x + Proximity 1.5x)**
- Combines section and moderate proximity weighting
- Match can get up to 2.25x weight (1.5 × 1.5) if in key section AND near numbers
- Columns: `substantive_full15`, `boilerplate_full15`, `tls_full15`
- **Use case**: Balanced signal extraction, emphasizes material + quantified language

#### 6. **Full Weighting (Section 1.5x + Proximity 2.0x)**
- Combines section and strong proximity weighting
- Match can get up to 3.0x weight (1.5 × 2.0) if in key section AND near numbers
- Columns: `substantive_full20`, `boilerplate_full20`, `tls_full20`
- **Use case**: Maximum signal extraction, strongest emphasis on material + quantified language

## All Output Columns (42 total)

### Identifiers (6)
- `ticker`, `cik`, `filing_date`, `accession`, `doc`, `tokens`

### Base Metrics (6)
- `substantive_base`, `boilerplate_base`, `tls_base`
- `substantive_base_z`, `boilerplate_base_z`, `tls_base_z`

### Section-Weighted Metrics (6)
- `substantive_section`, `boilerplate_section`, `tls_section`
- `substantive_section_z`, `boilerplate_section_z`, `tls_section_z`

### Proximity-Weighted Metrics - 1.5x (6)
- `substantive_prox15`, `boilerplate_prox15`, `tls_prox15`
- `substantive_prox15_z`, `boilerplate_prox15_z`, `tls_prox15_z`

### Proximity-Weighted Metrics - 2.0x (6)
- `substantive_prox20`, `boilerplate_prox20`, `tls_prox20`
- `substantive_prox20_z`, `boilerplate_prox20_z`, `tls_prox20_z`

### Full Weighted Metrics - 1.5x proximity (6)
- `substantive_full15`, `boilerplate_full15`, `tls_full15`
- `substantive_full15_z`, `boilerplate_full15_z`, `tls_full15_z`

### Full Weighted Metrics - 2.0x proximity (6)
- `substantive_full20`, `boilerplate_full20`, `tls_full20`
- `substantive_full20_z`, `boilerplate_full20_z`, `tls_full20_z`

## Additional Parameters You Can Vary

### Hardcoded in edgar.py (could be made configurable):
1. **Section weight multiplier**: Currently 1.5x (line 235)
2. **Proximity weight multipliers**: Currently testing 1.5x and 2.0x (line 333, 342)
3. **Proximity search window**: Currently ±120 chars (line 283)

### Lexicon variations:
- Modify `SUBSTANTIVE_TERMS` list (lines 40-81)
- Modify `BOILERPLATE_TERMS` list (lines 83-95)
- Use `--substantive-file` and `--boilerplate-file` to test different term lists

### Filing variations:
- `--max-filings N`: Analyze N most recent 10-Ks per ticker (1, 3, 5, etc.)

## Research Questions You Can Answer

1. **Does section-weighting improve signal?**
   - Compare correlations of `tls_base` vs `tls_section` with stock returns / carbon emissions

2. **What's the optimal proximity multiplier?**
   - Compare 1.5x vs 2.0x proximity weights
   - Does stronger weighting (2.0x) better identify real action vs greenwashing?

3. **What's the optimal combination?**
   - Test all 18 variations against external validation metrics
   - Which methodology best predicts actual capital deployment or emissions reduction?

4. **Does boilerplate subtract signal or just add noise?**
   - Compare `substantive_*` metrics vs `tls_*` metrics
   - Does subtracting boilerplate improve predictive power?

5. **Time-series: Are companies getting more substantive?**
   - With `--max-filings 3` or `5`, track evolution of metrics over time
   - Are proximity-weighted scores increasing faster than base scores?

6. **Do section and proximity weights interact?**
   - Compare `prox15` vs `full15` (additive effect of section weighting)
   - Compare `prox20` vs `full20`

## How edgar.py Works (Step-by-Step)

### 1. **Load Lexicons**
- Reads `substantive_terms.txt` (energy-transition terms)
- Reads `boilerplate_terms.txt` (generic ESG terms)
- Compiles all terms into regex patterns for fast matching

### 2. **Load Ticker Mapping**
- Reads `edgar_cache/company_tickers.json`
- Maps ticker symbols to SEC CIK numbers (required for EDGAR API)

### 3. **Resume Support**
- Checks if output CSV exists
- Loads already-processed tickers to skip them
- Continues from where it left off if interrupted

### 4. **Process Each Ticker**
For each ticker in the input list:
- Look up CIK from ticker symbol
- Download recent filings metadata from SEC EDGAR API
- Filter to 10-K and 10-K/A forms
- Select N most recent filings (per `--max-filings` parameter)

### 5. **Analyze Each Filing**
For each 10-K document:
- Download HTML from EDGAR archive
- Extract plain text (remove scripts, styles, markup)
- Identify key 10-K sections (Items 1, 1A, 7, 7A)
- Compute 18 scoring variations:
  - **Base**: Raw counts (no weights)
  - **Section**: 1.5x weight for key sections
  - **Prox15**: 1.5x weight near quantified mentions
  - **Prox20**: 2.0x weight near quantified mentions
  - **Full15**: Section (1.5x) + Prox (1.5x) = max 2.25x
  - **Full20**: Section (1.5x) + Prox (2.0x) = max 3.0x
- For each variation, compute 3 metrics:
  - Substantive count per 10k words
  - Boilerplate count per 10k words
  - TLS (substantive - boilerplate) per 10k words

### 6. **Save Results Continuously**
- After each ticker completes, immediately append to CSV
- Prevents data loss if script is interrupted
- Progress bar shows completion status

### 7. **Final Z-Score Computation**
- Loads all results from CSV
- Computes z-scores for all 18 variations
- Overwrites CSV with final dataset including z-scores

### 8. **Output**
- CSV with 42 columns:
  - 6 identifiers (ticker, cik, filing_date, etc.)
  - 36 metrics (18 variations × 2 forms: raw + z-score)

## Running the Full Analysis

```bash
source .venv/bin/activate && python edgar.py --tickers-file tickers.txt --max-filings 3 --out tls_scores_all.csv
```

This single run will give you 18 variations × ~10k companies × 3 filings = ~30k data points with 18 different methodologies to compare!

## Cost-Benefit Analysis

**Time cost**: ~14 hours for 10,142 tickers × 3 filings at 5 sec/ticker
**Benefit**: 18 different methodologies without re-downloading (would take 14 hours × 18 = 252 hours if run separately!)

## Methodology Comparison Matrix

| Methodology | Section Weight | Proximity Weight | Max Combined Weight | Best For |
|-------------|----------------|------------------|---------------------|----------|
| base | 1.0x | 1.0x | 1.0x | Baseline, unbiased counts |
| section | 1.5x | 1.0x | 1.5x | Material disclosures emphasis |
| prox15 | 1.0x | 1.5x | 1.5x | Moderate quantification emphasis |
| prox20 | 1.0x | 2.0x | 2.0x | Strong quantification emphasis |
| full15 | 1.5x | 1.5x | 2.25x | Balanced material + quantified |
| full20 | 1.5x | 2.0x | 3.0x | Maximum signal extraction |

---

## Running the Script

### Prerequisites
Ensure these files exist in your directory:
- `edgar_cache/company_tickers.json` - SEC ticker mapping
- `substantive_terms.txt` - Energy-transition lexicon
- `boilerplate_terms.txt` - ESG boilerplate lexicon
- `tickers.txt` - List of ticker symbols (space or comma separated)

### Command
```bash
source .venv/bin/activate && python edgar.py --tickers-file tickers.txt --max-filings 3 --out tls_scores_all.csv
```

### Parameters
- `--tickers-file tickers.txt` - Input file with ticker symbols
- `--max-filings 3` - Number of most recent 10-Ks per ticker
- `--out tls_scores_all.csv` - Output file path
- `--substantive-file substantive_terms.txt` - Optional (defaults to this)
- `--boilerplate-file boilerplate_terms.txt` - Optional (defaults to this)

### Resume Support
If interrupted, simply run the same command again. The script will:
1. Load existing results from `tls_scores_all.csv`
2. Skip already-processed tickers
3. Continue from where it stopped

### Expected Runtime
- **~5 seconds per ticker** (SEC rate limits)
- **10,142 tickers × 3 filings = ~14 hours total**
- Progress bar shows real-time status
