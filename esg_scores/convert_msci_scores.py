import pandas as pd

# Read the MSCI scores data
df = pd.read_csv('msci_scores.csv')

# Define the rating to numerical score mapping
# MSCI ESG Rating scale: AAA (best) to CCC (worst)
rating_map = {
    'AAA': 7,
    'AA': 6,
    'A': 5,
    'BBB': 4,
    'BB': 3,
    'B': 2,
    'CCC': 1
}

# Convert ratings to numerical scores
df['msci_esg_score'] = df['iva_company_rating'].map(rating_map)

# Select only ticker and numerical score
result = df[['issuer_ticker', 'msci_esg_score']].copy()
result.columns = ['ticker', 'msci_esg_score']

# Remove duplicates (keeping first occurrence)
result = result.drop_duplicates(subset='ticker', keep='first')

# Sort by ticker
result = result.sort_values('ticker').reset_index(drop=True)

# Save to CSV
result.to_csv('msci_esg_numerical.csv', index=False)

print(f"Processed {len(result)} unique tickers")
print(f"\nRating distribution:")
print(df['iva_company_rating'].value_counts().sort_index())
print(f"\nNumerical score distribution:")
print(result['msci_esg_score'].value_counts().sort_index())
print(f"\nFirst 10 rows:")
print(result.head(10))
