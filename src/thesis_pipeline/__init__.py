from .canonical import BuildConfig, build_all_canonical_tables

TLS_LOCK = (3.25, 1.0, 10)
R1000_SENTIMENT_LOCK = {
    "config_id": "finbert_core_c30_h1_filter_off_repeat_on_j088_f256",
    "score_name": "transition_sentiment_median",
    "transform": "winsor_5_95",
    "missing_score_policy": "drop",
}
T100_SENTIMENT_LOCK = {
    "config_id": "finbert_core_c30_h1_filter_off_repeat_on_j088_f256",
    "score_name": "transition_sentiment_median",
    "transform": "winsor_5_95",
    "missing_score_policy": "drop",
}
YEARS = list(range(2015, 2025))
FACTOR_CUTOFFS = [0.10, 0.25, 1.0 / 3.0]
