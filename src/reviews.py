import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def clean_and_aggregate_reviews(df):
    logger.info("Cleaning and aggregating reviews")
    col_map = {}
    for col in df.columns:
        col_map[col] = col.lower().strip()
    df = df.rename(columns=col_map)

    if 'listing_id' not in df.columns and 'id' in df.columns:
        df = df.rename(columns={'id': 'listing_id'})

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df = df.dropna(subset=['listing_id'])

    agg = df.groupby('listing_id').agg(
        review_count_total=('date', 'count'),
        first_review_date=('date', 'min'),
        last_review_date=('date', 'max'),
    ).reset_index()

    now = datetime.now()
    agg['days_since_last_review'] = (now - agg['last_review_date']).dt.days
    agg['months_active'] = ((agg['last_review_date'] - agg['first_review_date']).dt.days / 30.44).clip(lower=1)
    agg['reviews_per_month'] = agg['review_count_total'] / agg['months_active']

    logger.info(f"Reviews aggregated for {len(agg)} listings")
    return agg
