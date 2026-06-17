import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def parse_price_series(s):
    if s is None:
        return pd.Series(dtype=float)
    s = s.astype(str)
    s = s.str.replace(r'[$£€]', '', regex=True)
    s = s.str.replace(r'\bEUR\b', '', regex=True)
    s = s.str.replace(r'\bGBP\b', '', regex=True)
    s = s.str.replace('\u00a0', '', regex=False)
    s = s.str.replace(r'\s+', '', regex=True)
    s = s.str.strip()
    s = s.replace({'': np.nan, 'nan': np.nan, 'None': np.nan})

    def _parse_single(val):
        if pd.isna(val):
            return np.nan
        val = str(val)
        if ',' in val and '.' in val:
            last_comma = val.rfind(',')
            last_dot = val.rfind('.')
            if last_comma > last_dot:
                val = val.replace('.', '').replace(',', '.')
            else:
                val = val.replace(',', '')
        elif ',' in val:
            val = val.replace(',', '.')
        try:
            return float(val)
        except ValueError:
            return np.nan

    return s.apply(_parse_single)

def clean_calendar(df):
    logger.info("Cleaning calendar data")
    col_map = {}
    for col in df.columns:
        col_map[col] = col.lower().strip()
    df = df.rename(columns=col_map)

    if 'listing_id' not in df.columns and 'id' in df.columns:
        df = df.rename(columns={'id': 'listing_id'})

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    if 'available' in df.columns:
        df['available'] = df['available'].map({'t': 1, 'f': 0, 'T': 1, 'F': 0, True: 1, False: 0})
        df['available'] = pd.to_numeric(df['available'], errors='coerce')

    if 'price' in df.columns:
        df['price'] = parse_price_series(df['price'])

    logger.info(f"Calendar cleaned: {len(df)} rows")
    return df

def clean_listings(df):
    logger.info("Cleaning listings data")
    col_map = {}
    for col in df.columns:
        col_map[col] = col.lower().strip()
    df = df.rename(columns=col_map)

    if 'id' in df.columns and 'listing_id' not in df.columns:
        df = df.rename(columns={'id': 'listing_id'})

    if 'price' in df.columns:
        df['price'] = parse_price_series(df['price'])

    from .config import PRICE_OUTLIER_THRESHOLD_LOW, PRICE_OUTLIER_THRESHOLD_HIGH
    mask_price = (df['price'] >= PRICE_OUTLIER_THRESHOLD_LOW) & (df['price'] <= PRICE_OUTLIER_THRESHOLD_HIGH)
    n_dropped = (~mask_price).sum()
    df = df[mask_price].copy()
    logger.info(f"Dropped {n_dropped} listings with price outliers")

    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        mask_coords = df['latitude'].notna() & df['longitude'].notna()
        mask_coords &= (df['latitude'] != 0) & (df['longitude'] != 0)
        n_dropped = (~mask_coords).sum()
        df = df[mask_coords].copy()
        logger.info(f"Dropped {n_dropped} listings with invalid coordinates")

    if 'room_type' in df.columns:
        df['room_type'] = df['room_type'].str.strip().str.lower()

    df = df.drop_duplicates(subset=['listing_id'])
    logger.info(f"Listings cleaned: {len(df)} rows")
    return df
