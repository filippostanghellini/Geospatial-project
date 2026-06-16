import pandas as pd
import numpy as np
import logging
from .config import CBD_LAT, CBD_LON, CRS_WEB

logger = logging.getLogger(__name__)

def _parse_price_series(s):
    s = s.astype(str).str.replace('EUR', '', regex=False)
    s = s.str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce')

def load_listings_processed(path):
    df = pd.read_parquet(path)
    if 'listing_id' not in df.columns and 'id' in df.columns:
        df = df.rename(columns={'id': 'listing_id'})
    if 'price_numeric' not in df.columns and 'price' in df.columns:
        df['price_numeric'] = _parse_price_series(df['price'])
    return df

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def build_model_df(df, out_path=None, winsorize_q=(0.005, 0.995)):
    logger.info("Building model dataset")

    df = df.copy()
    if 'price' not in df.columns and 'price_numeric' in df.columns:
        df['price'] = df['price_numeric']

    df = df[df['price'].notna()].copy()
    logger.info(f"After price filter: {len(df)} rows")

    low_q, high_q = winsorize_q
    low_val = df['price'].quantile(low_q)
    high_val = df['price'].quantile(high_q)
    df['price'] = df['price'].clip(low_val, high_val)
    logger.info(f"Winsorized price to [{low_val:.2f}, {high_val:.2f}]")

    df['log_price'] = np.log(df['price'])

    for col in ['bedrooms', 'beds', 'bathrooms']:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    if 'host_is_superhost' in df.columns:
        df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0, 'True': 1, 'False': 0, True: 1, False: 0})
        df['host_is_superhost'] = pd.to_numeric(df['host_is_superhost'], errors='coerce').fillna(0).astype(int)

    if 'instant_bookable' in df.columns:
        df['instant_bookable'] = df['instant_bookable'].map({'t': 1, 'f': 0, 'True': 1, 'False': 0, True: 1, False: 0})
        df['instant_bookable'] = pd.to_numeric(df['instant_bookable'], errors='coerce').fillna(0).astype(int)

    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['dist_cbd_km'] = _haversine(df['latitude'], df['longitude'], CBD_LAT, CBD_LON)

    if 'room_type' in df.columns:
        room_dummies = pd.get_dummies(df['room_type'], prefix='room_type', drop_first=True, dtype=int)
        df = pd.concat([df, room_dummies], axis=1)

    if 'neighbourhood_cleansed' in df.columns:
        neigh_dummies = pd.get_dummies(df['neighbourhood_cleansed'], prefix='neigh', drop_first=True, dtype=int)
        df = pd.concat([df, neigh_dummies], axis=1)

    df = df.dropna(subset=['log_price', 'accommodates', 'minimum_nights'])
    logger.info(f"Final model dataset: {len(df)} rows")

    if out_path:
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved model dataset to {out_path}")

    return df

def get_y_X(model_df):
    exclude_cols = {'listing_id', 'id', 'latitude', 'longitude', 'price', 'price_numeric',
                    'log_price', 'geometry', 'neighbourhood_cleansed', 'neighbourhood',
                    'room_type', 'host_is_superhost', 'instant_bookable'}
    X_cols = [c for c in model_df.columns if c not in exclude_cols and model_df[c].dtype in ['int64', 'float64', 'int32', 'float32', 'int8', 'uint8']]

    X_df = model_df[X_cols].copy()
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    cols_with_nan = X_df.columns[X_df.isna().any()].tolist()
    if cols_with_nan:
        logger.warning(f"Dropping columns with NaN/inf: {cols_with_nan}")
        X_df = X_df.drop(columns=cols_with_nan)
        X_cols = [c for c in X_cols if c not in cols_with_nan]

    mask = X_df.notna().all(axis=1)
    if mask.sum() < len(model_df):
        logger.info(f"Dropping {len(model_df) - mask.sum()} rows with NaN in model features")
        model_df = model_df[mask].copy()
        X_df = X_df[mask].copy()

    y = model_df['log_price'].values
    X = X_df.values
    return y, X, X_cols
