import logging

logger = logging.getLogger(__name__)

def check_unique_ids(df, id_col='listing_id'):
    n_unique = df[id_col].nunique()
    n_total = len(df)
    n_null = df[id_col].isna().sum()
    assert n_unique == n_total, f"Duplicate IDs found: {n_unique} unique vs {n_total} total"
    assert n_null == 0, f"Null IDs found: {n_null}"
    logger.info(f"  [PASS] Unique IDs: {n_unique}")

def check_geometry_validity(gdf):
    n_invalid = (~gdf.geometry.is_valid).sum()
    n_empty = gdf.geometry.is_empty.sum()
    assert n_invalid == 0, f"Invalid geometries: {n_invalid}"
    assert n_empty == 0, f"Empty geometries: {n_empty}"
    logger.info(f"  [PASS] All geometries valid")

def check_crs(gdf, expected_crs='EPSG:4326'):
    actual = str(gdf.crs)
    assert actual == expected_crs, f"CRS mismatch: {actual} vs {expected_crs}"
    logger.info(f"  [PASS] CRS: {actual}")

def check_price_range(df, price_col='price', min_val=10, max_val=10000):
    n_outliers = ((df[price_col] < min_val) | (df[price_col] > max_val)).sum()
    logger.info(f"  [INFO] Price outliers (outside {min_val}-{max_val}): {n_outliers}")
