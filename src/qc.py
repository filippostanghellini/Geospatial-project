import logging

logger = logging.getLogger(__name__)

def check_unique_ids(df, id_col='listing_id'):
    n_unique = df[id_col].nunique()
    n_total = len(df)
    n_null = df[id_col].isna().sum()
    assert n_unique == n_total, f"Duplicate IDs found: {n_unique} unique vs {n_total} total"
    assert n_null == 0, f"Null IDs found: {n_null}"
    logger.info(f"  [PASS] Unique IDs: {n_unique}")

def check_no_negative_ids(df, id_col='listing_id'):
    n_neg = (df[id_col] < 0).sum()
    assert n_neg == 0, f"Negative IDs found: {n_neg}"
    logger.info(f"  [PASS] No negative IDs")

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

def check_availability_values(df):
    if 'available' in df.columns:
        unique_vals = df['available'].dropna().unique()
        assert set(unique_vals).issubset({0, 1, 0.0, 1.0}), f"Non-binary availability: {unique_vals}"
        logger.info(f"  [PASS] Availability is binary")

def check_date_range(df):
    if 'date' in df.columns:
        import pandas as pd
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        date_min = df['date'].min()
        date_max = df['date'].max()
        n_null = df['date'].isna().sum()
        logger.info(f"  [INFO] Date range: {date_min} to {date_max}, nulls: {n_null}")

def check_spatial_join_coverage(gdf, min_coverage=95):
    if 'index_right' in gdf.columns or 'neighbourhood' in gdf.columns:
        coverage = len(gdf.dropna(subset=['index_right'] if 'index_right' in gdf.columns else ['neighbourhood'])) / len(gdf) * 100
        assert coverage >= min_coverage, f"Spatial join coverage too low: {coverage:.1f}%"
        logger.info(f"  [PASS] Spatial join coverage: {coverage:.1f}%")

def print_qc_report(data_dict, checks):
    logger.info("=" * 60)
    logger.info("QC REPORT")
    logger.info("=" * 60)
    for name, (df, check_list) in checks.items():
        logger.info(f"\n--- {name} ---")
        for check_fn, kwargs in check_list:
            try:
                check_fn(df, **kwargs)
            except AssertionError as e:
                logger.error(f"  [FAIL] {e}")
            except Exception as e:
                logger.error(f"  [ERROR] {e}")
    logger.info("\n" + "=" * 60)
