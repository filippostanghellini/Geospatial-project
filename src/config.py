from pathlib import Path

def get_project_root():
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "data").is_dir() and (parent / "src").is_dir():
            return parent
    raise RuntimeError("Could not find project root")

PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
ORIGINAL_DIR = DATA_DIR / "original"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
MAPS_DIR = OUTPUTS_DIR / "maps"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
REPORT_MAPS_DIR = REPORTS_DIR / "maps"

for d in [PROCESSED_DIR, TABLES_DIR, MAPS_DIR, FIGURES_DIR, REPORT_MAPS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

INPUT_FILES = {
    "listings_detailed": ORIGINAL_DIR / "listings.csv",
    "calendar_detailed": ORIGINAL_DIR / "calendar.csv",
    "reviews_detailed": ORIGINAL_DIR / "reviews.csv",
    "neighbourhoods_geojson": ORIGINAL_DIR / "neighbourhoods.geojson",
}

OUTPUT_FILES = {
    "listings_clean": PROCESSED_DIR / "listings_clean.parquet",
    "calendar_clean": PROCESSED_DIR / "calendar_clean.parquet",
    "reviews_listing_features": PROCESSED_DIR / "reviews_listing_features.parquet",
    "neighbourhoods_enriched": PROCESSED_DIR / "neighbourhoods_enriched.parquet",
    "neighbourhoods_enriched_geojson": PROCESSED_DIR / "neighbourhoods_enriched.geojson",
    "model_sample": PROCESSED_DIR / "model_sample.parquet",
    "listings_points_enriched_sample": PROCESSED_DIR / "listings_points_enriched_sample.geojson",
    "map_points_sample": PROCESSED_DIR / "map_points_sample.geojson",
    "map_grid_cells": PROCESSED_DIR / "map_grid_cells.geojson",
    "residuals_for_map": TABLES_DIR / "residuals_for_map.csv",
    "ols_coefficients": TABLES_DIR / "ols_model_b_coefficients.csv",
    "morans_results": TABLES_DIR / "morans_results.csv",
    "lm_diagnostic_tests": TABLES_DIR / "lm_diagnostic_tests.csv",
    "model_comparison": TABLES_DIR / "model_comparison.csv",
    "spatial_effects": TABLES_DIR / "spatial_effects.csv",
    "spillover_listings": TABLES_DIR / "spillover_listings.csv",
    "spillover_neighbourhoods": MAPS_DIR / "spillover_neighbourhoods.geojson",
}

CRS_WEB = "EPSG:4326"
CRS_METRIC = "EPSG:32632"

CBD_LAT = 45.4642
CBD_LON = 9.1900
CBD_NAME = "Piazza del Duomo"

PRICE_OUTLIER_THRESHOLD_LOW = 10
PRICE_OUTLIER_THRESHOLD_HIGH = 10000

LISTINGS_SAMPLE_SIZE = 500
MAP_SAMPLE_SIZE = 5000
RANDOM_SEED = 42

CITY_NAME = "Milan"
COUNTRY = "Italy"
SNAPSHOT_DATE = "2025-09-22"
