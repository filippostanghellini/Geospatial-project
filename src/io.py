import pandas as pd
import geopandas as gpd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_csv(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV not found: {filepath}")
    logger.info(f"Loading CSV: {filepath}")
    return pd.read_csv(filepath)

def load_geojson(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"GeoJSON not found: {filepath}")
    logger.info(f"Loading GeoJSON: {filepath}")
    gdf = gpd.read_file(filepath)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf

def load_parquet(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Parquet not found: {filepath}")
    logger.info(f"Loading Parquet: {filepath}")
    try:
        return gpd.read_parquet(filepath)
    except (ValueError, Exception):
        return pd.read_parquet(filepath)

def save_parquet(df, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving Parquet: {filepath}")
    df.to_parquet(filepath, index=False)

def save_geojson(gdf, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving GeoJSON: {filepath}")
    if gdf.crs and str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(filepath, driver="GeoJSON")

def save_csv(df, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving CSV: {filepath}")
    df.to_csv(filepath, index=False)
