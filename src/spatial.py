import geopandas as gpd
import numpy as np
import logging
from .config import CRS_WEB, CRS_METRIC

logger = logging.getLogger(__name__)

def listings_to_geodataframe(df):
    logger.info("Converting listings to GeoDataFrame")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
        crs=CRS_WEB
    )
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, 'geometry'] = gdf.loc[invalid, 'geometry'].buffer(0)
        logger.info(f"Repaired {invalid.sum()} invalid geometries")
    return gdf

def clean_neighbourhoods(gdf):
    logger.info("Cleaning neighbourhoods")
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_WEB)
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, 'geometry'] = gdf.loc[invalid, 'geometry'].buffer(0)
        logger.info(f"Repaired {invalid.sum()} invalid neighbourhood geometries")
    return gdf

def spatial_join_listings_neighbourhoods(gdf_listings, gdf_neigh):
    logger.info("Spatial join: listings -> neighbourhoods")
    try:
        joined = gpd.sjoin(gdf_listings, gdf_neigh, how='inner', predicate='within')
    except Exception:
        joined = gpd.sjoin(gdf_listings, gdf_neigh, how='inner', predicate='intersects')

    coverage = len(joined) / len(gdf_listings) * 100
    logger.info(f"Spatial join coverage: {coverage:.1f}% ({len(joined)}/{len(gdf_listings)})")
    return joined

def aggregate_to_neighbourhoods(gdf_listings, gdf_neigh):
    logger.info("Aggregating listings to neighbourhood level")

    neigh_col = 'neighbourhood'
    if 'neighbourhood_cleansed' in gdf_listings.columns:
        neigh_col = 'neighbourhood_cleansed'
    elif 'name' in gdf_neigh.columns:
        neigh_col_gdf = 'name'
    else:
        neigh_col_gdf = gdf_neigh.columns[0]

    joined = spatial_join_listings_neighbourhoods(gdf_listings, gdf_neigh)

    if neigh_col in joined.columns:
        group_col = neigh_col
    elif 'name_right' in joined.columns:
        group_col = 'name_right'
    else:
        group_col = [c for c in joined.columns if c.startswith('name')][-1]

    agg = joined.groupby(group_col).agg(
        listing_count=('listing_id', 'count'),
        median_price=('price', 'median'),
        mean_price=('price', 'mean'),
    ).reset_index()

    gdf_neigh_proj = gdf_neigh.to_crs(CRS_METRIC)
    gdf_neigh['area_km2'] = gdf_neigh_proj.geometry.area / 1e6

    neigh_name_col = gdf_neigh.columns[0]
    if 'name' in gdf_neigh.columns:
        neigh_name_col = 'name'
    elif 'neighbourhood' in gdf_neigh.columns:
        neigh_name_col = 'neighbourhood'

    result = gdf_neigh.merge(agg, left_on=neigh_name_col, right_on=group_col, how='left')
    result['listing_density'] = result['listing_count'] / result['area_km2']

    result = result.to_crs(CRS_WEB)
    logger.info(f"Aggregated to {len(result)} neighbourhoods")
    return result
