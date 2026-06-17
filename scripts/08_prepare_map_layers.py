import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from src.config import OUTPUT_FILES, MAP_SAMPLE_SIZE, RANDOM_SEED
from src.io import save_geojson

def main():
    print("=" * 60)
    print("PREPARE MAP LAYERS")
    print("=" * 60)

    model_df = pd.read_parquet(OUTPUT_FILES['model_sample'])
    residuals_df = pd.read_csv(OUTPUT_FILES['residuals_for_map'])

    merged = model_df.merge(residuals_df, on='listing_id', how='inner')
    print(f"Merged: {len(merged)} listings")

    gdf = gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged['longitude'], merged['latitude']),
        crs='EPSG:4326'
    )

    sample = gdf.sample(n=min(MAP_SAMPLE_SIZE, len(gdf)), random_state=RANDOM_SEED)
    save_geojson(sample, OUTPUT_FILES['map_points_sample'])
    print(f"Point sample: {len(sample)} listings")

    bounds = gdf.total_bounds
    cell_size = 0.02
    x_min, y_min, x_max, y_max = bounds

    cells = []
    x = x_min
    while x < x_max:
        y = y_min
        while y < y_max:
            cells.append(box(x, y, x + cell_size, y + cell_size))
            y += cell_size
        x += cell_size

    grid = gpd.GeoDataFrame(geometry=cells, crs='EPSG:4326')

    grid_agg = gpd.sjoin(gdf, grid, how='inner', predicate='within')
    grid_agg = grid_agg.groupby('index_right').agg(
        mean_ols_residual=('ols_residual', 'mean'),
        mean_sar_residual=('sar_residual', 'mean'),
        mean_sem_residual=('sem_residual', 'mean'),
        listing_count=('listing_id', 'count'),
        mean_price=('price', 'mean'),
    ).reset_index()

    grid = grid.reset_index().rename(columns={'index': 'cell_id'})
    grid_final = grid.merge(grid_agg, left_on='cell_id', right_on='index_right', how='left')
    grid_final = grid_final.drop(columns=['index_right'], errors='ignore')

    save_geojson(grid_final, OUTPUT_FILES['map_grid_cells'])
    print(f"Grid cells: {len(grid_final)} ({grid_final['listing_count'].notna().sum()} with data)")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
