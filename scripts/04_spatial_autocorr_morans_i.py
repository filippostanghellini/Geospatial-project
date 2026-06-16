import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import geopandas as gpd
from libpysal.weights import KNN, Queen
from esda.moran import Moran
import matplotlib.pyplot as plt

from src.config import OUTPUT_FILES, TABLES_DIR, CRS_METRIC, CITY_NAME
from src.prep import build_model_df, get_y_X
from src.io import save_csv

def main():
    print("=" * 60)
    print("SPATIAL AUTOCORRELATION - MORAN'S I")
    print("=" * 60)

    model_df = pd.read_parquet(OUTPUT_FILES['model_sample'])
    y, X, X_cols = get_y_X(model_df)
    X_const = sm.add_constant(X)
    model_b = sm.OLS(y, X_const).fit(cov_type='HC1')
    residuals = model_b.resid

    print(f"\nModel sample: N={len(y)}")
    print(f"OLS residuals: mean={residuals.mean():.6f}, std={residuals.std():.4f}")

    print("\n--- Listing-level kNN Weights (k=8) ---")
    coords = model_df[['latitude', 'longitude']].values
    gdf_points = gpd.GeoDataFrame(
        model_df,
        geometry=gpd.points_from_xy(model_df['longitude'], model_df['latitude']),
        crs='EPSG:4326'
    )
    gdf_proj = gdf_points.to_crs(CRS_METRIC)

    w_knn = KNN.from_dataframe(gdf_proj, k=8)
    w_knn.transform = 'r'
    nnz = w_knn.sparse.nnz
    print(f"  Weights: N={w_knn.n}, nnz={nnz}, density={nnz/w_knn.n**2:.4f}")

    moran_knn = Moran(residuals, w_knn)
    print(f"  Moran's I: {moran_knn.I:.4f}")
    print(f"  E(I): {moran_knn.EI:.4f}")
    print(f"  p-value: {moran_knn.p_sim:.4e}")
    print(f"  z-score: {moran_knn.z_sim:.4f}")

    print("\n--- Neighbourhood-level Queen Contiguity ---")
    neigh_geojson = OUTPUT_FILES['neighbourhoods_enriched_geojson']
    if neigh_geojson.exists():
        gdf_neigh = gpd.read_file(neigh_geojson)

        neigh_col = 'neighbourhood_cleansed'
        if neigh_col not in model_df.columns:
            neigh_col = [c for c in model_df.columns if 'neigh' in c.lower() and not c.startswith('neigh_')][-1] if any('neigh' in c.lower() for c in model_df.columns) else None

        if neigh_col:
            model_df_copy = model_df.copy()
            model_df_copy['residual'] = residuals

            neigh_agg = model_df_copy.groupby(neigh_col)['residual'].mean().reset_index()
            neigh_agg.columns = ['neighbourhood', 'mean_residual']

            gdf_neigh_merged = gdf_neigh.merge(neigh_agg, left_on=gdf_neigh.columns[0], right_on='neighbourhood', how='inner')

            w_queen = Queen.from_dataframe(gdf_neigh_merged, use_index=False)
            w_queen.transform = 'r'
            print(f"  Weights: N={w_queen.n}, nnz={w_queen.sparse.nnz}")

            moran_queen = Moran(gdf_neigh_merged['mean_residual'].values, w_queen)
            print(f"  Moran's I: {moran_queen.I:.4f}")
            print(f"  E(I): {moran_queen.EI:.4f}")
            print(f"  p-value: {moran_queen.p_sim:.4e}")
            print(f"  z-score: {moran_queen.z_sim:.4f}")

    results = pd.DataFrame({
        'level': ['listing_knn8', 'neighbourhood_queen'],
        'moran_I': [moran_knn.I, moran_queen.I if neigh_geojson.exists() and neigh_col else np.nan],
        'E_I': [moran_knn.EI, moran_queen.EI if neigh_geojson.exists() and neigh_col else np.nan],
        'p_value': [moran_knn.p_sim, moran_queen.p_sim if neigh_geojson.exists() and neigh_col else np.nan],
        'z_score': [moran_knn.z_sim, moran_queen.z_sim if neigh_geojson.exists() and neigh_col else np.nan],
    })
    save_csv(results, TABLES_DIR / 'morans_results.csv')

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Listing-level (kNN k=8): I={moran_knn.I:.4f}, p={moran_knn.p_sim:.4e}")
    if neigh_geojson.exists() and neigh_col:
        print(f"Neighbourhood-level (Queen): I={moran_queen.I:.4f}, p={moran_queen.p_sim:.4e}")
    print("=" * 60)

if __name__ == "__main__":
    main()
