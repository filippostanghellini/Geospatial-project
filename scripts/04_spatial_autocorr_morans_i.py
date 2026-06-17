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

from src.config import OUTPUT_FILES, TABLES_DIR, FIGURES_DIR, CRS_METRIC, CITY_NAME
from src.prep import get_y_X
from src.io import save_csv

K_VALUES = [4, 8, 12]

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

    gdf_points = gpd.GeoDataFrame(
        model_df,
        geometry=gpd.points_from_xy(model_df['longitude'], model_df['latitude']),
        crs='EPSG:4326'
    )
    gdf_proj = gdf_points.to_crs(CRS_METRIC)

    print("\n--- Listing-level kNN Sensitivity (k=4,8,12) ---")
    sensitivity = []
    for k in K_VALUES:
        w_knn = KNN.from_dataframe(gdf_proj, k=k)
        w_knn.transform = 'r'
        nnz = w_knn.sparse.nnz
        moran_k = Moran(residuals, w_knn)
        sensitivity.append({
            'level': f'listing_knn{k}',
            'k': k,
            'moran_I': moran_k.I,
            'E_I': moran_k.EI,
            'p_value': moran_k.p_sim,
            'z_score': moran_k.z_sim,
            'nnz': nnz,
        })
        print(f"  k={k}: I={moran_k.I:.4f}  E(I)={moran_k.EI:+.4f}  p={moran_k.p_sim:.4e}  z={moran_k.z_sim:+.4f}  nnz={nnz}")

    moran_queen_I = np.nan
    moran_queen_p = np.nan
    moran_queen_EI = np.nan
    moran_queen_z = np.nan

    print("\n--- Neighbourhood-level Queen Contiguity ---")
    neigh_geojson = OUTPUT_FILES['neighbourhoods_enriched_geojson']
    if neigh_geojson.exists():
        gdf_neigh = gpd.read_file(neigh_geojson)

        neigh_col = 'neighbourhood_cleansed'
        if neigh_col not in model_df.columns:
            candidates = [c for c in model_df.columns if 'neigh' in c.lower() and not c.startswith('neigh_')]
            neigh_col = candidates[-1] if candidates else None

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
            moran_queen_I = moran_queen.I
            moran_queen_p = moran_queen.p_sim
            moran_queen_EI = moran_queen.EI
            moran_queen_z = moran_queen.z_sim
            print(f"  Moran's I: {moran_queen_I:.4f}")
            print(f"  E(I): {moran_queen_EI:.4f}")
            print(f"  p-value: {moran_queen_p:.4e}")
            print(f"  z-score: {moran_queen_z:.4f}")
        else:
            print("  No neighbourhood column found in model data")
    else:
        print("  Neighbourhood GeoJSON not found")

    results = sensitivity + [{
        'level': 'neighbourhood_queen',
        'k': None,
        'moran_I': moran_queen_I,
        'E_I': moran_queen_EI,
        'p_value': moran_queen_p,
        'z_score': moran_queen_z,
        'nnz': None,
    }]
    save_csv(pd.DataFrame(results), TABLES_DIR / 'morans_results.csv')

    fig, ax = plt.subplots(figsize=(8, 5))
    k_vals = [s['k'] for s in sensitivity]
    i_vals = [s['moran_I'] for s in sensitivity]
    ax.plot(k_vals, i_vals, 'o-', color='#3498db', linewidth=2, markersize=8)
    for k, iv in zip(k_vals, i_vals):
        ax.annotate(f'{iv:.4f}', (k, iv), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    ax.set_xlabel('k (nearest neighbours)', fontsize=12)
    ax.set_ylabel("Moran's I (OLS residuals)", fontsize=12)
    ax.set_title(f"kNN Sensitivity — Moran's I vs k ({CITY_NAME})", fontsize=14)
    ax.set_xticks(k_vals)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'morans_i_knn_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for s in sensitivity:
        print(f"listing_knn{s['k']}: I={s['moran_I']:.4f}, p={s['p_value']:.4e}")
    if neigh_geojson.exists() and neigh_col:
        print(f"Neighbourhood-level (Queen): I={moran_queen_I:.4f}, p={moran_queen_p:.4e}")
    print("=" * 60)

if __name__ == "__main__":
    main()
