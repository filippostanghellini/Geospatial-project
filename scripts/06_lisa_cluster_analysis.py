import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import geopandas as gpd
from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.lines import Line2D

from src.config import OUTPUT_FILES, TABLES_DIR, FIGURES_DIR, MAPS_DIR, CRS_METRIC, CITY_NAME
from src.prep import get_y_X
from src.io import save_csv, save_geojson

QUADRANT_LABELS = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
QUADRANT_COLORS = {1: '#e74c3c', 2: '#3498db', 3: '#2ecc71', 4: '#f39c12'}
QUADRANT_NAMES = {1: 'High-High', 2: 'Low-High', 3: 'Low-Low', 4: 'High-Low'}

def main():
    print("=" * 60)
    print("LISA — LOCAL INDICATORS OF SPATIAL ASSOCIATION")
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

    w = KNN.from_dataframe(gdf_proj, k=8)
    w.transform = 'r'

    print(f"\n  Weights: N={w.n}, k=8, nnz={w.sparse.nnz}")

    print("\n--- Global Moran's I ---")
    moran_global = Moran(residuals, w)
    print(f"  I = {moran_global.I:.4f}")
    print(f"  E(I) = {moran_global.EI:.4f}")
    print(f"  z-score = {moran_global.z_sim:+.4f}")
    print(f"  p-value = {moran_global.p_sim:.4e}")

    print("\n--- Local Moran's I (LISA) ---")
    lisa = Moran_Local(residuals, w, permutations=999)

    local_I = lisa.Is
    p_values = lisa.p_sim
    q = lisa.q
    z_scores = lisa.z_sim

    n_significant = (p_values < 0.05).sum()
    quadrant_counts = pd.Series(q).value_counts().sort_index()
    print(f"  Significant local clusters (p<0.05): {n_significant}/{len(q)} ({100*n_significant/len(q):.1f}%)")
    for qi in sorted(quadrant_counts.index):
        if qi in QUADRANT_LABELS:
            print(f"    {QUADRANT_NAMES[qi]}: {quadrant_counts[qi]} ({100*quadrant_counts[qi]/len(q):.1f}%)")

    # Moran scatterplot
    z = (residuals - residuals.mean()) / residuals.std()
    wz = w.sparse @ z

    fig, ax = plt.subplots(figsize=(9, 8))

    ns_mask = p_values >= 0.05
    sig_mask = p_values < 0.05

    ax.scatter(z[ns_mask], wz[ns_mask], c='lightgray', alpha=0.15, s=4, label='Not significant (p≥0.05)', zorder=1)

    for qi, color in QUADRANT_COLORS.items():
        q_mask = (q == qi) & sig_mask
        if q_mask.sum() > 0:
            ax.scatter(z[q_mask], wz[q_mask], c=color, alpha=0.4, s=6,
                       label=f'{QUADRANT_NAMES[qi]} ({QUADRANT_LABELS[qi]})', zorder=2)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)

    slope = moran_global.I
    x_range = np.linspace(z.min(), z.max(), 100)
    ax.plot(x_range, slope * x_range, '-', color='#8e44ad', linewidth=2, alpha=0.8,
            label=f"Moran's I = {slope:.4f}")

    ax.set_xlabel('Standardized Residual (z)', fontsize=12)
    ax.set_ylabel('Spatial Lag (Wz)', fontsize=12)
    ax.set_title(f'Moran Scatterplot — OLS Residuals ({CITY_NAME}, kNN k=8)', fontsize=14)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8, markerscale=2,
              framealpha=0.9)

    ax.set_xlim(z.min() - 0.5, z.max() + 0.5)
    ax.set_ylim(wz.min() - 0.5, wz.max() + 0.5)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'moran_scatterplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Moran scatterplot saved: {FIGURES_DIR / 'moran_scatterplot.png'}")

    # LISA cluster map — everything in EPSG:3857 for correct basemap alignment
    gdf_temp = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(model_df['longitude'], model_df['latitude']),
        crs='EPSG:4326'
    )
    gdf_wm = gdf_temp.to_crs(epsg=3857)
    x_wm = gdf_wm.geometry.x.values
    y_wm = gdf_wm.geometry.y.values

    fig, ax = plt.subplots(figsize=(14, 12))

    ax.scatter(x_wm[ns_mask], y_wm[ns_mask],
              c='lightgray', alpha=0.2, s=1, zorder=1)

    for qi, color in QUADRANT_COLORS.items():
        q_mask = (q == qi) & sig_mask
        if q_mask.sum() > 0:
            ax.scatter(x_wm[q_mask], y_wm[q_mask],
                      c=color, alpha=0.5, s=3,
                      label=f'{QUADRANT_NAMES[qi]} ({QUADRANT_LABELS[qi]})', zorder=2)

    ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.CartoDB.Positron, alpha=0.3)

    ax.set_axis_off()
    ax.set_title(f'LISA Cluster Map — OLS Residuals ({CITY_NAME}, kNN k=8)', fontsize=14)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=8, label='Not significant (p≥0.05)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='High-High (HH)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='Low-High (LH)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=8, label='Low-Low (LL)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f39c12', markersize=8, label='High-Low (HL)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, markerscale=1.5, framealpha=0.9)

    plt.savefig(FIGURES_DIR / 'lisa_cluster_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  LISA cluster map saved: {FIGURES_DIR / 'lisa_cluster_map.png'}")

    # Z-score significance histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(z_scores, bins=80, density=True, color='#3498db', alpha=0.7, edgecolor='white')
    ax.axvline(x=1.96, color='#e74c3c', linestyle='--', alpha=0.7, label='|z| = 1.96')
    ax.axvline(x=-1.96, color='#e74c3c', linestyle='--', alpha=0.7)
    ax.set_xlabel('Local Moran z-score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'LISA z-score Distribution ({CITY_NAME})', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'lisa_zscore_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save LISA results
    cluster_label = [QUADRANT_LABELS.get(qi, 'NS') if p < 0.05 else 'NS' for qi, p in zip(q, p_values)]

    lisa_df = pd.DataFrame({
        'listing_id': model_df['listing_id'].values,
        'residual': residuals,
        'local_I': local_I,
        'z_score': z_scores,
        'p_value': p_values,
        'quadrant': q,
        'cluster': cluster_label,
        'latitude': model_df['latitude'].values,
        'longitude': model_df['longitude'].values,
    })
    save_csv(lisa_df, TABLES_DIR / 'lisa_results.csv')
    print(f"  LISA results saved: {TABLES_DIR / 'lisa_results.csv'}")

    gdf_lisa = gpd.GeoDataFrame(
        lisa_df, geometry=gpd.points_from_xy(lisa_df['longitude'], lisa_df['latitude']),
        crs='EPSG:4326'
    )
    save_geojson(gdf_lisa, MAPS_DIR / 'lisa_clusters.geojson')
    print(f"  LISA GeoJSON saved: {MAPS_DIR / 'lisa_clusters.geojson'}")

    # Summary table
    cluster_label_arr = np.array(cluster_label)
    summary_rows = [
        {'statistic': 'Global I', 'value': moran_global.I},
        {'statistic': 'Num significant clusters (p<0.05)', 'value': n_significant},
        {'statistic': 'HH count', 'value': int((cluster_label_arr == 'HH').sum())},
        {'statistic': 'LL count', 'value': int((cluster_label_arr == 'LL').sum())},
        {'statistic': 'LH count', 'value': int((cluster_label_arr == 'LH').sum())},
        {'statistic': 'HL count', 'value': int((cluster_label_arr == 'HL').sum())},
        {'statistic': 'Mean local I', 'value': float(local_I.mean())},
        {'statistic': 'Std local I', 'value': float(local_I.std())},
    ]
    save_csv(pd.DataFrame(summary_rows), TABLES_DIR / 'lisa_summary.csv')

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for row in summary_rows:
        val = row['value']
        if isinstance(val, float):
            print(f"  {row['statistic']:<35s}: {val:.4f}")
        else:
            print(f"  {row['statistic']:<35s}: {val}")
    print("=" * 60)

if __name__ == "__main__":
    main()
