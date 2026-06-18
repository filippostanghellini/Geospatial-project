import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import geopandas as gpd
from libpysal.weights import KNN
from esda.moran import Moran
from spreg import GM_Lag, GM_Error
import matplotlib.pyplot as plt

from src.config import OUTPUT_FILES, TABLES_DIR, FIGURES_DIR, CRS_METRIC, CITY_NAME
from src.prep import get_y_X
from src.io import save_csv

def main():
    print("=" * 60)
    print("SPATIAL MODELS: SAR & SEM")
    print("=" * 60)

    model_df = pd.read_parquet(OUTPUT_FILES['model_sample'])
    y, X, X_cols = get_y_X(model_df)
    n = len(y)

    print(f"\nModel sample: N={n}, K={len(X_cols)}")

    gdf_points = gpd.GeoDataFrame(
        model_df,
        geometry=gpd.points_from_xy(model_df['longitude'], model_df['latitude']),
        crs='EPSG:4326'
    )
    gdf_proj = gdf_points.to_crs(CRS_METRIC)

    w = KNN.from_dataframe(gdf_proj, k=8)
    w.transform = 'r'

    print("\n--- OLS (Model B completo) ---")
    X_const = sm.add_constant(X)
    model_ols = sm.OLS(y, X_const).fit(cov_type='HC1')
    print(f"  R-squared: {model_ols.rsquared:.4f}")
    print(f"  Adj R-squared: {model_ols.rsquared_adj:.4f}")

    print("\n--- SAR Model (Spatial Lag, GMM) ---")
    try:
        model_sar = GM_Lag(y, X, w=w, name_y='log_price', name_x=X_cols, robust='white')
        print(f"  Pseudo R-squared: {model_sar.pr2:.4f}")
        if hasattr(model_sar, 'rho'):
            rho_val = model_sar.rho
            if hasattr(rho_val, '__len__'):
                rho_val = rho_val.flatten()[0] if len(rho_val.shape) > 0 else rho_val
            print(f"  rho: {float(rho_val):.4f}")
        if hasattr(model_sar, 'logll'):
            print(f"  Log-likelihood: {model_sar.logll:.4f}")
    except Exception as e:
        print(f"  SAR estimation failed: {e}")
        model_sar = None

    print("\n--- SEM Model (Spatial Error, GMM) ---")
    try:
        model_sem = GM_Error(y, X, w=w, name_y='log_price', name_x=X_cols)
        print(f"  Pseudo R-squared: {model_sem.pr2:.4f}")
        lam_val = getattr(model_sem, 'lam', getattr(model_sem, 'lambda_', None))
        if lam_val is not None:
            if hasattr(lam_val, '__len__'):
                lam_val = lam_val.flatten()[0] if len(lam_val.shape) > 0 else lam_val
            print(f"  lambda: {float(lam_val):.4f}")
        if hasattr(model_sem, 'logll'):
            print(f"  Log-likelihood: {model_sem.logll:.4f}")
    except Exception as e:
        print(f"  SEM estimation failed: {e}")
        model_sem = None

    print("\n--- Post-fit Moran's I on Residuals ---")

    moran_ols = Moran(model_ols.resid, w)
    print(f"  OLS residuals: I={moran_ols.I:.4f}, p={moran_ols.p_sim:.4e}")

    if model_sar:
        sar_resid = model_sar.u.flatten()
        moran_sar = Moran(sar_resid, w)
        print(f"  SAR residuals: I={moran_sar.I:.4f}, p={moran_sar.p_sim:.4e}")
    else:
        moran_sar = None

    if model_sem:
        sem_resid = model_sem.u.flatten()
        moran_sem = Moran(sem_resid, w)
        print(f"  SEM residuals: I={moran_sem.I:.4f}, p={moran_sem.p_sim:.4e}")
    else:
        moran_sem = None

    comparison = {
        'model': ['OLS', 'SAR', 'SEM'],
        'r_squared': [model_ols.rsquared, model_sar.pr2 if model_sar else np.nan, model_sem.pr2 if model_sem else np.nan],
        'moran_I': [moran_ols.I, moran_sar.I if moran_sar else np.nan, moran_sem.I if moran_sem else np.nan],
        'moran_p': [moran_ols.p_sim, moran_sar.p_sim if moran_sar else np.nan, moran_sem.p_sim if moran_sem else np.nan],
    }
    save_csv(pd.DataFrame(comparison), TABLES_DIR / 'model_comparison.csv')

    residuals_df = pd.DataFrame({
        'listing_id': model_df['listing_id'].values,
        'ols_residual': model_ols.resid,
        'sar_residual': model_sar.u.flatten() if model_sar else np.nan,
        'sem_residual': model_sem.u.flatten() if model_sem else np.nan,
    })
    save_csv(residuals_df, OUTPUT_FILES['residuals_for_map'])
    print(f"Residuals saved: {len(residuals_df)} listings")

    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['OLS', 'SAR', 'SEM']
    moran_vals = [moran_ols.I, moran_sar.I if moran_sar else 0, moran_sem.I if moran_sem else 0]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(models, moran_vals, color=colors, edgecolor='black')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f"Post-fit Moran's I by Model ({CITY_NAME})", fontsize=14)
    ax.set_ylabel("Moran's I")
    for bar, val in zip(bars, moran_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.4f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'model_comparison_morans_i.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"OLS: R2={model_ols.rsquared:.4f}, Moran's I={moran_ols.I:.4f}")
    if model_sar:
        rho_val = model_sar.rho
        if hasattr(rho_val, '__len__'):
            rho_val = rho_val.flatten()[0]
        print(f"SAR: PR2={model_sar.pr2:.4f}, rho={float(rho_val):.4f}, Moran's I={moran_sar.I:.4f}")
    if model_sem:
        lam_val = getattr(model_sem, 'lam', getattr(model_sem, 'lambda_', None))
        if lam_val is not None:
            if hasattr(lam_val, '__len__'):
                lam_val = lam_val.flatten()[0]
            lam_str = f"{float(lam_val):.4f}"
        else:
            lam_str = "N/A"
        print(f"SEM: PR2={model_sem.pr2:.4f}, lambda={lam_str}, Moran's I={moran_sem.I:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
