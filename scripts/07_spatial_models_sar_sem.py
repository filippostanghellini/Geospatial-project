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
from spreg import GM_Lag, GM_Error_Het
import matplotlib.pyplot as plt

from src.config import OUTPUT_FILES, TABLES_DIR, FIGURES_DIR, CRS_METRIC, CITY_NAME
from src.prep import get_y_X
from src.io import save_csv

def _extract_rho(model):
    """Robustly extract rho from a spreg model."""
    if hasattr(model, 'rho') and model.rho is not None:
        arr = np.asarray(model.rho).flatten()
        return float(arr[0]) if arr.size else float('nan')
    betas = np.asarray(model.betas).flatten()
    return float(betas[-1])

def _extract_lambda(model):
    """Extract lambda from a GM_Error(_Het) model.

    GM_Error/GM_Error_Het do NOT expose `lam`/`lambda_` attributes; lambda is
    stored as the last entry of `betas`. The previous version of this script
    used getattr(model_sem, 'lam', getattr(..., 'lambda_', None)) which always
    returned None and printed 'lambda: N/A', and the reported lambda=0.913 was
    not reproducible from the code.
    """
    betas = np.asarray(model.betas).flatten()
    return float(betas[-1])

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

    print("\n--- SAR Model (Spatial Lag, GMM, heteroskedastic-robust) ---")
    try:
        model_sar = GM_Lag(y, X, w=w, name_y='log_price', name_x=X_cols, robust='white')
        rho_val = _extract_rho(model_sar)
        print(f"  Pseudo R-squared: {model_sar.pr2:.4f}")
        print(f"  rho: {rho_val:.4f}")
    except Exception as e:
        print(f"  SAR estimation failed: {e}")
        model_sar = None
        rho_val = float('nan')

    print("\n--- SEM Model (Spatial Error, GMM, heteroskedastic-robust) ---")
    try:
        # GM_Error_Het for heteroskedastic-robust GMM, consistent with SAR's robust='white'.
        # The previous version used GM_Error (homoskedastic), an asymmetric choice.
        model_sem = GM_Error_Het(y, X, w=w, name_y='log_price', name_x=X_cols)
        lam_val = _extract_lambda(model_sem)
        print(f"  Pseudo R-squared: {model_sem.pr2:.4f}")
        print(f"  lambda: {lam_val:.4f}")
    except Exception as e:
        print(f"  SEM estimation failed: {e}")
        model_sem = None
        lam_val = float('nan')

    print("\n--- Post-fit Moran's I on Residuals ---")

    moran_ols = Moran(model_ols.resid, w)
    print(f"  OLS residuals: I={moran_ols.I:.4f}, p={moran_ols.p_sim:.4e}")

    if model_sar:
        # For SAR, u = y - rho*Wy - X*beta IS the clean innovation epsilon,
        # so Moran(u) is the correct test for residual autocorrelation.
        sar_resid = model_sar.u.flatten()
        moran_sar = Moran(sar_resid, w)
        print(f"  SAR residuals (u): I={moran_sar.I:.4f}, p={moran_sar.p_sim:.4e}")
    else:
        sar_resid = np.full(n, np.nan)
        moran_sar = None

    if model_sem:
        # For SEM, u = lambda*W*u + epsilon, so the raw residual u is
        # autocorrelated BY CONSTRUCTION. The correct residual to test is
        # e_filtered = (I - lambda*W)*u, which recovers the innovation epsilon.
        # The previous version computed Moran(model_sem.u), which is always
        # autocorrelated by construction and created the false 'SEM paradox'
        # (SEM apparently not removing autocorrelation). On e_filtered, SEM
        # correctly removes the spatial autocorrelation.
        sem_resid_raw = model_sem.u.flatten()
        sem_resid_filtered = model_sem.e_filtered.flatten()
        moran_sem_raw = Moran(sem_resid_raw, w)
        moran_sem = Moran(sem_resid_filtered, w)
        print(f"  SEM raw residual (u):          I={moran_sem_raw.I:.4f}, p={moran_sem_raw.p_sim:.4e}  [autocorrelated by construction]")
        print(f"  SEM filtered residual (eps):   I={moran_sem.I:.4f}, p={moran_sem.p_sim:.4e}  [CORRECT test]")
    else:
        sem_resid_filtered = np.full(n, np.nan)
        moran_sem = None

    comparison = {
        'model': ['OLS', 'SAR', 'SEM'],
        'r_squared': [model_ols.rsquared,
                      model_sar.pr2 if model_sar else np.nan,
                      model_sem.pr2 if model_sem else np.nan],
        'rho': [np.nan, rho_val, np.nan],
        'lambda': [np.nan, np.nan, lam_val],
        'moran_I': [moran_ols.I,
                    moran_sar.I if moran_sar else np.nan,
                    moran_sem.I if moran_sem else np.nan],
        'moran_p': [moran_ols.p_sim,
                    moran_sar.p_sim if moran_sar else np.nan,
                    moran_sem.p_sim if moran_sem else np.nan],
    }
    save_csv(pd.DataFrame(comparison), TABLES_DIR / 'model_comparison.csv')

    residuals_df = pd.DataFrame({
        'listing_id': model_df['listing_id'].values,
        'ols_residual': model_ols.resid,
        'sar_residual': sar_resid,
        # Save the FILTERED residual for SEM: this is the spatially clean
        # innovation, the meaningful residual for mapping and diagnostics.
        'sem_residual': sem_resid_filtered,
    })
    save_csv(residuals_df, OUTPUT_FILES['residuals_for_map'])
    print(f"Residuals saved: {len(residuals_df)} listings")

    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['OLS', 'SAR', 'SEM']
    moran_vals = [moran_ols.I,
                  moran_sar.I if moran_sar else 0,
                  moran_sem.I if moran_sem else 0]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(models, moran_vals, color=colors, edgecolor='black')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f"Post-fit Moran's I by Model ({CITY_NAME})", fontsize=14)
    ax.set_ylabel("Moran's I (on correct residual)")
    for bar, val in zip(bars, moran_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'model_comparison_morans_i.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"OLS: R2={model_ols.rsquared:.4f}, Moran's I={moran_ols.I:.4f}")
    if model_sar:
        print(f"SAR: PR2={model_sar.pr2:.4f}, rho={rho_val:.4f}, Moran's I={moran_sar.I:.4f}")
    if model_sem:
        print(f"SEM: PR2={model_sem.pr2:.4f}, lambda={lam_val:.4f}, "
              f"Moran's I (filtered)={moran_sem.I:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
