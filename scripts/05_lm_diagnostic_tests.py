import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import geopandas as gpd
from libpysal.weights import KNN
from scipy import stats

from src.config import OUTPUT_FILES, TABLES_DIR, CRS_METRIC
from src.prep import get_y_X
from src.io import save_csv

def main():
    print("=" * 60)
    print("LM DIAGNOSTIC TESTS")
    print("=" * 60)

    model_df = pd.read_parquet(OUTPUT_FILES['model_sample'])
    y, X, X_cols = get_y_X(model_df)
    X_const = sm.add_constant(X)
    model_ols = sm.OLS(y, X_const).fit()
    e = model_ols.resid
    beta = model_ols.params
    n = len(y)
    k = X_const.shape[1]

    print(f"\nModel: N={n}, K={k}")

    gdf_points = gpd.GeoDataFrame(
        model_df,
        geometry=gpd.points_from_xy(model_df['longitude'], model_df['latitude']),
        crs='EPSG:4326'
    )
    gdf_proj = gdf_points.to_crs(CRS_METRIC)

    w = KNN.from_dataframe(gdf_proj, k=8)
    w.transform = 'r'
    W = w.sparse

    print(f"Weights: N={w.n}, k=8, nnz={w.sparse.nnz}")

    We = W @ e
    eWe = float(e.T @ We)
    Wy = W @ y
    eWy = float(e.T @ Wy)
    ee = float(e.T @ e)
    sigma2 = ee / n

    XtX_inv = np.linalg.inv(X_const.T @ X_const)

    Xb = X_const @ beta
    WXb = W @ Xb
    beta_aux = XtX_inv @ (X_const.T @ WXb)
    WXb_residuals = WXb - X_const @ beta_aux
    J = float(WXb_residuals.T @ WXb_residuals / sigma2)

    T = float((W.T @ W + W @ W.T).diagonal().sum() / 2)

    LM_error = (eWe / sigma2) ** 2 / T
    LM_lag = (eWy / sigma2) ** 2 / (T + J)

    num_rlm_error = (eWe - (T / (T + J)) * eWy) / sigma2
    den_rlm_error = max(T - T**2 / (T + J), 1e-10)
    RLM_error = num_rlm_error ** 2 / den_rlm_error

    num_rlm_lag = (eWy - eWe) / sigma2
    den_rlm_lag = max(J, 1e-10)
    RLM_lag = num_rlm_lag ** 2 / den_rlm_lag

    p_lm_error = 1 - stats.chi2.cdf(LM_error, 1)
    p_lm_lag = 1 - stats.chi2.cdf(LM_lag, 1)
    p_rlm_error = 1 - stats.chi2.cdf(RLM_error, 1)
    p_rlm_lag = 1 - stats.chi2.cdf(RLM_lag, 1)

    print("\n--- LM Test Results ---")
    print(f"  LM-error:      {LM_error:10.4f}  p={p_lm_error:.4e}")
    print(f"  LM-lag:        {LM_lag:10.4f}  p={p_lm_lag:.4e}")
    print(f"  Robust LM-error:{RLM_error:10.4f}  p={p_rlm_error:.4e}")
    print(f"  Robust LM-lag: {RLM_lag:10.4f}  p={p_rlm_lag:.4e}")

    print("\n--- Decision Rule (Anselin 1988) ---")
    if p_rlm_lag < 0.05 and p_rlm_error < 0.05:
        if RLM_lag > RLM_error:
            print("  Both robust tests significant. RLM-lag > RLM-error -> SAR preferred")
        else:
            print("  Both robust tests significant. RLM-error > RLM-lag -> SEM preferred")
    elif p_rlm_lag < 0.05:
        print("  Only RLM-lag significant -> SAR model")
    elif p_rlm_error < 0.05:
        print("  Only RLM-error significant -> SEM model")
    else:
        print("  Neither robust test significant -> OLS may be adequate")

    results = pd.DataFrame({
        'test': ['LM-error', 'LM-lag', 'Robust LM-error', 'Robust LM-lag'],
        'statistic': [LM_error, LM_lag, RLM_error, RLM_lag],
        'p_value': [p_lm_error, p_lm_lag, p_rlm_error, p_rlm_lag],
    })
    save_csv(results, TABLES_DIR / 'lm_diagnostic_tests.csv')

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
