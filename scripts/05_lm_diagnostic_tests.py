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
    eWe = e.T @ We
    ee = e.T @ e
    sigma2 = ee / n

    XtX_inv = np.linalg.inv(X_const.T @ X_const)
    WX = W @ X_const
    WXb = WX - X_const @ (XtX_inv @ (X_const.T @ WX))

    tr_WtW = (W.T @ W).diagonal().sum()
    tr_W2 = (W @ W).diagonal().sum()
    T = tr_WtW + tr_W2

    J = (WXb.T @ WXb) / sigma2

    LM_lag = (eWe / sigma2) ** 2 / T
    LM_error = (eWe / sigma2) ** 2 / T - (eWe / sigma2) ** 2 / (n * T / (n - k))

    RLM_lag = LM_lag - LM_error
    RLM_error = LM_error

    p_lm_lag = 1 - stats.chi2.cdf(LM_lag, 1)
    p_lm_error = 1 - stats.chi2.cdf(LM_error, 1)
    p_rlm_lag = 1 - stats.chi2.cdf(RLM_lag, 1)
    p_rlm_error = 1 - stats.chi2.cdf(RLM_error, 1)

    print("\n--- LM Test Results ---")
    print(f"  LM-lag:        {LM_lag:10.4f}  p={p_lm_lag:.4e}")
    print(f"  LM-error:      {LM_error:10.4f}  p={p_lm_error:.4e}")
    print(f"  Robust LM-lag: {RLM_lag:10.4f}  p={p_rlm_lag:.4e}")
    print(f"  Robust LM-error:{RLM_error:10.4f}  p={p_rlm_error:.4e}")

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
        'test': ['LM-lag', 'LM-error', 'Robust LM-lag', 'Robust LM-error'],
        'statistic': [LM_lag, LM_error, RLM_lag, RLM_error],
        'p_value': [p_lm_lag, p_lm_error, p_rlm_lag, p_rlm_error],
    })
    save_csv(results, TABLES_DIR / 'lm_diagnostic_tests.csv')

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
