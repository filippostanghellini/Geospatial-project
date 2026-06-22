import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import KNN
from spreg import OLS as SpregOLS

from src.config import OUTPUT_FILES, TABLES_DIR, CRS_METRIC
from src.prep import get_y_X
from src.io import save_csv

def main():
    print("=" * 60)
    print("LM DIAGNOSTIC TESTS")
    print("=" * 60)

    model_df = pd.read_parquet(OUTPUT_FILES['model_sample'])
    y, X, X_cols = get_y_X(model_df)
    n = len(y)
    k = X.shape[1]

    print(f"\nModel: N={n}, K={k}")

    gdf_points = gpd.GeoDataFrame(
        model_df,
        geometry=gpd.points_from_xy(model_df['longitude'], model_df['latitude']),
        crs='EPSG:4326'
    )
    gdf_proj = gdf_points.to_crs(CRS_METRIC)

    w = KNN.from_dataframe(gdf_proj, k=8)
    w.transform = 'r'

    print(f"Weights: N={w.n}, k=8, nnz={w.sparse.nnz}")

    # spreg.OLS computes LM diagnostics using the correct ABFY (1996) trace
    # T = tr((W'+W)W) = tr(W'W) + tr(W^2), which the previous manual
    # implementation omitted (it used only tr(W'W)), inflating 3/4 statistics
    # and flipping the SAR-vs-SEM selection.
    # spreg exposes the diagnostics as (statistic, p-value) tuples.
    model_spreg = SpregOLS(y, X, w=w, spat_diag=True, moran=True,
                           name_y='log_price', name_x=X_cols)

    LM_error, p_lm_error = float(model_spreg.lm_error[0]), float(model_spreg.lm_error[1])
    LM_lag, p_lm_lag = float(model_spreg.lm_lag[0]), float(model_spreg.lm_lag[1])
    RLM_error, p_rlm_error = float(model_spreg.rlm_error[0]), float(model_spreg.rlm_error[1])
    RLM_lag, p_rlm_lag = float(model_spreg.rlm_lag[0]), float(model_spreg.rlm_lag[1])

    print("\n--- LM Test Results (spreg, correct trace T = tr(W'W) + tr(W^2)) ---")
    print(f"  LM-error:       {LM_error:10.4f}  p={p_lm_error:.4e}")
    print(f"  LM-lag:         {LM_lag:10.4f}  p={p_lm_lag:.4e}")
    print(f"  Robust LM-error:{RLM_error:10.4f}  p={p_rlm_error:.4e}")
    print(f"  Robust LM-lag:  {RLM_lag:10.4f}  p={p_rlm_lag:.4e}")

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
