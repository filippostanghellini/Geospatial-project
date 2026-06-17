import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import statsmodels.api as sm
import geopandas as gpd
from libpysal.weights import KNN
from spreg import GM_Lag, GM_Error

from src.config import OUTPUT_FILES, CRS_METRIC
from src.prep import get_y_X
from src.io import save_csv

def main():
    print("=" * 60)
    print("EXTRACT RESIDUALS FOR MAP")
    print("=" * 60)

    model_df = pd.read_parquet(OUTPUT_FILES['model_sample'])
    y, X, X_cols = get_y_X(model_df)

    gdf_points = gpd.GeoDataFrame(
        model_df,
        geometry=gpd.points_from_xy(model_df['longitude'], model_df['latitude']),
        crs='EPSG:4326'
    )
    gdf_proj = gdf_points.to_crs(CRS_METRIC)

    w = KNN.from_dataframe(gdf_proj, k=8)
    w.transform = 'r'

    structural_cols = [c for c in X_cols if not c.startswith('neigh_')]
    X_struct_idx = [X_cols.index(c) for c in structural_cols]
    X_struct = X[:, X_struct_idx]

    X_const = sm.add_constant(X_struct)
    model_ols = sm.OLS(y, X_const).fit(cov_type='HC1')
    ols_resid = model_ols.resid

    model_sar = GM_Lag(y, X_struct, w=w, name_y='log_price', name_x=structural_cols, robust='white')
    sar_resid = model_sar.u.flatten()

    model_sem = GM_Error(y, X_struct, w=w, name_y='log_price', name_x=structural_cols)
    sem_resid = model_sem.u.flatten()

    residuals_df = pd.DataFrame({
        'listing_id': model_df['listing_id'].values,
        'ols_residual': ols_resid,
        'sar_residual': sar_resid,
        'sem_residual': sem_resid,
    })
    save_csv(residuals_df, OUTPUT_FILES['residuals_for_map'])
    print(f"Residuals saved: {len(residuals_df)} listings")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
