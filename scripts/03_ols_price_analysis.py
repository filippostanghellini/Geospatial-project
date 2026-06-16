import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

from src.config import OUTPUT_FILES, FIGURES_DIR, TABLES_DIR, CITY_NAME
from src.prep import build_model_df, get_y_X
from src.io import save_csv

def main():
    print("=" * 60)
    print("OLS PRICE ANALYSIS")
    print("=" * 60)

    listings_path = OUTPUT_FILES['listings_clean']
    if not listings_path.exists():
        print("ERROR: listings_clean.parquet not found. Run the pipeline notebook first.")
        return

    df = pd.read_parquet(listings_path)
    model_df = build_model_df(df, out_path=OUTPUT_FILES['model_sample'])

    y, X, X_cols = get_y_X(model_df)
    print(f"\nModel sample: N={len(y)}, K={len(X_cols)}")

    X_const = sm.add_constant(X)
    X_const_cols = ['const'] + X_cols

    print("\n--- Model A: Property + Host + Room Type ---")
    structural_cols = [c for c in X_cols if not c.startswith('neigh_')]
    X_a_idx = [X_cols.index(c) for c in structural_cols]
    X_a = sm.add_constant(X[:, X_a_idx])

    model_a = sm.OLS(y, X_a).fit(cov_type='HC1')
    print(f"  R-squared: {model_a.rsquared:.4f}")
    print(f"  Adj R-squared: {model_a.rsquared_adj:.4f}")
    print(f"  F-statistic: {model_a.fvalue:.2f} (p={model_a.f_pvalue:.4e})")

    print("\n--- Model B: Model A + Location (dist_cbd + neighbourhood FE) ---")
    model_b = sm.OLS(y, X_const).fit(cov_type='HC1')
    print(f"  R-squared: {model_b.rsquared:.4f}")
    print(f"  Adj R-squared: {model_b.rsquared_adj:.4f}")
    print(f"  F-statistic: {model_b.fvalue:.2f} (p={model_b.f_pvalue:.4e})")

    print("\n--- Key Coefficients (Model B) ---")
    key_vars = ['const', 'accommodates', 'bedrooms', 'beds', 'bathrooms',
                'minimum_nights', 'host_is_superhost', 'host_listings_count',
                'number_of_reviews', 'review_scores_rating', 'instant_bookable',
                'dist_cbd_km']
    for var in key_vars:
        if var in X_const_cols:
            idx = X_const_cols.index(var)
            coef = model_b.params[idx]
            se = model_b.bse[idx]
            pval = model_b.pvalues[idx]
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            print(f"  {var:30s}  coef={coef:8.4f}  se={se:8.4f}  p={pval:.4e} {sig}")

    coef_df = pd.DataFrame({
        'variable': [f'x{i}' for i in range(len(model_b.params))],
        'coefficient': model_b.params,
        'std_error': model_b.bse,
        'p_value': model_b.pvalues,
        't_stat': model_b.tvalues,
    })
    save_csv(coef_df, TABLES_DIR / 'ols_model_b_coefficients.csv')

    residuals = model_b.resid
    fitted = model_b.fittedvalues

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black')
    axes[0].set_title('Residual Distribution')
    axes[0].set_xlabel('Residual')
    axes[0].set_ylabel('Density')

    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot')

    axes[2].scatter(fitted, residuals, alpha=0.3, s=10)
    axes[2].axhline(y=0, color='red', linestyle='--')
    axes[2].set_title('Fitted vs Residuals')
    axes[2].set_xlabel('Fitted Values')
    axes[2].set_ylabel('Residuals')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ols_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print(f"Model B: R2={model_b.rsquared:.4f}, Adj-R2={model_b.rsquared_adj:.4f}, N={len(y)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
