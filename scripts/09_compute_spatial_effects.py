import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.sparse import eye as sparse_eye
from scipy.sparse.linalg import spsolve
from scipy import stats
from libpysal.weights import KNN
from spreg import GM_Lag, GM_Combo

from src.config import OUTPUT_FILES, TABLES_DIR, MAPS_DIR, CRS_METRIC, RANDOM_SEED
from src.prep import get_y_X
from src.io import save_csv, save_geojson


def mc_traces(W, rho, n_mc=30, seed=42):
    """Monte Carlo trace estimation for R1 = tr((I-ρW)⁻¹) and R2 = tr(W(I-ρW)⁻¹).
    Uses Hutchinson's estimator with Rademacher random vectors.
    Returns (avg_direct_multiplier, avg_W_multiplier)."""
    rng = np.random.RandomState(seed)
    n = W.shape[0]
    A = (sparse_eye(n) - rho * W).tocsc()

    r1_sum = 0.0
    r2_sum = 0.0
    for _ in range(n_mc):
        u = rng.choice([-1, 1], size=n).astype(float)
        v = spsolve(A, u)
        r1_sum += float(np.dot(u, v))
        wv = W @ v
        r2_sum += float(np.dot(u, wv))

    return r1_sum / (n_mc * n), r2_sum / (n_mc * n)


def compute_sdm_impacts(W, rho, betas_X, betas_WX, var_names):
    """Decompose SDM coefficients into direct, indirect, total effects.
    SDM: y = ρWy + Xβ + WXθ + ε
    S_k(W) = (I-ρW)⁻¹(β_k·I + θ_k·W)
    Direct_k  = (1/N)·[β_k·tr((I-ρW)⁻¹) + θ_k·tr(W(I-ρW)⁻¹)]
    Total_k   = (β_k + θ_k) / (1-ρ)   [exact for row-standardized W]
    Indirect  = Total - Direct
    """
    avg_R1, avg_R2 = mc_traces(W, rho, n_mc=200, seed=RANDOM_SEED)
    total_mult = 1.0 / (1.0 - rho)

    impacts = []
    for name, beta_x, beta_wx in zip(var_names, betas_X, betas_WX):
        direct = beta_x * avg_R1 + beta_wx * avg_R2
        total = (beta_x + beta_wx) * total_mult
        indirect = total - direct

        impacts.append({
            'variable': name,
            'beta_X': round(float(beta_x), 6),
            'beta_WX': round(float(beta_wx), 6),
            'direct': round(float(direct), 6),
            'indirect': round(float(indirect), 6),
            'total': round(float(total), 6),
        })

    return impacts, avg_R1, avg_R2, total_mult


def compute_sar_impacts(W, rho, betas, var_names):
    """SAR: indirect/direct ratio is constrained (same for all variables)."""
    avg_R1, _ = mc_traces(W, rho, n_mc=200, seed=RANDOM_SEED)
    total_mult = 1.0 / (1.0 - rho)

    impacts = []
    for name, beta in zip(var_names, betas):
        direct = beta * avg_R1
        total = beta * total_mult
        indirect = total - direct

        impacts.append({
            'variable': name,
            'coefficient': round(float(beta), 6),
            'direct': round(float(direct), 6),
            'indirect': round(float(indirect), 6),
            'total': round(float(total), 6),
        })

    return impacts, avg_R1, total_mult - avg_R1, total_mult


def wald_test_sdm_restrictions(model_sdm, k, alpha=0.05):
    """Wald test for SDM nesting restrictions (Elhorst 2010, 2014).

    H0_A: θ = 0   → SDM reduces to SAR
    H0_B: θ + ρ·β = 0  → SDM reduces to SEM (common factor restriction)

    Uses the VC matrix from GM_Combo (vm=True required).
    Returns dict with test statistics, p-values, and decision.
    """
    rho = float(model_sdm.rho.flatten()[0])
    betas_all = model_sdm.betas.flatten()
    n_total = len(betas_all)
    if n_total == 2 * k + 2:
        w_offset = 1
    elif n_total == 2 * k + 3:
        w_offset = 2
    else:
        w_offset = max(0, n_total - 2 * k - 1)
    betas_X = betas_all[w_offset:w_offset + k]
    betas_WX = betas_all[w_offset + k:w_offset + 2 * k]

    if not hasattr(model_sdm, 'vm') or model_sdm.vm is None:
        print("  Wald test NOT available — no VC matrix. Use vm=True in GM_Combo.")
        return None

    V = model_sdm.vm
    n_params = len(betas_all)
    V_usable = V[:n_params, :n_params] if V.shape[0] > n_params else V

    print(f"\n  VC matrix: {V.shape},  betas: {n_params}  (k={k})")

    # ---- H0_A: θ = 0  (SDM → SAR) ----
    theta = betas_WX
    V_theta = V_usable[w_offset + k:w_offset + 2*k, w_offset + k:w_offset + 2*k]

    try:
        V_theta_inv = np.linalg.inv(V_theta)
        wald_A = float(theta.T @ V_theta_inv @ theta)
        p_A = 1 - stats.chi2.cdf(wald_A, k)
    except np.linalg.LinAlgError:
        ridge = np.eye(k) * 1e-6
        V_theta_inv = np.linalg.inv(V_theta + ridge)
        wald_A = float(theta.T @ V_theta_inv @ theta)
        p_A = 1 - stats.chi2.cdf(wald_A, k)

    reject_A = p_A < alpha

    print(f"  H0_A: θ=0 (SDM→SAR)")
    print(f"    Wald χ²({k}) = {wald_A:.2f}   p = {p_A:.4e}   → {'KEEP SDM: θ≠0, WX terms significant' if reject_A else 'SAR sufficient: θ=0 not rejected'}")

    # ---- H0_B: θ + ρ·β = 0  (SDM → SEM) ----
    c = betas_WX + rho * betas_X
    J = np.hstack([rho * np.eye(k), np.eye(k)])  # Jacobian d(θ+ρβ)/d(β,θ)
    V_block = V_usable[w_offset:w_offset + 2*k, w_offset:w_offset + 2*k]
    V_restriction = J @ V_block @ J.T

    try:
        V_r_inv = np.linalg.inv(V_restriction)
        wald_B = float(c.T @ V_r_inv @ c)
        p_B = 1 - stats.chi2.cdf(wald_B, k)
    except np.linalg.LinAlgError:
        ridge = np.eye(k) * 1e-6
        V_r_inv = np.linalg.inv(V_restriction + ridge)
        wald_B = float(c.T @ V_r_inv @ c)
        p_B = 1 - stats.chi2.cdf(wald_B, k)

    reject_B = p_B < alpha

    print(f"  H0_B: θ+ρβ=0 (SDM→SEM)")
    print(f"    Wald χ²({k}) = {wald_B:.2f}   p = {p_B:.4e}   → {'KEEP SDM: common factor restriction rejected' if reject_B else 'SEM sufficient: common factor holds'}")

    return {
        'wald_A_stat': wald_A, 'wald_A_p': p_A, 'wald_A_reject': reject_A,
        'wald_B_stat': wald_B, 'wald_B_p': p_B, 'wald_B_reject': reject_B,
        'k': k,
    }


def print_elhorst_decision(lm_results, wald_results):
    """Elhorst (2010) strategy: formal model selection with nested tests."""
    print("\n" + "=" * 60)
    print("ELHORST (2010) MODEL SELECTION STRATEGY")
    print("=" * 60)

    wA_reject = wald_results['wald_A_reject']
    wB_reject = wald_results['wald_B_reject']

    if wA_reject and wB_reject:
        print("  Both restrictions rejected → SDM is the correct specification")
        print("  The WX terms contain significant information.")
        print("  Proceed with SDM and per-variable impact decomposition.")
        choice = 'SDM'
    elif wA_reject and not wB_reject:
        print("  θ≠0 but θ+ρβ=0 not rejected → SEM is sufficient")
        print("  The spatial error specification captures the dependence.")
        choice = 'SEM'
    elif not wA_reject and wB_reject:
        print("  θ=0 not rejected but common factor rejected → theoretically inconsistent")
        print("  Likely SAR is adequate (WX terms are noise, check LM tests)")
        choice = 'SAR'
    else:
        print("  Neither restriction rejected → SAR or SEM are both sufficient")
        print("  Use the LM test decision rule to choose.")
        choice = 'LM-guided'

    print(f"\n  → Recommended model: {choice}")
    print("=" * 60)
    return choice


def main():
    print("=" * 60)
    print("SPATIAL EFFECTS — SAR vs SDM")
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
    W = w.sparse

    # ---- SAR ----
    print("\n[1/2] Estimating SAR...")
    model_sar = GM_Lag(y, X, w=w, name_y='log_price', name_x=X_cols, robust='white')
    rho_sar = float(model_sar.rho.flatten()[0])

    print(f"  ρ = {rho_sar:.4f}   Pseudo-R² = {model_sar.pr2:.4f}")

    sar_betas = model_sar.betas.flatten()
    if len(sar_betas) > len(X_cols):
        sar_betas = sar_betas[1:len(X_cols)+1]

    sar_impacts, sar_dir, sar_ind, sar_tot = compute_sar_impacts(
        W, rho_sar, sar_betas, X_cols
    )

    print(f"\n  Multipliers: direct={sar_dir:.4f}  indirect={sar_ind:.4f}  total={sar_tot:.4f}")
    print(f"  Indirect/direct ratio: {sar_ind/sar_dir:.3f} (constrained, same for all vars)")

    # ---- SDM ----
    # The SDM is estimated on STRUCTURAL variables only (excluding the 86
    # neighbourhood fixed-effect dummies). Including the neigh_* dummies in a
    # Spatial Durbin model makes X and WX nearly collinear (kNN neighbours
    # tend to live in the same neighbourhood), which collapses the model:
    # Pseudo-R² collapses to ~0 and coefficients explode to ~10^12.
    # The SAR above keeps the full Model B (with FE) for comparability with OLS;
    # the SDM drops the FE because WX|X for a one-hot-encoded neighbourhood
    # dummy is almost identical to X itself in dense urban areas.
    X_struct_cols = [c for c in X_cols if not c.startswith('neigh_')]
    X_struct = model_df[X_struct_cols].values
    k_struct = len(X_struct_cols)

    print(f"\n[2/2] Estimating SDM (Spatial Durbin, GMM) on {k_struct} structural variables (neigh FE dropped)")
    model_sdm = GM_Combo(y, X_struct, w=w, slx_lags=1, vm=True,
                          name_y='log_price', name_x=X_struct_cols)

    rho_sdm = float(model_sdm.rho.flatten()[0])
    betas_all = model_sdm.betas.flatten()
    n_total = len(betas_all)
    if n_total == 2 * k_struct + 2:
        sdm_offset = 1
    elif n_total == 2 * k_struct + 3:
        sdm_offset = 2
    else:
        sdm_offset = max(0, n_total - 2 * k_struct - 1)
    betas_X = betas_all[sdm_offset:sdm_offset + k_struct]
    betas_WX = betas_all[sdm_offset + k_struct:sdm_offset + 2 * k_struct]

    print(f"  ρ = {rho_sdm:.4f}   Pseudo-R² = {model_sdm.pr2:.4f}")
    print(f"  betas_X (n={len(betas_X)}), betas_WX (n={len(betas_WX)})")

    sdm_impacts, sdm_R1, sdm_R2, sdm_tot = compute_sdm_impacts(
        W, rho_sdm, betas_X, betas_WX, X_struct_cols
    )

    print(f"\n  Multipliers:  R₁={sdm_R1:.4f}  R₂={sdm_R2:.4f}  total={sdm_tot:.4f}")
    print(f"  (Each variable now has its own indirect/direct ratio)")

    # ---- Append SDM to model_comparison.csv (SAR/SEM rows come from script 07) ----
    comp_path = TABLES_DIR / 'model_comparison.csv'
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        if 'SDM' not in comp_df['model'].values:
            sdm_row = pd.DataFrame({
                'model': ['SDM'],
                'r_squared': [model_sdm.pr2],
                'rho': [rho_sdm],
                'lambda': [np.nan],
                'moran_I': [np.nan],
                'moran_p': [np.nan],
            })
            # Ensure columns align (rho/lambda may be missing in old CSVs)
            for c in ['rho', 'lambda', 'moran_I', 'moran_p']:
                if c not in comp_df.columns:
                    comp_df[c] = np.nan
            comp_df = pd.concat([comp_df, sdm_row[comp_df.columns]], ignore_index=True)
            save_csv(comp_df, comp_path)
            print(f"  Appended SDM row to {comp_path}")

    # ---- Elhorst (2010) formal restriction tests (on structural SDM) ----
    wald_results = wald_test_sdm_restrictions(model_sdm, k_struct, alpha=0.05)

    lm_df = pd.read_csv(TABLES_DIR / 'lm_diagnostic_tests.csv') if (TABLES_DIR / 'lm_diagnostic_tests.csv').exists() else None

    if wald_results:
        print_elhorst_decision(None, wald_results)

    # ---- Comparison table for key variables ----
    key_vars = ['accommodates', 'bathrooms', 'bedrooms', 'beds',
                'minimum_nights', 'availability_30', 'number_of_reviews']

    print("\n" + "=" * 80)
    print(f"{'Variable':<20s} │ {'SAR Direct':>10s} {'SAR Indirect':>12s} │ {'SDM Direct':>10s} {'SDM Indirect':>12s} {'SDM WX coef':>11s}")
    print("-" * 80)

    for var in key_vars:
        if var in X_struct_cols:
            idx = X_struct_cols.index(var)
            s = sar_impacts[X_cols.index(var)]
            d = sdm_impacts[idx]
            print(f"  {var:<18s} │ {s['direct']:10.4f} {s['indirect']:12.4f} │ "
                  f"{d['direct']:10.4f} {d['indirect']:12.4f} {d['beta_WX']:11.4f}")

    print("=" * 80)

    # ---- Save outputs ----
    df_sar = pd.DataFrame(sar_impacts)
    df_sar.insert(0, 'model', 'SAR')
    df_sdm = pd.DataFrame(sdm_impacts)
    df_sdm.insert(0, 'model', 'SDM')
    df_all = pd.concat([df_sar, df_sdm], ignore_index=True)
    save_csv(df_all, TABLES_DIR / 'spatial_effects.csv')
    print(f"\nEffects table saved: {TABLES_DIR / 'spatial_effects.csv'}")

    # ---- Spillover comparison SAR vs SDM ----
    sar_resid = model_sar.u.flatten()
    sdm_resid = model_sdm.u.flatten()

    # SAR uses the full Model B (124 X cols); SDM uses structural-only (no FE).
    Xb_sar = X @ sar_betas
    Xb_sdm = X_struct @ betas_X
    Wy = W @ y

    sar_intercept = float(model_sar.betas.flatten()[0])
    sdm_intercept = float(model_sdm.betas.flatten()[0])

    spillover = pd.DataFrame({
        'listing_id': model_df['listing_id'].values,
        'neighbourhood': model_df.get('neighbourhood_cleansed', pd.Series(['']*len(y))).values,
        'log_price': y,
        'sar_own': Xb_sar,
        'sar_rho_Wy': rho_sar * Wy,
        'sdm_own': Xb_sdm,
        'sdm_rho_Wy': rho_sdm * Wy,
        'sdm_WX_contrib': X_struct @ betas_WX,
        'sar_residual': sar_resid,
        'sdm_residual': sdm_resid,
        'lat': model_df['latitude'].values,
        'lon': model_df['longitude'].values,
    })

    spillover['sar_spillover_share'] = (
        np.abs(spillover['sar_rho_Wy']) /
        (np.abs(sar_intercept + spillover['sar_own']) + np.abs(spillover['sar_rho_Wy']) + 1e-6)
    )
    spillover['sdm_spillover_share'] = (
        np.abs(spillover['sdm_rho_Wy']) /
        (np.abs(sdm_intercept + spillover['sdm_own']) + np.abs(spillover['sdm_WX_contrib']) + np.abs(spillover['sdm_rho_Wy']) + 1e-6)
    )

    save_csv(spillover, TABLES_DIR / 'spillover_listings.csv')
    print(f"Spillover listing data saved ({len(spillover)} listings)")

    # ---- Neighbourhood aggregates for mapping ----
    if 'neighbourhood' in spillover.columns and not spillover['neighbourhood'].eq('').all():
        neigh_agg = spillover.groupby('neighbourhood').agg(
            sar_share=('sar_spillover_share', 'mean'),
            sdm_share=('sdm_spillover_share', 'mean'),
            mean_price=('log_price', 'mean'),
            listing_count=('listing_id', 'count'),
            mean_lat=('lat', 'mean'),
            mean_lon=('lon', 'mean'),
        ).reset_index()

        gdf_neigh = gpd.read_file(OUTPUT_FILES['neighbourhoods_enriched_geojson'])
        gdf_merged = gdf_neigh.merge(neigh_agg, left_on=gdf_neigh.columns[0],
                                      right_on='neighbourhood', how='inner')
        save_geojson(gdf_merged, MAPS_DIR / 'spillover_neighbourhoods.geojson')
        print(f"Neighbourhood spillover GeoJSON saved ({len(gdf_merged)} neighbourhoods)")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY — SAR vs SDM")
    print("=" * 60)
    print(f"  SAR: ρ={rho_sar:.4f}  Pseudo-R²={model_sar.pr2:.4f}  indirect/direct≡{sar_ind/sar_dir:.3f}")
    print(f"  SDM: ρ={rho_sdm:.4f}  Pseudo-R²={model_sdm.pr2:.4f}  per-variable spillover")
    print(f"  Better fit: {'SDM' if model_sdm.pr2 > model_sar.pr2 else 'SAR'}")
    print(f"  SDM advantage: each variable can have different spillover intensity")
    print("=" * 60)

if __name__ == "__main__":
    main()
