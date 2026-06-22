import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import numpy as np
from streamlit_folium import st_folium
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import OUTPUT_FILES, CITY_NAME, CBD_LAT, CBD_LON, TABLES_DIR, MAPS_DIR

st.set_page_config(page_title=f"{CITY_NAME} Airbnb Spatial Analysis", layout="wide")

st.title(f"{CITY_NAME} Airbnb - Spatial Price Analysis")
st.markdown("Interactive inspection of **residual patterns** and **spatial spillover effects** from SAR/SEM models.")

# ---------- data loading ----------

@st.cache_data
def load_data():
    model_df = pd.read_parquet(OUTPUT_FILES['model_sample'])
    residuals_df = pd.read_csv(OUTPUT_FILES['residuals_for_map'])
    merged = model_df.merge(residuals_df, on='listing_id', how='inner')

    points_sample = None
    if OUTPUT_FILES['map_points_sample'].exists():
        points_sample = gpd.read_file(OUTPUT_FILES['map_points_sample'])

    grid_cells = None
    if OUTPUT_FILES['map_grid_cells'].exists():
        grid_cells = gpd.read_file(OUTPUT_FILES['map_grid_cells'])

    effects_df = None
    effects_path = TABLES_DIR / 'spatial_effects.csv'
    if effects_path.exists():
        effects_df = pd.read_csv(effects_path)

    model_comp = None
    model_comp_path = TABLES_DIR / 'model_comparison.csv'
    if model_comp_path.exists():
        model_comp = pd.read_csv(model_comp_path)

    spillover_neigh = None
    spillover_path = OUTPUT_FILES['spillover_neighbourhoods']
    if spillover_path.exists():
        spillover_neigh = gpd.read_file(spillover_path)

    return merged, points_sample, grid_cells, effects_df, model_comp, spillover_neigh

merged, points_sample, grid_cells, effects_df, model_comp, spillover_neigh = load_data()

# ---------- compute dynamic model stats ----------

sar_rho = None
sdm_rho = None
sar_r2 = None
sdm_r2 = None
ols_r2 = None
sar_moran = None
ols_moran = None
sar_ind_dir_ratio = None
sdm_total_mult = None
sar_total_mult = None

if effects_df is not None:
    sar_eff = effects_df[effects_df['model'] == 'SAR']
    sdm_eff = effects_df[effects_df['model'] == 'SDM']

    if len(sar_eff) > 0 and 'coefficient' in sar_eff.columns and 'total' in sar_eff.columns:
        row0 = sar_eff.iloc[0]
        if row0['coefficient'] != 0:
            sar_total_mult = abs(row0['total'] / row0['coefficient'])
            sar_rho = round(1.0 - 1.0 / sar_total_mult, 4)
            sar_ind_dir_ratio = sar_total_mult - 1.0

    if len(sdm_eff) > 0 and 'beta_X' in sdm_eff.columns and 'total' in sdm_eff.columns:
        sdm_row0 = sdm_eff.iloc[0]
        denom = sdm_row0['beta_X'] + sdm_row0['beta_WX']
        if denom != 0:
            sdm_total_mult = abs(sdm_row0['total'] / denom)
            sdm_rho = round(1.0 - 1.0 / sdm_total_mult, 4)

if model_comp is not None:
    for _, row in model_comp.iterrows():
        if row['model'] == 'OLS':
            ols_r2 = row['r_squared']
            ols_moran = row['moran_I']
        elif row['model'] == 'SAR':
            sar_r2 = row['r_squared']
            sar_moran = row['moran_I']
        elif row['model'] == 'SDM' and 'r_squared' in row and pd.notna(row.get('r_squared')):
            sdm_r2 = row['r_squared']

# ---------- sidebar ----------

view_mode = st.sidebar.radio("Map View", ["Residual Map", "Spillover Analysis"], index=0)

if view_mode == "Residual Map":
    st.sidebar.header("Filters")
    price_range = st.sidebar.slider(
        "Price range (EUR/night)",
        min_value=int(merged['price'].min()),
        max_value=int(merged['price'].quantile(0.99)),
        value=(int(merged['price'].min()), int(merged['price'].quantile(0.95))),
        step=10
    )
    room_types = [c for c in merged.columns if c.startswith('room_type_')]
    selected_rooms = st.sidebar.multiselect("Room type", room_types, default=room_types)
    accommodates_range = st.sidebar.slider(
        "Accommodates (guests)",
        min_value=int(merged['accommodates'].min()),
        max_value=int(merged['accommodates'].max()),
        value=(int(merged['accommodates'].min()), int(merged['accommodates'].max()))
    )
    # WM-8: neighbourhood and CBD distance filters
    if 'neighbourhood_cleansed' in merged.columns:
        neighs = sorted(merged['neighbourhood_cleansed'].dropna().unique())
        selected_neighs = st.sidebar.multiselect("Neighbourhood", neighs, default=[])
    else:
        selected_neighs = []
    if 'dist_cbd_km' in merged.columns:
        cbd_range = st.sidebar.slider(
            "Distance from CBD (km)",
            min_value=0.0,
            max_value=float(merged['dist_cbd_km'].max()),
            value=(0.0, float(merged['dist_cbd_km'].max())),
            step=0.5
        )
    else:
        cbd_range = None
    st.sidebar.header("Residuals")
    residual_type = st.sidebar.radio("Show residuals from", ["OLS", "SAR", "SEM", "Comparison (OLS-SAR)"], index=0)
    highlight_threshold = st.sidebar.slider("Highlight threshold (|residual|)", 0.0, 2.0, 0.5, 0.1)
    show_points = st.sidebar.checkbox("Show individual listings", value=True)
    show_grid = st.sidebar.checkbox("Show neighborhood aggregates", value=True)

    mask = (
        (merged['price'] >= price_range[0]) &
        (merged['price'] <= price_range[1]) &
        (merged['accommodates'] >= accommodates_range[0]) &
        (merged['accommodates'] <= accommodates_range[1])
    )
    if selected_rooms:
        room_mask = merged[selected_rooms].any(axis=1)
        mask = mask & room_mask
    if selected_neighs and 'neighbourhood_cleansed' in merged.columns:
        mask = mask & merged['neighbourhood_cleansed'].isin(selected_neighs)
    if cbd_range is not None and 'dist_cbd_km' in merged.columns:
        mask = mask & (merged['dist_cbd_km'] >= cbd_range[0]) & (merged['dist_cbd_km'] <= cbd_range[1])

    filtered = merged[mask].copy()
    has_active_filters = (
        price_range != (int(merged['price'].min()), int(merged['price'].quantile(0.95))) or
        len(selected_rooms) < len(room_types) or
        accommodates_range != (int(merged['accommodates'].min()), int(merged['accommodates'].max())) or
        bool(selected_neighs) or
        (cbd_range is not None and cbd_range != (0.0, float(merged['dist_cbd_km'].max())))
    )
    st.sidebar.write(f"Filtered: {len(filtered)} / {len(merged)} listings")

    if residual_type == "OLS":
        resid_col = 'ols_residual'
    elif residual_type == "SAR":
        resid_col = 'sar_residual'
    elif residual_type == "SEM":
        resid_col = 'sem_residual'
    else:
        filtered['comparison_residual'] = filtered['ols_residual'] - filtered['sar_residual']
        resid_col = 'comparison_residual'

# ---------- residual color scale ----------

def get_residual_color(val):
    if val < -1.0:       return '#08519c'
    elif val < -0.5:     return '#3182bd'
    elif val < -0.1:     return '#9ecae1'
    elif val < 0.1:      return '#f7f7f7'
    elif val < 0.5:      return '#fc9272'
    elif val < 1.0:      return '#de2d26'
    else:                return '#a50f15'


def _get_neigh_name(row, columns):
    """Extract neighbourhood name from a GeoDataFrame row, checking common column names."""
    for col in ['neighbourhood', 'name', 'neighbourhood_cleansed', 'neighbourhood_group']:
        if col in columns and pd.notna(row.get(col)) and str(row[col]).strip():
            return str(row[col])
    return str(row.iloc[0])

# ---------- RESIDUAL MAP ----------

if view_mode == "Residual Map":
    m = folium.Map(location=[CBD_LAT, CBD_LON], zoom_start=12, tiles='CartoDB positron')

    if show_grid and grid_cells is not None:
        if has_active_filters:
            st.warning("Grid layer is not filtered. It shows all listings. Disable filters or hide grid for accurate comparison.")
        grid_filtered = grid_cells[grid_cells['listing_count'].notna()].copy()
        if residual_type == "OLS":
            grid_col = 'mean_ols_residual'
        elif residual_type == "SAR":
            grid_col = 'mean_sar_residual'
        elif residual_type == "SEM":
            grid_col = 'mean_sem_residual' if 'mean_sem_residual' in grid_filtered.columns else 'mean_ols_residual'
        else:
            grid_filtered['mean_comparison'] = grid_filtered['mean_ols_residual'] - grid_filtered['mean_sar_residual']
            grid_col = 'mean_comparison'

        for _, row in grid_filtered.iterrows():
            if pd.notna(row.get(grid_col)):
                color = get_residual_color(row[grid_col])
                opacity = 0.7 if abs(row[grid_col]) > highlight_threshold else 0.4
                weight = 2 if abs(row[grid_col]) > highlight_threshold else 1
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x, c=color, o=opacity, w=weight: {
                        'fillColor': c, 'fillOpacity': o, 'color': 'gray', 'weight': w,
                    },
                    tooltip=folium.Tooltip(
                        f"Mean residual: {row[grid_col]:.3f}<br>"
                        f"Listings: {int(row['listing_count'])}<br>"
                        f"Mean price: {row['mean_price']:.0f} EUR"
                    )
                ).add_to(m)

    if show_points and points_sample is not None:
        points_filtered = points_sample[points_sample['listing_id'].isin(filtered['listing_id'])].copy()
        for _, row in points_filtered.iterrows():
            if pd.notna(row.get(resid_col)):
                color = get_residual_color(row[resid_col])
                radius = 5 if abs(row[resid_col]) > highlight_threshold else 3
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=radius, color=color, fillColor=color, fillOpacity=0.7,
                    tooltip=folium.Tooltip(
                        f"ID: {row['listing_id']}<br>"
                        f"Price: {row['price']:.0f} EUR<br>"
                        f"log(price): {row['log_price']:.3f}<br>"
                        f"Residual: {row[resid_col]:.3f}<br>"
                        f"Accommodates: {row['accommodates']}"
                    )
                ).add_to(m)

    folium.Marker(
        [CBD_LAT, CBD_LON],
        popup='CBD (Piazza del Duomo)',
        icon=folium.Icon(color='black', icon='star')
    ).add_to(m)

    st_folium(m, width=None, height=600)

    st.subheader("Summary Statistics")
    if len(filtered) == 0:
        st.warning("No listings match the current filters. Adjust the sidebar filters to see results.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Mean log(price)", f"{filtered['log_price'].mean():.3f}")
            st.metric("Price range", f"{filtered['price'].min():.0f} - {filtered['price'].max():.0f} EUR")
        with c2:
            st.metric(f"Mean {residual_type} residual", f"{filtered[resid_col].mean():.4f}")
            st.metric(f"Std {residual_type} residual", f"{filtered[resid_col].std():.4f}")
        with c3:
            high_resid = (filtered[resid_col].abs() > 0.5).sum()
            st.metric("|r| > 0.5", f"{high_resid} ({high_resid/len(filtered)*100:.1f}%)")

    st.caption("Color: blue = model overestimates, red = underestimates, gray = good fit")

    if ols_r2 is not None or sar_r2 is not None:
        st.subheader("Model Performance")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            ols_r2_str = f"{ols_r2:.4f}" if ols_r2 is not None else "N/A"
            ols_m_str = f"{ols_moran:.4f}" if ols_moran is not None else "N/A"
            st.metric("OLS R²", ols_r2_str)
            st.metric("OLS Moran's I", ols_m_str)
        with mc2:
            sar_r2_str2 = f"{sar_r2:.4f}" if sar_r2 is not None else "N/A"
            sar_m_str = f"{sar_moran:.4f}" if sar_moran is not None else "N/A"
            st.metric("SAR Pseudo-R²", sar_r2_str2)
            st.metric("SAR Moran's I", sar_m_str)
        with mc3:
            sar_rho_str2 = f"{sar_rho:.4f}" if sar_rho is not None else "N/A"
            st.metric("SAR ρ (rho)", sar_rho_str2)
        st.caption("Higher ρ = stronger spatial dependence. Lower Moran's I = better model.")

    st.subheader("Residual Color Scale")
    st.markdown("""
    <div style="display:flex; align-items:center; font-size:0.85rem; gap:0;">
      <div style="background:#08519c;color:white;padding:4px 10px;text-align:center;">&lt;-1</div>
      <div style="background:#3182bd;color:white;padding:4px 10px;text-align:center;">-1..-0.5</div>
      <div style="background:#9ecae1;color:#333;padding:4px 10px;text-align:center;">-0.5..-0.1</div>
      <div style="background:#f7f7f7;color:#333;padding:4px 10px;text-align:center;">-0.1..0.1</div>
      <div style="background:#fc9272;color:#333;padding:4px 10px;text-align:center;">0.1..0.5</div>
      <div style="background:#de2d26;color:white;padding:4px 10px;text-align:center;">0.5..1</div>
      <div style="background:#a50f15;color:white;padding:4px 10px;text-align:center;">&gt;1</div>
    </div>
    <div style="font-size:0.8rem;margin-top:4px;color:#666;">
      <b>&larr; Overestimates</b> (predicted &gt; actual) &nbsp;|&nbsp;
      <b>Good fit</b> &nbsp;|&nbsp;
      <b>Underestimates &rarr;</b> (predicted &lt; actual)
    </div>
    """, unsafe_allow_html=True)

# ---------- SPILLOVER MAP ----------

elif view_mode == "Spillover Analysis":
    m = folium.Map(location=[CBD_LAT, CBD_LON], zoom_start=12, tiles='CartoDB positron')

    # Defaults in case spillover_neigh is None (used by Interpretation section below)
    shares_all = pd.Series(dtype=float)
    s_min, s_max = 0.0, 1.0

    if spillover_neigh is not None:
        share_col_map = 'sar_share' if 'sar_share' in spillover_neigh.columns else 'sdm_share'
        shares_all = spillover_neigh[share_col_map].dropna()

        # Data-driven color bins: the SAR spillover share is nearly constant
        # (mean ~17.4%, std ~0.6%) because it is mathematically constrained by
        # the single ρ parameter. Fixed bins (10-30%) made the map appear
        # uniformly light green. We compute bins from the actual data range
        # so the map shows meaningful relative variation.
        if len(shares_all) > 0:
            s_min = float(shares_all.min())
            s_max = float(shares_all.max())
        else:
            s_min, s_max = 0.0, 1.0

        # 7 color steps from light to dark green, evenly spaced over [s_min, s_max]
        green_palette = ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45']
        n_colors = len(green_palette)

        def spillover_color(val):
            if val is None or pd.isna(val):
                return '#e0e0e0'
            if s_max == s_min:
                return green_palette[n_colors // 2]
            # Normalize val to [0, 1] over the actual data range
            t = (val - s_min) / (s_max - s_min)
            t = max(0.0, min(1.0, t))
            idx = min(int(t * n_colors), n_colors - 1)
            return green_palette[idx]

        for _, row in spillover_neigh.iterrows():
            share = row.get('sar_share', row.get('sdm_share', None))
            if share is None or pd.isna(share):
                continue
            color = spillover_color(share)
            pct = share * 100
            neigh_name = _get_neigh_name(row, spillover_neigh.columns)
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x, c=color: {
                    'fillColor': c, 'fillOpacity': 0.7,
                    'color': '#333333', 'weight': 1,
                },
                tooltip=folium.Tooltip(
                    f"<b>{neigh_name}</b><br>"
                    f"Spillover share: {pct:.1f}%<br>"
                    f"Listings: {int(row.get('listing_count_y', row.get('listing_count_x', 0)))}<br>"
                    f"Mean log price: {row.get('mean_price_y', row.get('mean_price_x', 0)):.3f}"
                )
            ).add_to(m)

        folium.Marker(
            [CBD_LAT, CBD_LON],
            popup='CBD (Piazza del Duomo)',
            icon=folium.Icon(color='black', icon='star')
        ).add_to(m)

    st_folium(m, width=None, height=600)

    if spillover_neigh is not None:
        share_col = 'sar_share' if 'sar_share' in spillover_neigh.columns else 'sdm_share'
        shares = spillover_neigh[share_col].dropna()
        st.subheader("Spillover Statistics by Neighbourhood")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Mean spillover share", f"{shares.mean()*100:.1f}%")
        with c2:
            st.metric("Max spillover share", f"{shares.max()*100:.1f}%")
        with c3:
            st.metric("Min spillover share", f"{shares.min()*100:.1f}%")
        with c4:
            st.metric("Neighbourhoods", len(shares))

        top5 = spillover_neigh.nlargest(5, share_col)
        st.markdown("**Top 5 neighbourhoods by spillover intensity:**")
        for _, r in top5.iterrows():
            name = _get_neigh_name(r, spillover_neigh.columns)
            share = r[share_col] * 100
            st.markdown(f"- **{name}**: {share:.1f}% of predicted price from neighbor effects")

    st.subheader("Spatial Feedback Multipliers")
    sar_rho_str = f"{sar_rho:.4f}" if sar_rho is not None else "N/A"
    sdm_rho_str = f"{sdm_rho:.4f}" if sdm_rho is not None else "N/A"
    sar_r2_str = f"{sar_r2:.4f}" if sar_r2 is not None else "N/A"
    sdm_r2_str = f"{sdm_r2:.4f}" if sdm_r2 is not None else "N/A"
    sar_mult_str = f"{sar_total_mult:.3f}" if sar_total_mult is not None else "N/A"
    sdm_mult_str = f"{sdm_total_mult:.3f}" if sdm_total_mult is not None else "N/A"
    sar_ratio_str = f"{sar_ind_dir_ratio:.3f}" if sar_ind_dir_ratio is not None else "N/A"

    st.markdown(f"""
    The SAR model and SDM model decompose each variable's effect:

    | Model | ρ | Pseudo-R² | Spillover pattern |
    |---|---|---|---|
    | **SAR** | {sar_rho_str} | {sar_r2_str} | All variables share the same indirect/direct ratio ({sar_ratio_str}) |
    | **SDM** | {sdm_rho_str} | {sdm_r2_str} | Each variable has its own unique spillover intensity |

    **Total multipliers**: SAR = {sar_mult_str}, SDM = {sdm_mult_str}.
    """)

    st.warning(
        f"**SDM caveat**: The SDM is estimated on structural variables only (neighbourhood "
        f"fixed effects dropped to avoid X–WX collinearity). The resulting ρ={sdm_rho_str} is "
        f"near the unit-root boundary (<1), which inflates the total multiplier to "
        f"{sdm_mult_str}. This happens because the spatial lag absorbs the omitted "
        f"neighbourhood heterogeneity. **Total effects from SDM should be interpreted with "
        f"caution** — the SAR model (with neighbourhood FE, ρ={sar_rho_str}) provides the "
        f"more reliable and economically plausible impact estimates. The SDM is shown here "
        f"for completeness and to illustrate why the neighbourhood FE are necessary."
    )

    if effects_df is not None:
        st.subheader("Top Variable Effects (SDM — structural variables only)")
        st.caption(
            "⚠️ SDM total effects are inflated by the near-unit-root ρ. "
            "Direct effects (β) are more reliable than total effects. "
            "For economically interpretable impacts, refer to the SAR model."
        )
        key_vars = ['accommodates', 'bathrooms', 'bedrooms', 'beds',
                    'minimum_nights', 'availability_30', 'number_of_reviews',
                    'dist_cbd_km', 'review_scores_rating', 'has_reviews']
        sdm = effects_df[effects_df['model'] == 'SDM']
        display = sdm[sdm['variable'].isin(key_vars)].copy()

        def fmt(x):
            return f"{float(x):+.4f}"

        for col in ['beta_X', 'beta_WX', 'direct', 'indirect', 'total']:
            if col in display.columns:
                display[col] = display[col].apply(fmt)

        rename = {'variable': 'Variable', 'beta_X': 'Own (β)', 'beta_WX': 'Spatial (θ)',
                   'direct': 'Direct Effect', 'indirect': 'Indirect Effect', 'total': 'Total Effect'}
        display.rename(columns={k: v for k, v in rename.items() if k in display.columns}, inplace=True)
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Interpretation")
    col1, col2 = st.columns(2)
    with col1:
        if len(shares_all) > 0:
            st.markdown(f"""
            **Color Legend (Spillover Map)**

            Colors are scaled to the actual data range
            ({s_min*100:.1f}% – {s_max*100:.1f}%):

            - Light green: lower spillover share (relative to other neighbourhoods)
            - Medium green: moderate spillover
            - Dark green: higher spillover share

            **Note:** In the SAR model the spillover share is nearly uniform
            across neighbourhoods (mean {shares_all.mean()*100:.1f}%, std
            {shares_all.std()*100:.2f}%). This is expected: the share is driven
            by a single spatial parameter ρ={sar_rho_str}, so it varies little
            between areas. The color scale is stretched over the narrow observed
            range to make relative differences visible.
            """)
        else:
            st.markdown("""
            **Color Legend (Spillover Map)**
            - Light green: Low spillover
            - Medium green: Moderate spillover
            - Dark green: High spillover
            """)
    with col2:
        st.markdown("""
        **Why this matters**
        - Even small differences in spillover share indicate where neighbor
          effects matter more or less for pricing
        - Areas with higher share are more sensitive to neighborhood changes
        - Policy implications: regulating one listing type affects the whole area
        """)
