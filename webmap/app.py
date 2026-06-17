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

from src.config import OUTPUT_FILES, CITY_NAME, CBD_LAT, CBD_LON, TABLES_DIR

st.set_page_config(page_title=f"{CITY_NAME} Airbnb Spatial Analysis", layout="wide")

st.title(f"{CITY_NAME} Airbnb - Spatial Price Analysis")
st.markdown("Interactive inspection of **residual patterns** and **spatial spillover effects** from SAR model.")

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

    spillover_neigh = None
    spillover_path = TABLES_DIR.parent / 'maps' / 'spillover_neighbourhoods.geojson'
    if spillover_path.exists():
        spillover_neigh = gpd.read_file(spillover_path)

    return merged, points_sample, grid_cells, effects_df, spillover_neigh

merged, points_sample, grid_cells, effects_df, spillover_neigh = load_data()

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
    st.sidebar.header("Residuals")
    residual_type = st.sidebar.radio("Show residuals from", ["OLS", "SAR", "Comparison (OLS-SAR)"], index=0)
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

    filtered = merged[mask].copy()
    st.sidebar.write(f"Filtered: {len(filtered)} / {len(merged)} listings")

    if residual_type == "OLS":
        resid_col = 'ols_residual'
    elif residual_type == "SAR":
        resid_col = 'sar_residual'
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

# ---------- RESIDUAL MAP ----------

if view_mode == "Residual Map":
    m = folium.Map(location=[CBD_LAT, CBD_LON], zoom_start=12, tiles='CartoDB positron')

    if show_grid and grid_cells is not None:
        grid_filtered = grid_cells[grid_cells['listing_count'].notna()].copy()
        if residual_type == "OLS":
            grid_col = 'mean_ols_residual'
        elif residual_type == "SAR":
            grid_col = 'mean_sar_residual'
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

# ---------- SPILLOVER MAP ----------

elif view_mode == "Spillover Analysis":
    m = folium.Map(location=[CBD_LAT, CBD_LON], zoom_start=12, tiles='CartoDB positron')

    if spillover_neigh is not None:
        def spillover_color(val):
            if val is None or pd.isna(val):
                return '#e0e0e0'
            val = max(0, min(1, val))
            if val < 0.10:      return '#f7fcf5'
            elif val < 0.15:    return '#e5f5e0'
            elif val < 0.18:    return '#c7e9c0'
            elif val < 0.20:    return '#a1d99b'
            elif val < 0.22:    return '#74c476'
            elif val < 0.25:    return '#41ab5d'
            elif val < 0.30:    return '#238b45'
            else:               return '#005a32'

        for _, row in spillover_neigh.iterrows():
            share = row.get('sdm_share', row.get('sar_share', None))
            if share is None or pd.isna(share):
                continue
            color = spillover_color(share)
            pct = share * 100
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x, c=color: {
                    'fillColor': c, 'fillOpacity': 0.7,
                    'color': '#333333', 'weight': 1,
                },
                tooltip=folium.Tooltip(
                    f"<b>{row.get('neighbourhood', row.iloc[0])}</b><br>"
                    f"Spillover share: {pct:.1f}%<br>"
                    f"Listings: {int(row.get('listing_count', 0))}<br>"
                    f"Mean log price: {row.get('mean_price', 0):.3f}"
                )
            ).add_to(m)

        folium.Marker(
            [CBD_LAT, CBD_LON],
            popup='CBD (Piazza del Duomo)',
            icon=folium.Icon(color='black', icon='star')
        ).add_to(m)

    st_folium(m, width=None, height=600)

    if spillover_neigh is not None:
        share_col = 'sdm_share' if 'sdm_share' in spillover_neigh.columns else 'sar_share'
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
            name = r.get('neighbourhood', r.iloc[0])
            share = r[share_col] * 100
            st.markdown(f"- **{name}**: {share:.1f}% of predicted price from neighbor effects")

    st.subheader("Spatial Feedback Multipliers")
    st.markdown(f"""
    The SAR model (simpler) and SDM model (richer) decompose each variable's effect:

    | Model | ρ | Pseudo-R² | Spillover pattern |
    |---|---|---|---|
    | **SAR** | 0.251 | 0.476 | All variables share the same indirect/direct ratio (0.325) |
    | **SDM** | 0.872 | 0.472 | Each variable has its own unique spillover intensity |

    **Why SDM matters**: In SAR, `bedrooms` and `accommodates` have the same spillover ratio.
    In SDM, `bedrooms` has a **strong positive** spillover (more bedrooms nearby → higher prices),
    while `accommodates` has a **negative** spillover (more guests nearby → higher competition → lower prices).

    **Total multipliers**: SAR = 1.335, SDM = 7.806 (higher ρ in SDM means stronger spatial feedback,
    but offset by negative WX coefficients).
    """)

    if effects_df is not None:
        st.subheader("Top Variable Effects (SDM)")
        key_vars = ['accommodates', 'bathrooms', 'bedrooms', 'beds',
                    'minimum_nights', 'availability_30', 'number_of_reviews']
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
        st.markdown("""
        **Color Legend (Spillover Map)**
        - Light green: Low spillover (%) - price mostly from own characteristics
        - Medium green: Moderate spillover
        - Dark green: High spillover - price heavily influenced by neighbors
        """)
    with col2:
        st.markdown("""
        **Why this matters**
        - High-spillover areas are more sensitive to neighborhood changes
        - A new amenity or luxury listing in these areas has amplified effects
        - Policy implications: regulating one listing type affects the whole area
        """)
