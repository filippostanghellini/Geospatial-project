import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import OUTPUT_FILES, CITY_NAME, CBD_LAT, CBD_LON

st.set_page_config(page_title=f"{CITY_NAME} Airbnb Spatial Analysis", layout="wide")

st.title(f"{CITY_NAME} Airbnb - Spatial Price Analysis")
st.markdown("Interactive map for inspecting residual patterns from OLS and spatial regression models.")

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

    return merged, points_sample, grid_cells

merged, points_sample, grid_cells = load_data()

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

def get_color(val, threshold=0.5):
    if val < -1.0:
        return '#08519c'
    elif val < -0.5:
        return '#3182bd'
    elif val < -0.1:
        return '#9ecae1'
    elif val < 0.1:
        return '#f7f7f7'
    elif val < 0.5:
        return '#fc9272'
    elif val < 1.0:
        return '#de2d26'
    else:
        return '#a50f15'

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
            color = get_color(row[grid_col], highlight_threshold)
            opacity = 0.7 if abs(row[grid_col]) > highlight_threshold else 0.4
            weight = 2 if abs(row[grid_col]) > highlight_threshold else 1

            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x, c=color, o=opacity, w=weight: {
                    'fillColor': c,
                    'fillOpacity': o,
                    'color': 'gray',
                    'weight': w,
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
            color = get_color(row[resid_col], highlight_threshold)
            radius = 5 if abs(row[resid_col]) > highlight_threshold else 3

            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=radius,
                color=color,
                fillColor=color,
                fillOpacity=0.7,
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
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Mean log(price)", f"{filtered['log_price'].mean():.3f}")
    st.metric("Price range", f"{filtered['price'].min():.0f} - {filtered['price'].max():.0f} EUR")

with col2:
    st.metric(f"Mean {residual_type} residual", f"{filtered[resid_col].mean():.4f}")
    st.metric(f"Std {residual_type} residual", f"{filtered[resid_col].std():.4f}")

with col3:
    high_resid = (filtered[resid_col].abs() > 0.5).sum()
    st.metric("High residual listings (|r|>0.5)", f"{high_resid}")
    st.metric("% high residual", f"{high_resid/len(filtered)*100:.1f}%")

st.subheader("Interpretation Guide")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Color Legend**
    - Dark blue: Model strongly overestimates
    - Light blue: Slight overestimate
    - Gray: Good fit
    - Light red: Slight underestimate
    - Dark red: Model strongly underestimates
    """)

with col2:
    st.markdown(f"""
    **Current Filters**
    - Price: {price_range[0]}-{price_range[1]} EUR
    - Accommodates: {accommodates_range[0]}-{accommodates_range[1]}
    - Listings shown: {len(filtered)}
    - Residual type: {residual_type}
    """)

with col3:
    st.markdown("""
    **Diagnostic Scope**
    - Exploratory visualization
    - Not standalone evidence
    - Use with formal tests (Moran's I, LM tests)
    """)
