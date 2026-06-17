import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

from src.config import OUTPUT_FILES, FIGURES_DIR, CITY_NAME, CBD_LAT, CBD_LON, CRS_WEB

def main():
    print("=" * 60)
    print("STATIC MAP OVERVIEW")
    print("=" * 60)

    model_df = pd.read_parquet(OUTPUT_FILES['model_sample'])
    gdf = gpd.GeoDataFrame(
        model_df,
        geometry=gpd.points_from_xy(model_df['longitude'], model_df['latitude']),
        crs=CRS_WEB
    )

    neigh_path = OUTPUT_FILES['neighbourhoods_enriched_geojson']
    gdf_neigh = None
    if neigh_path.exists():
        gdf_neigh = gpd.read_file(neigh_path)

    fig, ax = plt.subplots(figsize=(14, 12))

    if gdf_neigh is not None:
        gdf_neigh.plot(ax=ax, facecolor='lightgray', edgecolor='gray', alpha=0.3)

    gdf.plot(
        ax=ax,
        column='price',
        cmap='YlOrRd',
        markersize=3,
        alpha=0.5,
        legend=True,
        legend_kwds={'label': 'Price (EUR/night)', 'orientation': 'vertical', 'shrink': 0.6}
    )

    ax.plot(CBD_LON, CBD_LAT, 'k*', markersize=15, label=f'CBD ({CBD_LAT:.4f}, {CBD_LON:.4f})')

    ax.set_title(f'{CITY_NAME} Airbnb Listings by Price', fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='upper left')

    try:
        gdf_3857 = gdf.to_crs(epsg=3857)
        bounds_3857 = gdf_3857.total_bounds
        ax_3857 = fig.add_axes(ax.get_position(), projection=None, zorder=-1)
        ax_3857.set_xlim(bounds_3857[0], bounds_3857[2])
        ax_3857.set_ylim(bounds_3857[1], bounds_3857[3])
        ctx.add_basemap(ax_3857, crs='EPSG:3857', source=ctx.providers.CartoDB.Positron, alpha=0.3)
        ax_3857.set_axis_off()
    except Exception as e:
        print(f"  Could not add basemap: {e}")

    output_path = FIGURES_DIR / f'fig_{CITY_NAME.lower()}_overview_price.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Map saved to {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
