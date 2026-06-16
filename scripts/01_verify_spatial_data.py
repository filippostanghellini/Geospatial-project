import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import OUTPUT_FILES
from src.io import load_parquet, load_geojson

def main():
    print("=" * 60)
    print("VERIFY SPATIAL DATA")
    print("=" * 60)

    print("\n--- Processed Files ---")
    for name, path in OUTPUT_FILES.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {name}: {path.name} ({size_mb:.2f} MB)")
        else:
            print(f"  {name}: NOT FOUND")

    print("\n--- Listings Clean ---")
    if OUTPUT_FILES['listings_clean'].exists():
        df = load_parquet(OUTPUT_FILES['listings_clean'])
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns[:20])}...")
        print(f"  Price stats:")
        if 'price' in df.columns:
            print(f"    mean={df['price'].mean():.2f}, median={df['price'].median():.2f}")
            print(f"    min={df['price'].min():.2f}, max={df['price'].max():.2f}")

    print("\n--- Neighbourhoods Enriched ---")
    if OUTPUT_FILES['neighbourhoods_enriched_geojson'].exists():
        gdf = load_geojson(OUTPUT_FILES['neighbourhoods_enriched_geojson'])
        print(f"  Shape: {gdf.shape}")
        print(f"  Columns: {list(gdf.columns)}")
        if 'listing_count' in gdf.columns:
            print(f"  Total listings: {gdf['listing_count'].sum()}")
        if 'median_price' in gdf.columns:
            print(f"  Median price range: {gdf['median_price'].min():.2f} - {gdf['median_price'].max():.2f}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
