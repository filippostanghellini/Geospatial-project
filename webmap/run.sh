#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "Milan Airbnb Webmap Launcher"
echo "============================================"

cd "$PROJECT_ROOT"

if [ ! -f "data/processed/model_sample.parquet" ]; then
    echo "ERROR: model_sample.parquet not found."
    echo "Run the data pipeline notebook AND the OLS script first:"
    echo "  jupyter execute notebooks/01_data_pipeline.ipynb --inplace"
    echo "  python scripts/03_ols_price_analysis.py"
    exit 1
fi

if [ ! -f "outputs/tables/residuals_for_map.csv" ]; then
    echo "ERROR: residuals_for_map.csv not found."
    echo "Run the full pipeline first. See README.md for instructions."
    exit 1
else
    echo "Residuals found, verifying..."
    python scripts/07b_extract_residuals.py
fi

if [ ! -f "data/processed/map_points_sample.geojson" ]; then
    echo "Generating map layers..."
    python scripts/08_prepare_map_layers.py
fi

echo "Launching Streamlit app..."
streamlit run webmap/app.py --server.port 8501
