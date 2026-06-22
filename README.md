# Milan Airbnb — Geospatial Analysis

This repository provides a reproducible geospatial workflow to study Airbnb prices in Milan using Inside Airbnb data. The analysis combines data preparation, OLS and spatial econometric models (SAR/SEM), and map-based inspection of residual patterns.

**Research question:** To what extent does location (accessibility + neighbourhood context) explain nightly prices beyond listing/host characteristics, and when are spatial models preferable to OLS?

## Quickstart

### 1) Create environment

```bash
micromamba env create -f environment/environment.yml
micromamba activate geo
```

### 2) Data

Data is already downloaded in `data/original/` (Milan snapshot **22 September 2025** from Inside Airbnb).

### 3) Run notebook (data preparation)

```bash
jupyter execute notebooks/01_data_pipeline.ipynb --inplace
```

### 4) Run scripts (analysis)

All scripts must be run with the `geo` environment activated (step 1).

```bash
python scripts/01_verify_spatial_data.py
python scripts/02_make_static_map_overview.py
python scripts/03_ols_price_analysis.py
python scripts/04_spatial_autocorr_morans_i.py
python scripts/05_lm_diagnostic_tests.py
python scripts/06_lisa_cluster_analysis.py
python scripts/07_spatial_models_sar_sem.py
python scripts/07b_extract_residuals.py
python scripts/08_prepare_map_layers.py
python scripts/09_compute_spatial_effects.py
```

### 5) Web map

```bash
bash webmap/run.sh
# or
streamlit run webmap/app.py
```

## Repository structure

```
Geopsatial-project/
├── data/
│   ├── original/        # raw inputs (Inside Airbnb)
│   └── processed/       # generated datasets
├── environment/         # environment specification
├── notebooks/           # preparation notebooks
├── scripts/             # analysis entrypoints
├── src/                 # reusable modules
├── outputs/             # tables, maps + intermediate artifacts
│   ├── tables/
│   └── maps/
├── reports/
│   ├── figures/         # final report figures
│   └── maps/            # final report maps
├── .report/             # internal analysis reports (methodological_analysis.md, models_and_results.md)
└── webmap/              # Streamlit app
```

## Analysis Workflow

1. **Data Preparation** (`notebooks/01_data_pipeline.ipynb`)
   - Clean listings, calendar, reviews
   - Spatial join listings to neighbourhoods
   - Aggregate metrics to neighbourhood level
   - Quality checks and visualization

2. **OLS Regression** (`scripts/03_ols_price_analysis.py`)
   - Model A: Property + host + room type characteristics + distance to CBD + review scores
   - Model B: Model A + neighbourhood fixed effects (86 dummies)

3. **Spatial Autocorrelation** (`scripts/04_spatial_autocorr_morans_i.py`)
   - Moran's I on OLS residuals (listing-level kNN, neighbourhood-level Queen)
   - kNN sensitivity test (k=4, 8, 12)

4. **LM Diagnostic Tests** (`scripts/05_lm_diagnostic_tests.py`)
   - LM-lag, LM-error, Robust LM-lag, Robust LM-error (via `spreg.OLS`)

5. **LISA Cluster Analysis** (`scripts/06_lisa_cluster_analysis.py`)
   - Local Indicators of Spatial Association on OLS residuals (999 permutations)
   - Moran scatterplot, LISA cluster map, z-score distribution

6. **Spatial Models** (`scripts/07_spatial_models_sar_sem.py`)
   - SAR (Spatial Autoregressive Model, GMM heteroskedastic-robust)
   - SEM (Spatial Error Model, GMM heteroskedastic-robust)
   - Post-fit Moran's I on residuals (filtered innovation for SEM)

7. **Residual Extraction** (`scripts/07b_extract_residuals.py`)
   - Load OLS, SAR, and SEM residuals for map visualization

8. **Map Layers** (`scripts/08_prepare_map_layers.py`)
   - Prepare point and grid GeoJSON layers for the web map

9. **Spatial Effects** (`scripts/09_compute_spatial_effects.py`)
   - Direct, indirect, and total effects from the SAR model (LeSage-Pace decomposition)
   - SDM (Spatial Durbin Model) on structural variables only (neighbourhood FE dropped)
   - Monte Carlo trace estimation (Hutchinson, 200 iterations)
   - Wald restriction tests (Elhorst 2010)

## CRS Policy

| Purpose | CRS | EPSG |
|---|---|---|
| Web/display | WGS 84 | EPSG:4326 |
| Metric computations | UTM Zone 32N | EPSG:32632 |

## Data Source

- [Inside Airbnb - Milan](https://insideairbnb.com/milan/)
- Snapshot date: 22 September 2025
