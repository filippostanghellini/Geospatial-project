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

```bash
python scripts/01_verify_spatial_data.py
python scripts/03_ols_price_analysis.py
python scripts/04_spatial_autocorr_morans_i.py
python scripts/05_lm_diagnostic_tests.py
python scripts/07_spatial_models_sar_sem.py
python scripts/02_make_static_map_overview.py
```

### 5) Web map (optional)

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
├── outputs/             # tables + intermediate artifacts
├── reports/figures/     # final report figures
└── webmap/              # Streamlit app
```

## Analysis Workflow

1. **Data Preparation** (`notebooks/01_data_pipeline.ipynb`)
   - Clean listings, calendar, reviews
   - Spatial join listings to neighbourhoods
   - Aggregate metrics to neighbourhood level
   - Quality checks and visualization

2. **OLS Regression** (`scripts/03_ols_price_analysis.py`)
   - Model A: Property + host + room type characteristics
   - Model B: Model A + distance to CBD + neighbourhood fixed effects

3. **Spatial Autocorrelation** (`scripts/04_spatial_autocorr_morans_i.py`)
   - Moran's I on OLS residuals (listing-level kNN, neighbourhood-level Queen)

4. **LM Diagnostic Tests** (`scripts/05_lm_diagnostic_tests.py`)
   - LM-lag, LM-error, Robust LM-lag, Robust LM-error

5. **Spatial Models** (`scripts/07_spatial_models_sar_sem.py`)
   - SAR (Spatial Autoregressive Model, GMM)
   - SEM (Spatial Error Model, GMM)
   - Post-fit Moran's I comparison

## CRS Policy

| Purpose | CRS | EPSG |
|---|---|---|
| Web/display | WGS 84 | EPSG:4326 |
| Metric computations | UTM Zone 32N | EPSG:32632 |

## Data Source

- [Inside Airbnb - Milan](https://insideairbnb.com/milan/)
- Snapshot date: 22 September 2025

## References

- Anselin, L. (1988). *Spatial Econometrics: Methods and Models*
- Rey, S.J., Arribas-Bel, D., Wolf, L.J. (2020). *Geographic Data Science with PySAL and the PyData Stack*
