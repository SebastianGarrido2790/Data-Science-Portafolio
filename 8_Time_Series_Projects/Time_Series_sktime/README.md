# Time Series Forecasting with Multiple Techniques

This project focuses on analyzing and forecasting time series data, specifically daily sales data, using multiple techniques: **Prophet**, **sktime**, and **XGBoost**. It includes data processing, exploratory data analysis (EDA), model training, backtesting, and final forecasting, all organized within a structured directory layout. The primary dataset, `sales_for_fc.csv`, represents aggregated daily sales data derived from various subcategories.

## Project Overview

The goal is to predict future sales by leveraging different time series forecasting methods:

- **Prophet**: A robust tool for handling seasonality and trends, with hyperparameter tuning and holiday effects.
- **sktime**: A flexible time series toolkit, here using AutoARIMA or Prophet implementations for forecasting.
- **XGBoost**: A machine learning approach with feature engineering for time series prediction.

The project supports reproducibility through a Conda environment and provides visualizations and performance metrics for evaluation.

## Setup Instructions

To set up the project environment:

1. **Install Conda (if not already installed):** Follow the instructions at [conda.io](https://conda.io).

2. **Create the Conda environment using the provided `time_series_env.yml`:**

```bash
conda env create -f time_series_env.yml
```

3. **Activate the environment:**

```bash
conda activate time_series_env
```

The environment includes dependencies like `prophet`, `sktime`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `jupyterlab`, and more, as specified in `time_series_env.yml`.

## Folder Structure

The project is organized as follows:

- **data/**: Contains datasets at various stages.
- **external/**: Data from third-party sources.
- **interim/**: Intermediate data (e.g., `sales_ts.csv`, `total_sales.csv`).
- **processed/**: Final datasets for modeling (e.g., `sales_for_fc.csv`).
- **raw/**: Original, unprocessed data.
- **docs/**: Project documentation (default Sphinx setup).
- **models/**: Stores trained models, predictions, and summaries (e.g., `prophet_params.pkl`).
- **notebooks/**: Jupyter notebooks for analysis.
  - `EDA.ipynb`: Exploratory data analysis with visualizations and stationarity tests.
- **references/**: Supplementary materials like data dictionaries or manuals.
- **reports/**: Generated reports and visualizations.
  - **figures/**: Plots from forecasting (e.g., `sales_forecast.png`).
- **src/**: Source code directory.
  - **data/**: Scripts for data preparation (e.g., `make_dataset.py`).
  - **features/**: Feature engineering scripts (e.g., `build_features.py`).
  - **models/**: Model training and prediction scripts.
    - **Prophet/**:
      - `train_model.py`: Trains Prophet models with hyperparameter tuning.
      - `backtesting.py`: Evaluates models via backtesting.
      - `final_forecasting.py`: Generates final forecasts and plots.
    - `predict_with_skime.py`: Forecasts using sktime (e.g., AutoARIMA or Prophet).
    - `predict_with_xgboost.py`: Forecasts using XGBoost with feature engineering.
  - **utils/**: Utility scripts (e.g., `plot_settings.py`).
  - **visualization/**: Visualization scripts (includes `EDA.ipynb`).
- `LICENSE`: Project license file.
- `README.md`: This file.
- `requirements.txt`: Pip-generated dependency list (secondary to `time_series_env.yml`).
- `time_series_env.yml`: Conda environment specification.

## Data

The main dataset is `sales_for_fc.csv` in `data/processed/`, containing daily sales aggregated from subcategories after filtering out low-volume ones (see `EDA.ipynb`). The data originates from `data/interim/total_sales.csv`, processed to focus on mid- and high-volume subcategories.

## Models

Three forecasting approaches are implemented:

### Prophet (src/models/Prophet/)
Handles seasonality, trends, and holidays.

**Scripts:**
- `train_model.py`: Tunes hyperparameters and saves models.
- `backtesting.py`: Validates forecasts over a 30-day horizon.
- `final_forecasting.py`: Predicts 30 days ahead with plots.

### sktime (src/models/predict_with_skime.py)
Uses AutoARIMA or Prophet for forecasting.

- Supports validation (last 30 days) and future forecasting with confidence intervals.

### XGBoost (src/models/predict_with_xgboost.py)
Employs feature engineering (lags, rolling stats, holidays) and hyperparameter tuning.

- Forecasts 30 days beyond the dataset.

## Notebooks

**EDA.ipynb (src/visualization/):**
- Performs exploratory data analysis.
- Includes time series decomposition, stationarity tests (ADF), moving averages, and data filtering.
- Outputs `sales_for_fc.csv` after processing.

## Running the Project

1. **Set up the environment:** (see Setup Instructions).

2. **Prepare the data:**
   - Run `EDA.ipynb` to generate `sales_for_fc.csv` from `total_sales.csv`.

3. **Run the models:**

### Prophet:
```bash
python src/models/Prophet/train_model.py      # Train and save model parameters
python src/models/Prophet/backtesting.py        # Backtest over 30 days
python src/models/Prophet/final_forecasting.py   # Forecast 30 days ahead
```

### sktime:
```bash
python src/models/predict_with_skime.py          # Validate and forecast 30 days
```

### XGBoost:
```bash
python src/models/predict_with_xgboost.py        # Train, validate, and forecast 30 days
```

4. **View results:**
- Model outputs (e.g., `prophet_params.pkl`) are in `models/`.
- Forecast plots and metrics are in `reports/figures/`.

## Example Commands

To run the full Prophet pipeline:

```bash
conda activate time_series_env
python src/models/Prophet/train_model.py
python src/models/Prophet/backtesting.py
python src/models/Prophet/final_forecasting.py
```

## Results

- **Prophet:** Backtesting metrics (MAPE, SMAPE, MAE) in console; forecasts and component plots in `reports/figures/`.
- **sktime:** MAE, RMSE, MAPE from validation; forecasts in console and plots.
- **XGBoost:** MAE, RMSE, SMAPE from test set; 30-day forecasts plotted and printed.

Results are stored in:
- `models/`: Model parameters and predictions.
- `reports/figures/`: Visualization outputs.

## License

This project is licensed under the terms in the `MIT LICENSE` (./LICENSE) file.
