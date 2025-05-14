# Online Retail Customer Segmentation

This project demonstrates how to use data science techniques—specifically, RFM (Recency, Frequency, Monetary) analysis and KMeans clustering—to segment customers of an online retail business. The insights derived from this analysis can be used to drive customer-centric marketing strategies.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Cleaning and Feature Engineering](#data-cleaning-and-feature-engineering)
  - [Cluster Analysis](#cluster-analysis)
  - [Visualization](#visualization)
- [Outlier Removal](#outlier-removal)
- [Future Work](#future-work)
- [License](#license)

## Overview

This project focuses on:
- Importing and preprocessing customer transaction data.
- Creating RFM metrics (Recency, Frequency, and Monetary) for each customer.
- Removing outliers using the interquartile range (IQR) method.
- Applying KMeans clustering to segment customers based on their purchasing behavior.
- Visualizing the clusters using various plots (e.g., violin plots, pairplots, PCA projections).

The final goal is to provide actionable insights into customer segments that can inform customer-centric marketing strategies.

## Project Structure

```plaintext
├── LICENSE
├── README.md                    <- This file.
├── data
│   ├── external                 <- Data from third party sources.
│   ├── interim                  <- Intermediate data that has been transformed.
│   ├── processed                <- Final, canonical datasets for modeling.
│   └── raw                      <- The original, immutable data dump.
├── docs                         <- Documentation (e.g., Sphinx docs).
├── models                       <- Trained models and model predictions.
├── notebooks                    <- Jupyter notebooks for exploration and analysis.
├── references                   <- Data dictionaries, manuals, and supporting docs.
├── reports                      <- Generated reports (HTML, PDF, etc.).
├── requirements.txt             <- Python dependencies.
└── src                          <- Source code for the project.
    ├── __init__.py
    ├── data
    │   ├── make_dataset.py
    │   └── data_ingestor.py
    ├── features
    │   └── build_features.py      <- Data cleaning and feature engineering scripts.
    ├── models
    │   ├── cluster_analysis.ipynb <- Notebook for cluster analysis.
    │   └── train_model.py         <- Script to train KMeans clustering model.
    └── visualization
        ├── EDA.ipynb              <- Exploratory data analysis notebook.
        └── plot_settings.py       <- Custom visualization settings.
```

## Dataset

The project uses the **Online Retail II** dataset containing transactions for a UK-based online retail business between December 2009 and December 2011. The dataset includes the following key variables:
- **Invoice**: Invoice number (unique per transaction).
- **StockCode**: Product code.
- **Description**: Product name.
- **Quantity**: Number of items per transaction.
- **InvoiceDate**: Date and time of the transaction.
- **Price**: Unit price in GBP.
- **CustomerID**: Unique customer identifier.
- **Country**: Customer country.

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/SebastianGarrido2790/online-retail-customer-segmentation.git
cd online-retail-customer-segmentation
```

2. **Create and activate a virtual environment (optional but recommended):**

```bash
python -m venv env
env\Scripts\activate  # On Mac source env/bin/activate
```

3. **Install required packages:**

```bash
pip install -r requirements.txt
```

## Usage

### Data Cleaning and Feature Engineering

- **Script:** `src/features/build_features.py`
- **Description:** This script reads the raw data, converts data types (e.g., parsing `InvoiceDate` as datetime), renames columns (e.g., `Customer ID` to `CustomerID`), calculates `TotalSales`, and creates RFM metrics. Outliers in `Frequency` and `Monetary` are removed using an IQR-based method.
- **Output:** Cleaned dataset saved to `data/processed/online_retail_2009_2010_without_outliers.csv`.

### Cluster Analysis

- **Notebook:** `src/models/cluster_analysis.ipynb`
- **Description:** This notebook:
  - Loads the cleaned RFM data.
  - Applies KMeans clustering.
  - Uses the Elbow Method and silhouette scores to determine the optimal number of clusters.
  - Visualizes the clustering results (cluster distribution, RFM averages, PCA projections, etc.).
- **Output:** Cluster labels added to the dataset and various plots to interpret customer segments.

### Visualization

- **Directory:** `src/visualization/`
- **Description:** Contains scripts and notebooks for generating visualizations. Custom plot settings are defined in `plot_settings.py` for consistent style across figures.

## Outlier Removal

Outliers in the `Frequency` and `Monetary` features are removed using the Interquartile Range (IQR) method. The function `remove_outliers()` in `src/features/build_features.py` applies this method and saves the resulting dataset to the `data/processed` folder.

## Future Work

- **Feature Expansion:** Incorporate additional features such as customer demographics or web browsing behavior.
- **Model Improvements:** Experiment with alternative clustering algorithms (e.g., DBSCAN, hierarchical clustering) and compare their performance.
- **Visualization Enhancements:** Develop interactive dashboards (e.g., using Plotly or Dash) to explore customer segments in real time.

## License

This project is licensed under the [MIT License](LICENSE).
