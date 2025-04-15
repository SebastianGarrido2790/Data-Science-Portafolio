# Sentiment Analysis of Amazon Alexa Reviews

This project performs sentiment analysis on Amazon Alexa reviews using a fine-tuned DistilBERT model, Random Forest, and XGBoost. The goal is to classify reviews as positive or negative, with a focus on achieving high recall for negative reviews (achieved: recall 0.82, precision 0.44 at threshold 0.95). The project includes scripts for data preprocessing, model training, prediction, and deployment via a Flask API.

## Project Overview

- **Objective**: Classify Amazon Alexa reviews as positive or negative using a DistilBERT model.
- **Model**: Fine-tuned DistilBERT with a custom threshold (0.95) to prioritize negative recall.
- **Performance**:
  - Negative recall: 0.82
  - Negative precision: 0.44
  - Threshold: 0.95
- **Dataset**: Amazon Alexa reviews (see `docs/about_dataset.txt` for details).
- **Hardware**: Developed on an Intel Celeron N4020 with 8 GB RAM (CPU-only inference).

## Folder Structure

```plaintext
├── LICENSE                    # Project license
├── README.md                  # The top-level README for developers using this project
├── data
│   ├── external               # Data from third-party sources
│   ├── interim                # Intermediate data that has been transformed
│   ├── processed              # The final, canonical datasets for modeling
│   └── raw                    # The original, immutable data dump
│
├── docs                       # Documentation files
│   ├── about_dataset.txt      # Description of the dataset
│   └── project_notes.txt      # Comments on the steps taken throughout the project
│
├── models                     # Trained models, prediction scripts, and API
│   ├── distilbert_model       # Saved DistilBERT model and tokenizer
│   ├── deploy_model.py        # Flask API for sentiment prediction (supports batch inputs)
│   └── test_api_batch.py      # Script to test the Flask API with batch inputs
│
├── references                 # Data dictionaries, manuals, and explanatory materials
│
├── reports                    # Generated analysis reports
│   └── figures                # Generated graphics and figures for reporting
│
├── environment.yml            # Conda environment file for reproducing the analysis environment
│
├── src                        # Source code for the project
│   ├── __init__.py            # Makes src a Python module
│   │
│   ├── data                   # Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features               # Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models                 # Scripts for training and prediction
│   │   ├── predict_model.py   # Script to predict sentiment on new texts
│   │   └── train_model.py     # Script to train the DistilBERT model
│   │
│   └── visualization          # Scripts for exploratory and results-oriented visualizations
│       └── visualize.py
```

## Prerequisites

- **Operating System**: Windows 10/11 (tested on Windows 10, version 10.0.22631.5039)
- **Hardware**: Minimum 8 GB RAM, CPU (e.g., Intel Celeron N4020)
- **Software**:
  - Python 3.9
  - Conda (for environment management)
  - Git (optional, for cloning the repository)

## Setup Instructions

### 1. Clone the Repository (Optional)

If the project is hosted on a Git repository, clone it:

### 2. Set Up the Conda Environment

Install Conda (if not already installed):

- Download and install Miniconda or Anaconda from [conda.io](https://conda.io/)

Create the environment:

Activate the environment:

Update dependencies if needed:

### 3. Prepare the Dataset

Place the raw dataset in `data/raw/` and run the preprocessing scripts:

This generates processed data in `data/processed/`.

### 4. Train the Model

To train the DistilBERT model:

The trained model will be saved in `models/distilbert_model/`.

**Note**: Training on a CPU (e.g., Intel Celeron N4020) may take several hours.

## Making Predictions

### Option 1: Using the Prediction Script

Run the script to predict sentiment:

**Example Output:**

Alternatively, import the function in your code:

### Option 2: Using the Flask API

Start the API:

The API will start at `http://127.0.0.1:5000`.

Test the API:

**Example JSON Output:**

Alternatively, use `test_api_batch.py`:

## Documentation

- **Dataset Details**: See `docs/about_dataset.txt`.
- **Project Notes**: See `docs/project_notes.txt`.

## Performance Notes

- **Inference Speed**: \~2–3 seconds per batch (Intel Celeron N4020, 8 GB RAM)
- **Memory Usage**: \~1–2 GB RAM during inference
- **Scalability**: Suitable for low-traffic scenarios; for production, use a more powerful server.

## License

See the `LICENSE` file for details.

## Contact

For issues or inquiries, open an issue on the repository or contact the project maintainer at `[sebastiangarrido2790@gmail.com]`.
