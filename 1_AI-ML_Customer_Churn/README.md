# Churn Prediction with AI-enhanced Text Summarization and Embeddings

This repository implements a complete machine learning pipeline for predicting customer churn. The pipeline leverages advanced AI techniques—including text summarization and embeddings—to extract meaningful information from customer ticket notes, which are then used alongside numerical features to train an XGBoost classifier.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Deploying the Model](#deploying-the-model)
- [Pipeline Details](#pipeline-details)
  - [Data Preparation](#data-preparation)
  - [Text Summarization and Embeddings](#text-summarization-and-embeddings)
  - [Feature Engineering](#feature-engineering)
  - [Model Training and Evaluation](#model-training-and-evaluation)
- [Logging and Monitoring](#logging-and-monitoring)
- [License](#license)

---

## Overview

This project aims to predict customer churn by incorporating both structured data (e.g., age, tenure, spending rate, plan type) and unstructured text data from customer support tickets. The key innovation lies in:
- **Text Summarization:** Using AI (either Hugging Face or OpenAI) to condense long ticket notes into concise summaries.
- **Embeddings:** Transforming summaries into numerical representations via pre-trained language models.
- **Modeling:** Integrating these enriched features into an XGBoost classifier that is trained, evaluated, and saved for deployment.

---

## Project Structure

```plaintext
├── LICENSE
├── README.md          <- This file, providing project overview and usage instructions.
├── data
│   ├── external       <- Data from third-party sources.
│   ├── interim        <- Intermediate data, including generated summaries.
│   ├── processed      <- Final datasets with embeddings ready for modeling.
│   └── raw            <- Original, immutable data dumps.
│
├── docs               <- Sphinx documentation files.
│
├── models             <- Trained models, predictions, and model summaries.
│
├── notebooks          <- Jupyter notebooks for exploratory analysis.
│
├── references         <- Data dictionaries and supplementary materials.
│
├── reports            <- Analysis reports in HTML, PDF, LaTeX, etc.
│   └── figures        <- Figures generated during analysis.
│
├── .env
├── .gitignore
├── config.yml         <- Main configuration file.
├── environment.yml    <- Conda environment configuration.
├── requirements.txt   <- Pip dependencies.
│
└── src                <- Source code for the project.
    ├── __init__.py
    ├── config
    │   ├── __init__.py
    │   ├── factories.py      <- Factories for initializing embeddings and summarization clients.
    │   ├── logger.py         <- Custom logging setup.
    │   └── settings.py       <- Environment-specific settings using pydantic.
    │
    ├── data
    │   └── make_dataset.py   <- Data download and preprocessing scripts.
    │
    ├── features
    │   └── build_features.py <- Scripts for feature engineering.
    │
    ├── models
    │   ├── train_model.py    <- Model training pipeline integrating text summarization and embeddings.
    │   └── predict_model.py  <- Model inference and prediction scripts.
    │
    └── visualization
        └── visualize.py      <- Scripts for generating visualizations.
```

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/SebastianGarrido2790/churn-project.git
cd churn-project
```

2. **Setup the Conda environment:**
```bash
conda env create -f environment.yml
conda activate churn-project-env
```

3. **Install additional Python packages:**
```bash
pip install -r requirements.txt
```

4. **Configure Environment Variables:**

Create a `.env` file in the root directory and add your API keys if using OpenAI:
```dotenv
OPENAI_API_KEY=your_openai_api_key
```

---

## Configuration

The project configuration is managed via `config.yml` and `environment.yml`. Key configuration sections include:

- **Embedding Settings:** Choose between `huggingface` and `openai` for generating text embeddings.
- **Summarization Settings:** Select the summarization provider and set model-specific parameters (e.g., model name, token limits, temperature).
- **Data Settings:** Configure test sizes, stratification, and batch processing parameters.
- **Model Settings:** Define XGBoost parameters, early stopping rounds, learning rate, and save formats.
- **Logging Settings:** Set logging levels, formats, and log file paths.

For advanced configuration (e.g., LLM provider settings), see `src/config/settings.py`.

---

## Usage

### Training the Model

Run the training pipeline by executing:
```bash
python src/models/train_model.py
```
This script will:
- Load the raw customer churn dataset.
- Generate text summaries for customer ticket notes (or load existing ones).
- Compute text embeddings for the summaries.
- Prepare numerical and textual features.
- Train an XGBoost model using early stopping.
- Evaluate the model using classification metrics and ROC curves.
- Save the trained model, scaler, and metadata for deployment.

### Deploying the Model

After training, the model is saved in both JSON and pickle formats (depending on the `save_format` specified in `config.yml`). To perform predictions:
- Load the model and scaler from the `models` directory.
- Prepare new input data by generating summaries and embeddings.
- Scale the numerical features and align them with the training feature set.
- Use the model to predict churn for new customer data.

Refer to the deployment section at the end of `train_model.py` for detailed code on model loading and inference.

---

## Pipeline Details

### Data Preparation

- **Raw Data:** The original customer churn CSV file is stored under `data/raw`.
- **Interim Data:** Summaries generated by the AI summarization provider are saved under `data/interim`.
- **Processed Data:** Data enriched with text embeddings is saved under `data/processed`.

### Text Summarization and Embeddings

- **Summarization:** Depending on your configuration (`config.yml`), the pipeline uses either Hugging Face or OpenAI for text summarization.
- **Embeddings:** The summarization output is converted into embeddings using a pre-trained language model (e.g., `sentence-transformers/all-MiniLM-L6-v2`).

### Feature Engineering

- **Label Encoding:** Categorical features such as `plan_type` are encoded.
- **Scaling:** Numerical features (`age`, `tenure`, `spend_rate`) are scaled using `StandardScaler`.
- **Feature Concatenation:** The pipeline combines both the original numerical features and the generated embedding features for model training.

### Model Training and Evaluation

- **Model:** An XGBoost classifier is used with configurable parameters such as `n_estimators`, `max_depth`, and `learning_rate`.
- **Early Stopping:** The model supports early stopping to prevent overfitting.
- **Evaluation:** The model is evaluated using classification reports and ROC curves. Feature importance is visualized to interpret model decisions.

---

## Logging and Monitoring

The project utilizes a custom logging setup defined in `src/config/logger.py`:
- Logs are output to both the console and a file.
- Log levels, formats, and file paths are configurable via `config.yml`.

This ensures that all key steps, from data processing to model evaluation, are tracked for debugging and audit purposes.

---

## License

This project is licensed under the terms of the [LICENSE](./LICENSE.txt) file. Ensure you comply with the licensing agreements when using or modifying the code.

---

## Acknowledgements

This project integrates state-of-the-art AI methods for text summarization and embeddings, leveraging libraries from Hugging Face, OpenAI, and the broader Python ecosystem for data science and machine learning.

---

## Contact

For any questions or support, please open an issue in the repository or contact the project maintainer at [sebastiangarrido2790@gmail.com].
