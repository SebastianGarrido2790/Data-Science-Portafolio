# --------------------------------
# 1. Import Necessary Dependencies
# --------------------------------

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Absolute imports
from src.config.logger import setup_logging
from src.config.factories import EmbeddingFactory, SummaryFactory

import json
import yaml
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import xgboost as xgb
import joblib
import ast
import datetime
import matplotlib.pyplot as plt

# --------------------------------
# 2. Define Paths Concisely
# --------------------------------

# Define Paths Concisely
PATHS = {
    "config": os.path.join(project_root, "config.yml"),
    "raw_data": os.path.join(project_root, "data", "raw", "customer_churn.csv"),
    "summaries": os.path.join(
        project_root, "data", "interim", "customer_churn_summary.csv"
    ),
    "embeddings": os.path.join(
        project_root, "data", "processed", "customer_churn_summary_embeddings.csv"
    ),
    "models": os.path.join(project_root, "models"),
    "roc_fig": os.path.join(project_root, "reports", "figures", "roc_curve.png"),
    "feat_fig": os.path.join(
        project_root, "reports", "figures", "feature_importance.png"
    ),
}

for path in ["summaries", "embeddings", "models", "roc_fig", "feat_fig"]:
    os.makedirs(os.path.dirname(PATHS[path]), exist_ok=True)

# --------------------------------
# 3. Load Configuration
# --------------------------------

with open(PATHS["config"], "r") as file:
    config = yaml.safe_load(file)

embedding_config = config["embedding"]
summary_config = config["summary"]
data_config = config["data"]
model_config = config["model"]
logging_config = config["logging"]

# --------------------------------
# 4. Setup Logging
# --------------------------------

setup_logging(logging_config)
logger = logging.getLogger(__name__)
# New logging line to mark script start
logger.info(
    "Script execution started at %s",
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
)

# --------------------------------
# 5. Setup Factories
# --------------------------------

# Embedding factory
provider = embedding_config["embedding_model"]
try:
    llm = EmbeddingFactory(provider)
    logger.info(f"Initialized EmbeddingFactory with provider: {provider}")
except ValueError as e:
    logger.error(f"Failed to initialize EmbeddingFactory: {e}")
    raise

# Summary factory
summary_provider = summary_config["summary_provider"]
try:
    summary_llm = SummaryFactory(summary_provider)
    logger.info(f"Initialized SummaryFactory with provider: {summary_provider}")
except ValueError as e:
    logger.error(f"Failed to initialize SummaryFactory: {e}")
    raise

# --------------------------------
# 6. Load Dataset
# --------------------------------

logger.info("Loading dataset...")
df = pd.read_csv(PATHS["raw_data"])

# --------------------------------
# 7. Generate Summaries with LLM
# --------------------------------

if os.path.exists(PATHS["summaries"]):
    logger.info("Loading existing summaries...")
    df = pd.read_csv(PATHS["summaries"])
else:
    logger.info("Generating ticket summaries (this may take time)...")
    # Batch processing (reduces the overhead of processing each row individually)
    batch_size = data_config.get("batch_size", 10)  # Default to 10 if not specified
    summaries = []
    for i in range(0, len(df), batch_size):
        batch = df["ticket_notes"][i : i + batch_size].tolist()
        logger.info(
            f"Processing batch {i // batch_size + 1}/{(len(df) // batch_size) + 1}"
        )
        batch_summaries = [summary_llm.summarize(text) for text in batch]
        summaries.extend(batch_summaries)
    df["ticket_summary"] = summaries
    df.to_csv(PATHS["summaries"], index=False)
    logger.info(f"Summaries saved to {PATHS['summaries']}")

# Inspect summaries
df[["ticket_notes", "ticket_summary"]]

# --------------------------------
# 8. Get Embeddings for Summaries
# --------------------------------

if os.path.exists(PATHS["embeddings"]):
    logger.info("Loading existing embeddings...")
    df = pd.read_csv(PATHS["embeddings"])
    df["summary_embedding"] = df["summary_embedding"].apply(ast.literal_eval)
else:
    logger.info("Generating embeddings (this may take time)...")
    df["summary_embedding"] = df["ticket_summary"].apply(llm.get_embeddings)
    df = df[df["summary_embedding"].notnull()]
    df.to_csv(PATHS["embeddings"], index=False)
    logger.info(f"Embeddings saved to {PATHS['embeddings']}")

# --------------------------------
# 9. Prepare Features
# --------------------------------

logger.info("Preparing features...")

le = LabelEncoder()
df["plan_type_encoded"] = le.fit_transform(df["plan_type"])
joblib.dump(le, os.path.join(PATHS["models"], "label_encoder.pkl"))

embeddings_df = pd.DataFrame(df["summary_embedding"].tolist(), index=df.index)

X_df = pd.concat(
    [df[["age", "tenure", "spend_rate", "plan_type_encoded"]], embeddings_df], axis=1
)
X_df.columns = X_df.columns.astype(str)
y = df["churn"].values

# Save feature names
with open(os.path.join(PATHS["models"], "feature_names.txt"), "w") as f:
    for feature in X_df.columns:
        f.write(f"{feature}\n")

# --------------------------------
# 10. Split Data
# --------------------------------

logger.info("Splitting data...")
stratify = y if data_config["stratify"] else None
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_df,
    y,
    test_size=data_config["test_size"],
    stratify=stratify,
    random_state=model_config["random_state"],
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    stratify=y_train_full if data_config["stratify"] else None,
    random_state=model_config["random_state"],
)

numeric_features = ["age", "tenure", "spend_rate"]
if model_config["scale_features"]:
    logger.info("Scaling numeric features...")
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_val[numeric_features] = scaler.transform(X_val[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    joblib.dump(scaler, os.path.join(PATHS["models"], "scaler.pkl"))
else:
    logger.info("Skipping feature scaling")

# --------------------------------
# 11. Train XGBoost Model
# --------------------------------

logger.info("Training XGBoost model...")
model_params = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": model_config["learning_rate"],
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": model_config["random_state"],
}
model = XGBClassifier(**model_params)
eval_set = [(X_val, y_val)]

try:
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        early_stopping_rounds=model_config["early_stopping_rounds"],
        verbose=True,
    )
except TypeError as e:
    logger.error(f"Early stopping failed: {e}")
    logger.info("Falling back to training without early stopping.")
    model.fit(X_train, y_train, verbose=True)  # Fallback

# --------------------------------
# 12. Evaluate Model
# --------------------------------

logger.info("Evaluating model on test set...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")

plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color="blue", label=f"ROC AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(PATHS["roc_fig"])
plt.show()

# --------------------------------
# 13. Interpret Model
# --------------------------------

# Convert model to booster
booster = model.get_booster()

# Plot feature importance
logger.info("Plotting feature importance...")
plt.figure(figsize=(12, 6))
xgb.plot_importance(booster)
plt.savefig(PATHS["feat_fig"])
plt.tight_layout()
plt.show()

importances = model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": importances}
).sort_values("Importance", ascending=False)
importance_df.head(10)

# Inspect most relevant feature
X_df["65"]
top_feature_importance = df.copy()
top_feature_importance["65"] = X_df["65"]
top_feature_importance.sort_values(by="65", ascending=False).head(10)

# --------------------------------
# 14. Save Model and Metadata
# --------------------------------

logger.info("Saving model and assets for deployment...")
save_format = model_config.get("save_format", "json")
if save_format not in ["json", "pickle", "both"]:
    logger.error(f"Invalid save_format: {save_format}. Using 'json'.")
    save_format = "json"

if save_format in ["json", "both"]:
    model.save_model(os.path.join(PATHS["models"], "xgb_churn_model.json"))
if save_format in ["pickle", "both"]:
    joblib.dump(model, os.path.join(PATHS["models"], "xgb_churn_model.pkl"))

metadata = {
    "model_save_format": save_format,
    "feature_names": list(X_train.columns),
    "model_params": model.get_params(),
    "config": config,
    "embedding_dim": len(df["summary_embedding"].iloc[0]),
}
with open(os.path.join(PATHS["models"], "metadata.json"), "w") as f:
    json.dump(metadata, f)

logger.info(f"Model, scaler, and metadata saved in {PATHS['models']}")

# -------------------------------------------
# 15. Load Model and Metadata for Deployment
# -------------------------------------------

logger.info("Loading model and assets for deployment...")

# Load the model (choose format based on preference or config)
model = xgb.XGBClassifier()
model.load_model(os.path.join(PATHS["models"], "xgb_churn_model.json"))

# Load scaler
scaler = joblib.load(os.path.join(PATHS["models"], "scaler.pkl"))

# Load feature names
with open(os.path.join(PATHS["models"], "feature_names.txt"), "r") as f:
    feature_names = [line.strip() for line in f]

# Generate actual embedding for new data
new_ticket_notes = "Customer reported a billing issue that was resolved quickly."
logger.info("Generating summary for new data...")
summary = summary_llm.summarize(new_ticket_notes)
logger.info("Generating embedding for new data summary...")
summary_embedding = llm.get_embeddings(summary)

# Prepare new data
new_data = pd.DataFrame(
    {
        "age": [50],
        "tenure": [10],
        "spend_rate": [100],
        "plan_type_encoded": [1],
    }
)

# Scale numeric features
numeric_features = ["age", "tenure", "spend_rate"]
new_data_scaled = new_data.copy()
new_data_scaled[numeric_features] = scaler.transform(new_data[numeric_features])

# Add the summary_embedding as separate columns
embedding_dim = len(summary_embedding)
embedding_columns = [f"{i}" for i in range(embedding_dim)]
embedding_df = pd.DataFrame([summary_embedding], columns=embedding_columns)
new_data_final = pd.concat([new_data_scaled, embedding_df], axis=1)

# Ensure columns match training data
new_data_final = new_data_final[feature_names]

# Make prediction
prediction = model.predict(new_data_final)
logger.info(f"Prediction for new data: {prediction[0]}")
