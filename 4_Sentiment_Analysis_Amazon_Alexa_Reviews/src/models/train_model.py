import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    recall_score,
    precision_score,
)
from xgboost import XGBClassifier
import pickle

# from transformers import pipeline
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset
import torch
import accelerate
import transformers
from transformers.utils import is_accelerate_available

print("Torch version:", torch.__version__)  # Should be 2.5.0
print("Accelerate version:", accelerate.__version__)  # Should be 1.5.2
print("Transformers version:", transformers.__version__)  # Should be 4.49.0
print("Accelerate available:", is_accelerate_available())  # Should be True

df = pd.read_csv("../../data/processed/preprocessed_data.csv")

# Split data
X = df["processed_reviews"]
y = df["feedback"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"X train: {X_train.shape}")
print(f"y train: {y_train.shape}")
print(f"X test: {X_test.shape}")
print(f"y test: {y_test.shape}")

# Calculate class weights for imbalance
neg_count = len(y[y == 0])
pos_count = len(y[y == 1])
class_weight_dict = {0: pos_count / neg_count, 1: 1.0}

print(f"Class distribution - Negative: {neg_count}, Positive: {pos_count}")
print(f"Class weights: {class_weight_dict}")

# Vectorize text
vectorizer = CountVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Save vectorizer
with open("../../models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


# ---------------------------
# 1. RandomForestClassifier
# ---------------------------

print("\nTraining RandomForestClassifier...")
# class_weight_dict_rf = {0: 20.0, 1: 1.0}  # Double the weight for negatives
rf_clf = RandomForestClassifier(
    n_estimators=100, class_weight=class_weight_dict, random_state=42, n_jobs=-1
)

# Hyperparameter tuning
rf_param_grid = {
    "n_estimators": [200, 300, 400],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 7, 9],
    "min_samples_leaf": [1, 2, 4, 6],
    "max_features": ["sqrt", "log2", None],
}

rf_grid = RandomizedSearchCV(
    rf_clf,
    rf_param_grid,
    cv=StratifiedKFold(n_splits=5),
    scoring="f1_weighted",
    n_iter=20,
    random_state=42,
)
rf_grid.fit(X_train_vec, y_train)

rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test_vec)
print("Random Forest Best Params:", rf_grid.best_params_)
# print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))

# Threshold tuning (default is 0.5)
rf_probs = rf_best.predict_proba(X_test_vec)[:, 1]  # Probability of class 1
for threshold in [0.4, 0.45, 0.5, 0.55, 0.6]:
    rf_pred_adjusted = (rf_probs >= threshold).astype(int)
    print(f"\nThreshold: {threshold}")
    print(classification_report(y_test, rf_pred_adjusted))

# Recall tuning
thresholds = np.arange(0.1, 1.0, 0.05)
rf_recall_scores = [
    recall_score(y_test, (rf_probs >= t).astype(int), pos_label=0) for t in thresholds
]
best_recall_threshold = thresholds[np.argmax(rf_recall_scores)]
print(f"\nBest threshold for negative recall: {best_recall_threshold}")

# ROC-AUC/PR-AUC
rf_probs = rf_best.predict_proba(X_test_vec)[:, 1]
print("\nROC-AUC:", roc_auc_score(y_test, rf_probs))
precision, recall, _ = precision_recall_curve(y_test, rf_probs)
print("PR-AUC:", auc(recall, precision))

# Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)
print("\nConfusion Matrix:\n", cm)

# Save RF model
# with open("../../models/rf_model.pkl", "wb") as f:
#     pickle.dump(rf_best, f)


# ------------------
# 2. XGBClassifier
# ------------------

print("\nTraining XGBClassifier...")
xgb_clf = XGBClassifier(
    scale_pos_weight=class_weight_dict[0],  # 10.70703125
    # scale_pos_weight=20.0,
    eval_metric="logloss",
    random_state=42,
)

# Hyperparameter tuning
xgb_param_grid = {
    "n_estimators": [200, 300, 400],
    "max_depth": [10, 12, 14, None],
    "learning_rate": [0.1, 0.2, 0.4],
    "gamma": [0, 0.1, 0.2],
    "reg_alpha": [0, 0.1, 0.2],
    "reg_lambda": [1, 1.2, 1.4],
}

xgb_grid = RandomizedSearchCV(
    xgb_clf,
    xgb_param_grid,
    cv=StratifiedKFold(n_splits=5),
    scoring="f1_weighted",
    n_iter=20,
    random_state=42,
)
xgb_grid.fit(X_train_vec, y_train)

xgb_best = xgb_grid.best_estimator_
xgb_pred = xgb_best.predict(X_test_vec)
print("XGBoost Best Params:", xgb_grid.best_params_)
# print("XGBoost Classification Report:\n", classification_report(y_test, xgb_pred))

# Threshold tuning (default is 0.5)
xgb_probs = xgb_best.predict_proba(X_test_vec)[:, 1]
for threshold in [0.4, 0.45, 0.5, 0.55, 0.6]:
    xgb_pred_adjusted = (xgb_probs >= threshold).astype(int)
    print(f"\nThreshold: {threshold}")
    print(classification_report(y_test, xgb_pred_adjusted))

# Recall tuning
thresholds = np.arange(0.1, 1.0, 0.05)
xgb_recall_scores = [
    recall_score(y_test, (xgb_probs >= t).astype(int), pos_label=0) for t in thresholds
]
best_recall_threshold = thresholds[np.argmax(xgb_recall_scores)]
print(f"\nBest threshold for negative recall: {best_recall_threshold}")

# ROC-AUC/PR-AUC
xg_probs = xgb_best.predict_proba(X_test_vec)[:, 1]
print("\nROC-AUC:", roc_auc_score(y_test, xg_probs))
precision, recall, _ = precision_recall_curve(y_test, xg_probs)
print("PR-AUC:", auc(recall, precision))

# Confusion Matrix
cm = confusion_matrix(y_test, xgb_pred)
print("\nConfusion Matrix:\n", cm)

# Save XGB model
# with open("../../models/xgb_model.pkl", "wb") as f:
#     pickle.dump(xgb_best, f)

# Winner: Random Forest (Latest, Threshold 0.75, Weight 10.70703125)
rf_pred_075 = (rf_probs >= 0.75).astype(int)
print(classification_report(y_test, rf_pred_075))
print(confusion_matrix(y_test, rf_pred_075))

# Save RF model
with open("../../models/rf_model_0.75.pkl", "wb") as f:
    pickle.dump(rf_best, f)


# ----------------------------------
# 3. Transformer Model (DistilBERT)
# ----------------------------------

print("\nTraining Transformer Model (DistilBERT)...")
print("GPU Available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

# Class weights
neg_count = len(y[y == 0])  # 256
pos_count = len(y[y == 1])  # 2741
class_weight_dict = {0: pos_count / neg_count, 1: 1.0}  # {0: 10.70703125, 1: 1.0}
print(f"Class distribution - Negative: {neg_count}, Positive: {pos_count}")
print(f"Class weights: {class_weight_dict}")


# Custom Dataset Class
class SentimentDataset(Dataset):
    """A custom PyTorch Dataset for sentiment analysis with pre-tokenized text data.

    This class prepares text data and corresponding labels for training or evaluation
    with a transformer model (e.g., DistilBERT). It pre-tokenizes all texts during
    initialization to reduce runtime overhead, ensuring compatibility with batched
    training pipelines like Hugging Face's Trainer. It is optimized for memory-constrained
    environments by controlling the maximum sequence length.

    Attributes:
        encodings (dict): Pre-tokenized encodings containing 'input_ids' and
            'attention_mask' as torch.Tensor, shape [n_samples, max_length].
        labels (list or array-like): The list of sentiment labels (e.g., 0 or 1).
    """

    def __init__(self, texts, labels, tokenizer, max_length=128):
        """Initialize the SentimentDataset with pre-tokenized texts and labels.

        Args:
            texts (list or array-like): Input text samples (e.g., reviews).
            labels (list or array-like): Corresponding labels (e.g., 0 for negative,
                1 for positive).
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer instance (e.g.,
                DistilBertTokenizer) to convert text to token IDs and attention masks.
            max_length (int, optional): Maximum sequence length for tokenization.
                Defaults to 128.

        Notes:
            - Pre-tokenizes all texts upfront, which may increase memory usage but
              reduces per-sample tokenization overhead during training.
            - Assumes texts are valid strings; invalid entries (e.g., NaN) should be
              preprocessed before passing.
        """
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",  # Pads shorter sequences with zeros up to 128 tokens
            truncation=True,  # Cuts off any tokens beyond 128 to fit the limit
            return_tensors="pt",  # PyTorch tensors (not lists or NumPy arrays), matching transformers’ expectations
        )
        self.labels = labels

    def __len__(self):
        """Return the total number of samples in the dataset.

        Returns:
            int: The length of the labels list (number of samples).
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """Retrieve a single pre-tokenized sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'input_ids' (torch.Tensor): Tokenized input IDs, shape [max_length].
                - 'attention_mask' (torch.Tensor): Attention mask, shape [max_length].
                - 'labels' (torch.Tensor): Sentiment label as a long tensor.

        Notes:
            - Uses pre-tokenized encodings, so no additional tokenization is performed.
            - Ensures consistent sequence lengths via pre-applied padding and truncation.
        """
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# Load model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
model.to(device)

# Prepare datasets
train_dataset = SentimentDataset(X_train.tolist(), y_train.tolist(), tokenizer)
test_dataset = SentimentDataset(X_test.tolist(), y_test.tolist(), tokenizer)

# Class weights for imbalance
class_weights = torch.tensor([class_weight_dict[0], 1.0]).to(device)


# Custom Trainer with Weighted Loss
class WeightedTrainer(Trainer):
    """A custom Trainer class with weighted loss for imbalanced sentiment classification.

    This class extends the Hugging Face Trainer to incorporate class weights in the
    loss computation, addressing imbalance in sentiment datasets (e.g., more positive
    than negative reviews). It overrides the compute_loss method to use a weighted
    CrossEntropyLoss, optimizing for metrics like negative recall.

    Attributes:
        Inherited from transformers.Trainer, plus any additional args passed to Trainer.
    """

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute the weighted loss for a batch of inputs.

        Args:
            model (transformers.PreTrainedModel): The model being trained (e.g.,
                DistilBertForSequenceClassification).
            inputs (dict): A dictionary containing batched input data, including:
                - 'input_ids' (torch.Tensor): Tokenized input IDs.
                - 'attention_mask' (torch.Tensor): Attention masks.
                - 'labels' (torch.Tensor): Ground truth labels.
            return_outputs (bool, optional): If True, return both loss and model outputs.
                Defaults to False.
            num_items_in_batch (int, optional): Number of items in the batch, passed by
                Trainer.training_step in newer transformers versions. Ignored here.
                Defaults to None.

        Returns:
            torch.Tensor or tuple: If return_outputs=False, returns the weighted loss
                (torch.Tensor). If return_outputs=True, returns a tuple of (loss,
                outputs), where outputs is the model's forward pass result.

        Notes:
            - Assumes class_weights is defined in the outer scope as a torch.Tensor
              with weights for each class (e.g., [weight_neg, weight_pos]).
            - Assumes device is defined in the outer scope (e.g., "cpu").
            - Moves inputs and labels to the specified device before computation.
        """
        # num_items_in_batch is ignored since we don't need it for our loss
        labels = inputs.pop("labels").to(device)
        outputs = model(**{k: v.to(device) for k, v in inputs.items()})
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Training Arguments (Optimized for 8 GB RAM CPU)
training_args = TrainingArguments(
    output_dir="./distilbert_results",
    num_train_epochs=2,
    per_device_train_batch_size=8,  # Reduced for low RAM
    per_device_eval_batch_size=8,  # Reduced for low RAM
    gradient_accumulation_steps=1,  # Simulate batch size 8
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./distilbert_logs",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="recall_neg",  # Optimize for negative recall
    greater_is_better=True,
    use_cpu=True,  # Explicitly force CPU usage
)


# Compute Metrics including Negative Recall and Precision
def compute_metrics(pred):
    """Compute evaluation metrics for sentiment classification predictions.

    This function calculates accuracy, negative recall, and negative precision from
    model predictions, tailored for evaluating a binary sentiment classifier with
    an emphasis on the negative class (label 0). It is designed to be passed to
    Hugging Face's Trainer for metric computation during training and evaluation.

    Args:
        pred (transformers.trainer_utils.EvalPrediction): Prediction object containing:
            - label_ids (np.ndarray): Ground truth labels (e.g., [0, 1, 0, ...]).
            - predictions (np.ndarray): Model logits or probabilities, shape [n_samples, 2].

    Returns:
        dict: A dictionary with the following metrics:
            - 'accuracy' (float): Overall accuracy across all samples.
            - 'recall_neg' (float): Recall for the negative class (label 0).
            - 'precision_neg' (float): Precision for the negative class (label 0).

    Notes:
        - Assumes binary classification with labels 0 (negative) and 1 (positive).
        - Uses zero_division=0 in precision_score to handle cases with no predicted
          negatives, returning 0 instead of raising an error.
        - predictions are converted from logits to class predictions using argmax.
    """
    labels = pred.label_ids  # NumPy array of true labels
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "recall_neg": recall_score(labels, preds, pos_label=0),
        "precision_neg": precision_score(labels, preds, pos_label=0, zero_division=0),
    }


# Initialize Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Predictions and Threshold Tuning with Precision Constraint
predictions = trainer.predict(test_dataset)
probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1)[
    :, 1
].numpy()  # Probabilities for positive class
transformer_pred_default = predictions.predictions.argmax(-1)  # Default threshold 0.5
print(
    "Transformer Classification Report (Default Threshold 0.5):\n",
    classification_report(y_test, transformer_pred_default),
)
print(
    "Confusion Matrix (Default Threshold 0.5):\n",
    confusion_matrix(y_test, transformer_pred_default),
)

# Threshold Tuning for Negative Recall with Minimum Precision 0.3
thresholds = np.arange(0.1, 1.0, 0.05)
recall_scores = [
    recall_score(y_test, (probs >= t).astype(int), pos_label=0) for t in thresholds
]
precision_scores = [
    precision_score(y_test, (probs >= t).astype(int), pos_label=0, zero_division=0)
    for t in thresholds
]

# Find best threshold with precision >= 0.3
valid_thresholds = [
    (t, r, p)
    for t, r, p in zip(thresholds, recall_scores, precision_scores)
    if p >= 0.3
]
if valid_thresholds:
    best_recall_threshold, best_recall, best_precision = max(
        valid_thresholds, key=lambda x: x[1]
    )  # Maximize recall
    print(
        f"\nBest threshold for negative recall (precision >= 0.3): {best_recall_threshold}"
    )
    print(f"Best negative recall: {best_recall:.2f}, Precision: {best_precision:.2f}")
else:
    print("\nNo threshold found with precision >= 0.3")

# Metrics at Best Threshold
if valid_thresholds:
    transformer_pred_adjusted = (probs >= best_recall_threshold).astype(int)
    print(
        f"\nTransformer Classification Report (Threshold {best_recall_threshold}):\n",
        classification_report(y_test, transformer_pred_adjusted),
    )
    print(
        f"Confusion Matrix (Threshold {best_recall_threshold}):\n",
        confusion_matrix(y_test, transformer_pred_adjusted),
    )

# ROC-AUC and PR-AUC
print("\nROC-AUC:", roc_auc_score(y_test, probs))
precision, recall, _ = precision_recall_curve(y_test, probs)
print("PR-AUC:", auc(recall, precision))

# Save the model
model.save_pretrained("../../models/distilbert_model")
tokenizer.save_pretrained("../../models/distilbert_model")
