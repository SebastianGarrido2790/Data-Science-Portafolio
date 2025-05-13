import torch
import argparse
from typing import List, Dict, Union
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd

# Constants
MODEL_PATH = "../../models/distilbert_model"
BEST_THRESHOLD = 0.95

# Load model and tokenizer with error handling
try:
    print("Loading model and tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model or tokenizer from {MODEL_PATH}: {str(e)}")

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)
model.eval()  # Set to evaluation mode for inference


def predict_sentiment(
    texts: Union[str, List[str]],
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizer,
    threshold: float = BEST_THRESHOLD,
) -> List[Dict[str, Union[str, float]]]:
    """Predict sentiment for a single text or a list of texts using a trained model.

    Args:
        texts (str or list[str]): The input text(s) to classify. Can be a single string
            or a list of strings for batch prediction.
        model (transformers.PreTrainedModel): The trained DistilBERT model for prediction.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for text preprocessing.
        threshold (float, optional): Decision threshold for positive sentiment.
            Defaults to BEST_THRESHOLD (0.95).

    Returns:
        list[dict]: A list of predictions, each containing:
            - 'text' (str): The input text.
            - 'sentiment' (str): Predicted sentiment ("positive" or "negative").
            - 'probability_positive' (float): Probability of the positive class.
            - 'threshold' (float): The threshold used for classification.

    Raises:
        ValueError: If inputs are invalid (e.g., empty list, non-string items).

    Notes:
        - Handles invalid inputs (e.g., NaN, non-strings) by converting to empty strings.
        - Applies padding and truncation to ensure consistent sequence lengths.
        - Processes all texts in a single batch for efficiency.
    """
    # Normalize input to a list
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list):
        raise ValueError("Input 'texts' must be a string or a list of strings")
    if not texts:
        raise ValueError("Input 'texts' list cannot be empty")

    # Clean and validate texts
    cleaned_texts = [
        str(text) if isinstance(text, str) or pd.notna(text) else "" for text in texts
    ]

    # Prevent memory issues with large batches
    if len(texts) > 10:
        raise ValueError("Batch size exceeds limit of 10 texts")

    # Tokenize the batch
    inputs = tokenizer(
        cleaned_texts,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prob_positive = probs[:, 1].tolist()  # Probabilities of positive class
    except Exception as e:
        raise RuntimeError(f"Inference failed: {str(e)}")

    # Generate predictions
    predictions = [
        {
            "text": text,
            "sentiment": "positive" if prob >= threshold else "negative",
            "probability_positive": prob,
            "threshold": threshold,
        }
        for text, prob in zip(cleaned_texts, prob_positive)
    ]

    return predictions


def main():
    """Command-line interface for sentiment prediction.

    Allows users to input a single text or a list of texts via command-line arguments
    and prints the predicted sentiment for each.
    """
    parser = argparse.ArgumentParser(
        description="Predict sentiment using a trained DistilBERT model."
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        default=["This movie was great!", "This movie was terrible!"],
        help="Text(s) to classify (space-separated). Default: two example texts.",
    )
    args = parser.parse_args()

    # Predict sentiment
    try:
        predictions = predict_sentiment(args.texts, model, tokenizer)
        print("\nPredictions:\n")
        for pred in predictions:
            print(f"Text: {pred['text']}")
            print(f"Sentiment: {pred['sentiment']}")
            print(f"Probability (Positive): {pred['probability_positive']:.4f}")
            print(f"Threshold: {pred['threshold']:.4f}")
            print("-" * 50)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
