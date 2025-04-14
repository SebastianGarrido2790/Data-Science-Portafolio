import torch
from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and tokenizer
print("Loading model and tokenizer...")
device = "cpu"  # Your setup uses CPU
tokenizer = DistilBertTokenizer.from_pretrained("./distilbert_model")
model = DistilBertForSequenceClassification.from_pretrained("./distilbert_model")
model.to(device)
model.eval()  # Set model to evaluation mode (disables dropout, etc.)

# Define the prediction threshold (from training results)
BEST_THRESHOLD = 0.95


@app.route("/predict", methods=["POST"])
def predict():
    """Predict sentiment for a list of text inputs.

    Expects a JSON payload with a 'texts' field containing a list of reviews to classify.
    Returns a list of predictions with the predicted sentiment (positive/negative) for each
    text using the trained DistilBERT model.

    Example request:
        curl -X POST -H "Content-Type: application/json" \
             -d '{"texts": ["I love my Alexa!", "This Alexa is terrible, it never works."]}' \
             http://127.0.0.1:5000/predict

    Returns:
        JSON response with a list of predictions, each containing the text, predicted sentiment,
        and probability.
    """
    try:
        # Get the input texts from the request
        data = request.get_json()
        if not data or "texts" not in data:
            return jsonify({"error": "Missing 'texts' field in request"}), 400

        texts = data["texts"]
        if not isinstance(texts, list):
            return jsonify({"error": "'texts' must be a list"}), 400

        if not texts:
            return jsonify({"error": "'texts' list cannot be empty"}), 400

        # Validate each text in the list
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                return (
                    jsonify(
                        {"error": "Each item in 'texts' must be a non-empty string"}
                    ),
                    400,
                )
        if len(texts) > 10:  # Adjust limit based on your RAM
            return jsonify({"error": "Batch size exceeds limit of 10 texts"}), 400

        # Tokenize the batch of texts
        inputs = tokenizer(
            texts,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Move inputs to the correct device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run inference on the batch
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prob_positive = probs[:, 1].tolist()  # Probabilities of positive class (1)

        # Apply the best threshold to get predictions
        predictions = [
            {
                "text": text,
                "sentiment": "negative" if prob < BEST_THRESHOLD else "positive",
                "probability_positive": prob,
                "threshold": BEST_THRESHOLD,
            }
            for text, prob in zip(texts, prob_positive)
        ]

        # Return the list of predictions
        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="127.0.0.1", port=5000, debug=False)
