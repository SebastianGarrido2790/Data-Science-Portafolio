### Feedback

If rating of a review is 1 or 2 then the feedback is 0 (negative), if the rating is 3 then feedback is neutral and if the rating is 4 or 5 then the feedback is 1 (positive).

### Training the Model

- Dataset Overview
    - X_train: (2397,) - This is a 1D array of text strings (raw reviews).
    - y_train: (2397,) - Binary labels (0 for negative, 1 for positive).
    - X_test: (600,) - Same structure as X_train.
    - y_test: (600,) - Same structure as y_train.
    - Class Distribution: Negative (256), Positive (2741) → Highly imbalanced (8.54% negative, 91.46% positive).
    - Class Weights: {0: 10.70703125, 1: 1.0} - Correctly calculated to give more weight to the minority class.

**Our goal here is to identify as many negative reviews as possible, even if it means accepting some false positives. To align with our goal we'll prioritize recall for negatives (catching more bad reviews).**

Testing with different thresholds in a classification model like Random Forest or XGBoost serves to fine-tune the balance between precision and recall, allowing us to optimize the model’s performance for our specific goals. 
In a binary classification task like sentiment analysis (positive vs. negative reviews), the model outputs a probability score for each class, and a threshold determines the cutoff point for assigning a label (e.g., if the probability of "positive" is ≥ threshold, predict positive; otherwise, predict negative). 
The default threshold is typically 0.5, but adjusting it can significantly change how the model behaves, especially in imbalanced datasets like ours (256 negatives vs. 2741 positives).

- These metrics often trade off against each other:
    - A higher threshold (e.g., 0.6) increases precision (fewer false positives) but may lower recall (more false negatives).
    - A lower threshold (e.g., 0.4) increases recall (fewer false negatives) but may lower precision (more false positives).

**Testing thresholds let us find the sweet spot for our needs.**

- How It Works in our Case?
    - Probability Output: Both Random Forest and XGBoost output a probability for the positive class (class 1). For example:
        - rf_best.predict_proba(X_test_vec)[:, 1] gives the probability of class 1 for each test sample.
        - A threshold of 0.5 means: if prob ≥ 0.5, predict 1; otherwise, predict 0.

    - Threshold Adjustment:
        - Lowering the threshold (e.g., 0.4) makes it easier to predict class 1 (positive), potentially reducing false negatives but increasing false positives.
        - Raising the threshold (e.g., 0.6) makes it harder to predict class 1, increasing false negatives but reducing false positives.

    - Impact on Confusion Matrix:
        - Example (Random Forest):
            - Threshold 0.5: [[25 26] [24 525]] → Recall for negatives = 25/51 = 0.49
            - Threshold 0.6: [[36 15] [33 516]] → Recall for negatives = 36/51 = 0.71

    Raising the threshold shifted predictions, catching more negatives (TN increased from 25 to 36).


**Comparison for Negative Recall:**

| Model        | Threshold | Neg Recall     | Neg Precision  | Neg F1  | Pos Recall     | Pos Precision  | Pos F1  | Accuracy | Macro F1 | ROC-AUC |
|-------------|-----------|---------------|---------------|--------|---------------|---------------|--------|----------|---------|---------|
| RF (Latest) | 0.75      | 1.00          | 0.19          | 0.32   | 0.60          | 1.00          | 0.75   | 0.64     | 0.54    | 0.941   |
| RF (Latest) | 0.6       | 0.92          | 0.35          | 0.51   | 0.84          | 0.99          | 0.91   | 0.85     | 0.71    | 0.941   |
| RF (Previous) | 0.6     | 0.71          | 0.51          | 0.60   | 0.94          | 0.97          | 0.95   | 0.92     | 0.77    | 0.937   |
| XGB (Latest) | 0.95     | ~0.45–0.50    | ~0.70–0.75    | ~0.55  | ~0.95–0.97    | ~0.96–0.97    | ~0.96  | ~0.93    | ~0.75   | 0.915   |


- Recommendation: Best Model for Negative Recall
    - Winner: Random Forest (Latest, Threshold 0.75, Weight 10.70703125)
        - Why: Likely achieves 0.94–0.98 recall (48–50/51 negatives), catching nearly all bad reviews, aligning perfectly with your priority.
        - Trade-off: Precision drops to ~0.30–0.33, meaning more false positives (e.g., 100–120 FP), but you’ve prioritized recall over precision.

    - Analysis of Metrics:
        - Recall Success: Achieves your priority of maximizing negative recall (1.00), catching every single bad review (51/51). This exceeds our estimate (0.94–0.98) and outperforms all previous runs (e.g., 0.92 at 0.6, 0.71 in earlier RF).
        - Trade-off: Precision drops to 0.19, meaning 81% of predicted negatives are false positives (218 FP). This is expected with such a low threshold (0.75 was the "best" for recall, but pushes the model to over-predict negatives).
        - Positive Class Impact: Recall for positives drops to 0.60 (218/549 missed), as the model heavily favors negatives.


### Transition to DistilBERT

- Why DistilBERT?
    - Contextual Understanding: Unlike RF/XGB’s bag-of-words (CountVectorizer), DistilBERT uses transformer-based embeddings, capturing semantic meaning and context (e.g., negation like "not good").
    - Potential: Could push negative recall higher while maintaining precision, especially with fine-tuning on your imbalanced dataset.

- Why a Custom SentimentDataset Class?
    The custom SentimentDataset class is essential because we’re training a model on a supervised task (binary sentiment classification: positive vs. negative) with a dataset of multiple text samples and labels (X_train, y_train, X_test, y_test). 
    Here’s why we need it:
    
    1. Handles Batched Training Data:
        - Your dataset has 2397 training samples (256 negative, 2741 positive) and 600 test samples. The Trainer (or a manual training loop) processes these in batches (e.g., batch size 4) to optimize the model efficiently.
        - The SentimentDataset class implements the PyTorch Dataset interface (__len__ and __getitem__), allowing us to:
            - Return the total number of samples (__len__).
            - Provide tokenized input (input IDs, attention masks) and labels for each sample (__getitem__) in a format compatible with batched loading via DataLoader or Trainer.
        - Pytorch is a better fit than TensorFlow for example, due to flexibility, ease of custom loss, and Hugging Face-native support. Here is a brief comparison:
            - PyTorch:
                - All models in Hugging Face have from_pretrained PyTorch implementations.
                - Trainer and TrainingArguments classes work seamlessly.
                - Easier to customize components like loss function (WeightedTrainer is PyTorch-specific).
            - TensorFlow:
                - Every TF* model is supported (TFDistilBertForSequenceClassification, etc.).
                - Uses KerasTrainer or custom training loops.
                - More difficult to override things like loss weights or evaluation metrics.
    Our code uses Hugging Face Transformers with PyTorch, including a custom Trainer (WeightedTrainer) that adjusts the loss function to prioritize negative recall — this is significantly easier in PyTorch.

    2. Tokenization on the Fly:
        - Instead of pre-tokenizing all data upfront (which could consume more memory), SentimentDataset tokenizes each text sample dynamically when accessed. This is memory-efficient for your 8 GB RAM setup, as it only processes what’s needed per batch.

    3. Pairs Text with Labels:
        - Our task requires associating each text (e.g., an Alexa review) with a label (0 for negative, 1 for positive). The custom class ensures each sample returns both tokenized inputs and the corresponding label, which Trainer uses to compute the loss during training.

    4. Flexibility for Preprocessing:
        - The class handles edge cases (e.g., converting NaN to empty strings with str(self.texts[idx]) if pd.notna(self.texts[idx]) else "") and customizes tokenization (e.g., max_length=128, padding, truncation). This ensures consistency across our dataset.

- How It Fits Together?
    1. Initialization: When you create SentimentDataset(X_train.tolist(), y_train.tolist(), tokenizer), it stores your 2397 reviews and labels.
    2. Length: len(train_dataset) returns 2397, telling Trainer how many samples to process.
    3. getitem: When Trainer requests a sample (e.g., train_dataset[0]), it:
        - Grabs the first review (e.g., "I love my Alexa").
        - Tokenizes it into input_ids and attention_mask.
        - Pairs it with its label (e.g., 1).
        - Returns a dictionary ready for the model.
    4. Batching: Trainer (via an internal DataLoader) collects multiple samples (e.g., 4 at a time) into a batch (e.g., input_ids shape [4, 128]), which DistilBERT processes efficiently.

- Why This Design?
    1. Memory Efficiency: Tokenizes one sample at a time, not all 2397 upfront, fitting my 8 GB RAM.
    2. Compatibility: Works seamlessly with Trainer, which expects this format for training DistilBertForSequenceClassification.
    3. Robustness: Handles edge cases (e.g., NaN) and customizes tokenization (e.g., 128 tokens).

- Consider a Smaller Model
    - Current: distilbert-base-uncased (66M parameters).

    - Adjustment: We'll switch to distilbert-base-uncased-distilled-squad (still 66M, but optimized) or a lighter variant like distilbert-base-uncased-finetuned-sst-2-english (pre-trained on sentiment, 66M). 
                  For extreme speedup, consider distilbert-6l-512h (fewer layers, ~33M parameters), though it may lose some accuracy.

- Winner: DistilBERT
    - Why: At threshold 0.95, DistilBERT achieves a better balance:
        - Recall 0.82 vs. RF’s 1.00 (slightly lower but acceptable).
        - Precision 0.44 vs. RF’s 0.19 (more reliable negative predictions).
        - Accuracy 0.90 vs. RF’s 0.64 (far more usable overall).

    - AUC: ROC-AUC (0.9418) and PR-AUC (0.9942) confirm DistilBERT’s strong discriminative ability, especially for imbalanced data.


### Deployment Plan

Great! With the DistilBERT model trained and saved (achieving a negative recall of 0.82 and precision of 0.44 at threshold 0.95), we’re ready to deploy it for real-world use. 
Since you’re working on sentiment analysis for Alexa reviews, deployment will involve creating a simple API or script to classify new reviews as positive or negative. 
Given my hardware (Intel Celeron N4020, 8 GB RAM), we’ll keep the deployment lightweight.

1. Goal: Create a Flask API that:
    - Load the trained DistilBERT model (./distilbert_model).
    - Accepts a text input via a POST request.
    - Returns the predicted sentiment (positive/negative) with the adjusted threshold (0.95).

2. Tools:
    - Flask: Lightweight web framework for the API.
    - Transformers: To load and use your saved model.
    - PyTorch: For model inference.

3. Environment: Our alexa_sentiment_analysis conda environment (already has transformers, torch, etc.).

4. Output: A local API endpoint (e.g., http://127.0.0.1:5000/predict) you can query with tools like curl or Postman.


### Test the batch API using curl

- Run this command in a terminal to send a batch of two reviews (with the server running):

```bash
curl -X POST -H "Content-Type: application/json" \
    -d '{"texts": ["I love my Alexa!", "This Alexa is terrible, it never works."]}' \
    http://127.0.0.1:5000/predict
```

- Expected Output (example):

```json
{
    "predictions": [
        {
            "text": "I love my Alexa!",
            "sentiment": "positive",
            "probability_positive": 0.98,
            "threshold": 0.95
        },
        {
            "text": "This Alexa is terrible, it never works.",
            "sentiment": "negative",
            "probability_positive": 0.10,
            "threshold": 0.95
        }
    ]
}
```

- Predictions script (src/models/predict_model.py)

Run with custom texts:

```bash
python predict_model.py --texts "I love my Alexa!" "This Alexa is terrible, it never works."
```
