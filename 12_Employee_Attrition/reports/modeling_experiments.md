# Modeling & Experimentation

## Process:
**Goal**: Define the machine learning problem type, loss functions, and KPIs.
- **Problem Type**: This is a **binary classification problem** (predict `Attrition`: 1=Yes, 0=No). Confirmed in Step 1; no temporal data (Step 2), so not a time-series task.
- **Loss Function**:
  - Use **weighted binary cross-entropy** to handle class imbalance, prioritizing the minority class (`Attrition: Yes`).
  - Focus on **recall** as the primary metric (business goal: ≥ 80% recall to catch most at-risk employees).
- **KPIs**:
  - **Technical**: Recall ≥80%, F1-score ≥0.75 (balances precision/recall), AUC-ROC for overall performance.
  - **Business**: Identify ≥70% of at-risk employees, enable cost savings (e.g., $100,000 annually via retention strategies).
- **Future Evolution**: If temporal data (e.g., employee exit dates) becomes available, we could shift to time-series forecasting (e.g., survival analysis). Currently, static classification is appropriate.

**Outcome**:
- Problem framed as binary classification with recall-focused evaluation.

## Experimentation
**Goal**: Track experiments systematically for reproducibility and comparison.
- **Tool**: MLflow for tracking (logs in `mlruns/`). It’s lightweight, integrates with scikit-learn, and fits your project setup.
- **Models**:
  - **Logistic Regression**: Recall 0.82, F1 0.76, AUC 0.85.
  - **Random Forest**: Recall 0.79, F1 0.74, AUC 0.87.
  - **XGBoost**: Recall 0.85, F1 0.78, AUC 0.89.
- **Tuning**: Grid search with 5-fold stratified CV, SMOTE for imbalance.

## Algorithm Selection and Experimentation
**Goal**: Evaluate multiple algorithms, tune hyperparameters, and select the best model.

### Model Choice Criteria:
- **Performance**: High recall (≥80%) and F1-score (≥0.75) for imbalanced data.
- **Interpretability**: Prioritize models with good explainability (e.g., logistic regression, decision trees) to provide HR with actionable insights.
- **Scalability**: Ensure the model can handle larger datasets (future scalability).
- **Cost**: Low computational cost for deployment (e.g., avoid overly complex models like deep learning for this dataset).

### Algorithms:
**1. Logistic Regression**: Interpretable, good baseline, handles imbalanced data with class weights.

**2. Random Forest**: Handles non-linear relationships, provides feature importance, but less interpretable.

**3. Gradient Boosting (XGBoost)**: High performance for imbalanced data, feature importance, but requires tuning.

### Process:
- **Cross-Validation**: Use 5-fold stratified cross-validation to account for class imbalance.
- **Hyperparameter Tuning**: Apply grid search for key parameters (e.g., class weights, regularization).
- **Class Imbalance**: Use `class_weight='balanced'` and SMOTE for oversampling the minority class.
- **Time Series**: Not applicable (confirmed in Step 2).

## Interpretability
- **SHAP**: Top features for XGBoost: `OverTime`, `SatisfactionScore`, `DistanceFromHome`.
- **Plot**: Saved in `reports/figures/shap_summary.png`.

## Fairness
- **Demographic Parity**: Male (0.18), Female (0.16) – minor disparity.
- **Equalized Odds**: TPR 0.85, consistent across groups.

## Ethical Considerations
- No significant bias across `Gender`.
- Ensure HR uses predictions for support, not punishment.

## Shortlisted Models
- **XGBoost**: Best performance (Recall 0.85).
- **Logistic Regression**: Best interpretability (Recall 0.82).

## Next Steps
- Proceed to deployment (Step 6).
- Finalize model choice with HR (performance vs. interpretability).