# 📊 Data Science Portfolio – Sebastián Garrido

Welcome to my professional data science project portfolio. This repository presents a curated collection of projects that demonstrate expertise in data analysis, machine learning, predictive modeling, data visualization, and time series forecasting. Each project is accompanied by well-structured code, comprehensive documentation, and practical approaches to solving real-world problems.

## 🔍 Project Summary

| Project | Description |
|---------|-------------|
| [1. AI/ML Customer Churn Prediction](./01_AI-ML_Customer_Churn/) | Complete ML pipeline integrating embeddings, text summarization, and XGBoost for customer churn prediction. |
| [2. Healthcare Insurance Cost Prediction](./2_Predicción_de_Costos_Médicos/) | Application of linear and tree-based models to forecast individual medical expenses. |
| [3. KMeans Online Retail](./3_KMeans_Online_Retail/) | Customer segmentation through RFM analysis and clustering using KMeans. |
| [4. Sentiment Analysis – Amazon Alexa](./4_Análisis_de_Sentimiento_–_Amazon_Alexa/) | Review classification utilizing BERT, XGBoost, and a Flask API for deployment. |
| [5. Sentiment Analysis with Neural Networks](./5_Análisis_de_Sentimiento_con_Redes_Neuronales/) | Comparative study of FFNN, CNN, RNN, and LSTM architectures for sentiment analysis. |
| [6. Superstore Sales Analysis](./6_Análisis_de_Ventas_–_Superstore/) | Sales and profitability analysis across customer segments, product categories, and regions. |
| [7. U.S. Traffic Accidents](./7_Accidentes_de_Trafico_en_EEUU/) | Geospatial, temporal, and meteorological analysis of over 7 million traffic accidents. |
| [8. Time Series Projects](./8_Time_Series_Projects/) | Forecasting of energy consumption and sales using XGBoost, Prophet, and sktime. |
| [9. Churn Case Study](./9_Caso_Estudio_Churn/) | Comprehensive case study applying an end-to-end data science lifecycle to customer churn prediction. |
| [10. NYC Taxis Project](./10_NYC_Taxis_Project/) | Exploratory analysis of NYC taxi ride patterns from a business intelligence perspective. |
| [11. PDF Summarization with LLMs](./11_PDF_Summary_with_LLMs/) | Automated summarization of PDF files using OpenAI and Anthropic LLMs, with flexible output formats. |
| [12. Employee Attrition Prediction](./12_Employee_Attrition/) | End-to-end ML pipeline to predict employee attrition, deployed on AWS ECS with batch and real-time API capabilities. |
| [13. SQL Data Scientist Job Market Analysis](./13_SQL_Project_Data_Job_Analysis/) | SQL-driven analysis of data science job postings to uncover trends in salaries, skills, and career opportunities. |

## 🛠 Technologies

- Python, Pandas, NumPy, Scikit-Learn, XGBoost, LightGBM
- TensorFlow, Keras, PyTorch
- Natural Language Processing (BERT, GloVe), Transformers
- LangChain
- Flask and FastAPI for development
- Prophet, sktime, Time Series Forecasting
- Matplotlib, Seaborn, Plotly
- SQL, PostgreSQL
- Conda, Git, GitHub
- Visual Studio Code, Google Colab
- uv (package management for specific projects)

## 📌 Project Details

### [1. AI/ML Customer Churn Prediction](./01_AI-ML_Customer_Churn/)
**Summary:** Developed a complete machine learning pipeline to predict customer churn, leveraging advanced AI techniques for feature enrichment. Text summarization and embedding extraction from customer support ticket notes were integrated into an XGBoost classifier.

🔹 Key components:
- **Text Summarization:** Utilized Hugging Face/OpenAI models to create concise summaries of ticket notes.
- **Embeddings:** Converted text summaries into dense numerical representations.
- **Modeling:** Combined numerical and embedded features to train and optimize an XGBoost classifier for deployment.

### 2. Healthcare Insurance Cost Prediction
**Summary:** Built predictive models for healthcare insurance costs using multiple regression techniques, including Linear Regression, Ridge, Lasso, ElasticNet, Polynomial Regression, and Decision Trees.

🔹 Highlights:
- Principal model: Multivariate regression
- Metrics evaluated: MAE, RMSE, R²

### 3. KMeans Online Retail
**Summary:** Applied RFM (Recency, Frequency, Monetary Value) analysis and KMeans clustering to segment e-commerce customers, enabling data-driven marketing strategies.

🔹 Key steps:
- Outlier detection and preprocessing
- RFM feature engineering
- Cluster identification and visualization

### 4. Sentiment Analysis – Amazon Alexa
**Summary:** Performed sentiment analysis on Amazon Alexa reviews using DistilBERT fine-tuning, Random Forest, and XGBoost classifiers, with emphasis on maximizing recall for negative reviews.

🔹 Implementation:
- Advanced NLP preprocessing
- Model training and evaluation
- API development using Flask for real-time prediction

### 5. Sentiment Analysis with Neural Networks
**Summary:** Conducted a comparative study of various deep learning architectures for sentiment classification on IMDb, Amazon, and Yelp review datasets.

🔹 Models compared:
- Feedforward Neural Networks (FFNN) with pre-trained embeddings
- CNN + LSTM hybrid models
- Bidirectional RNNs and LSTMs

### 6. Superstore Sales Analysis
**Summary:** Analyzed sales performance across different customer segments, product categories, and geographic regions to identify strategic business insights.

🔹 Areas of focus:
- Customer segmentation and profitability
- Analysis of shipping methods and delivery performance
- Regional sales trends visualization

### 7. U.S. Traffic Accidents
**Summary:** Explored a dataset of over seven million U.S. traffic accidents to identify trends based on time, location, and weather conditions.

🔹 Research questions addressed:
- Peak accident times
- Cities with the highest accident rates
- Weather factors influencing accident severity

### 8. Time Series Projects
#### 8.1 Time Series Forecasting with XGBoost
**Summary:** Developed an hourly energy consumption forecasting model using XGBoost, with enhanced feature engineering for improved accuracy during off-peak hours.

🔹 Techniques:
- Advanced temporal feature extraction
- Model ensembling with LightGBM and LSTM

#### 8.2 Time Series Forecasting with Prophet and sktime
**Summary:** Forecasted daily sales data using Prophet, sktime, and XGBoost, integrating both classical and modern time series forecasting techniques.

🔹 Techniques applied:
- Seasonality and trend modeling with Prophet
- Classical and machine learning time series models with sktime
- Feature-enriched XGBoost forecasting

### 9. Churn Case Study
**Summary:** For this six-month project, we created a fictional company, StreamHub, Inc., a streaming platform with 10 million subscribers similar to Netflix and Spotify, using real-world figures to simulate the industry's reality. The objective is to conduct a case study aimed to predict customer churn and reduce the 5% monthly churn rate by 10% (saving $500,000/month) applying a complete data science lifecycle. Using the CRISP-DM methodology, we developed, deployed, and operationalized an XGBoost model (AUC-ROC = 0.85, recall = 0.74), delivering $2.93M in annual net savings (166% ROI). The project showcases end-to-end data science expertise, from business understanding to production-grade MLOps, aligning technical solutions with strategic business goals.

🔹 **Phase 1: Business Understanding (June 2025)**
- Established SMART goals: Reduce churn by 10% within six months, achieve AUC-ROC ≥ 0.85.
- Formulated hypotheses (e.g., low engagement predicts churn).
- Engaged stakeholders (marketing, executives) via bi-weekly meetings.

🔹 **Phase 2: Data Understanding and Governance (July 2025)**
- Identified datasets: user demographics, viewing history, subscription details, customer interactions (10M users, ~6GB).
- Evaluated quality: 92% completeness, 99% accuracy, with minor missing values (e.g., 5% in age).
- Confirmed temporal components (e.g., timestamps for seasonality).

🔹 **Phase 3: Exploratory Data Analysis & Insight Generation (August 2025)**
- Conducted univariate, bivariate, multivariate, and time-series analyses (e.g., January churn spikes at 6.5%).
- Tested seven hypotheses, confirming predictors like low watch_time (7% churn rate) and inactivity (10% churn rate).
- Visualized findings (heatmaps, box plots, time-series plots).

🔹 **Phase 4: Data Preparation & Feature Engineering (September 2025)**
- Cleaned data: Imputed missing values (e.g., median for age), removed duplicates (0.05%), capped outliers.
- Engineered 10 features (e.g., inactive_30_days, plan_downgrade_flag) based on EDA.
- Automated preprocessing pipelines (scikit-learn, Airflow) and used time-based train/test split (Jan 2023–Apr 2025 vs. May–Jun 2025).

🔹 **Phase 5: Modeling & Experimentation (October 2025)**
- Framed as binary classification, using weighted log-loss to prioritize churners.
- Tested algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM) with MLflow tracking.
- Achieved AUC-ROC = 0.86 (XGBoost) and recall = 0.75 via Bayesian optimization.
- Ensured interpretability (SHAP/LIME) and fairness (5% demographic parity).

🔹 **Phase 6: Model Evaluation & Business Review (November 1–15, 2025)**
- Evaluated XGBoost on test set: AUC-ROC = 0.85, recall = 0.74, identifying 74% of churners.
- Conducted cost-benefit analysis: $370,000/month savings, $2.93M/year net ROI (166%).
- Performed error analysis: 38,120 false positives ($190,600/month cost), 26,000 false negatives ($260,000/month).
- Presented to stakeholders, securing go decision.

🔹 **Phase 7: Deployment & MLOps (November 16–30, 2025)**
- Deployed XGBoost as a real-time FastAPI endpoint on AWS ECS (150ms latency, 10M predictions/month).
- Containerized (Docker) and integrated with CI/CD (GitHub Actions).
- Connected to Salesforce and Braze for campaign automation.
- Set up monitoring (CloudWatch, Streamlit dashboard) and quarterly retraining.
- Conducted training and post-deployment review.

### 10. NYC Taxis Project
**Summary:** Conducted an end-to-end exploratory data analysis of NYC taxi rides, following a full ML pipeline to deliver actionable business insights.

🔹 Business objective:
- Predict taxi ride demand in Manhattan to support operational and policy decision-making.

🔹 Project structure:
- Data ingestion and transformation
- Baseline model development and improvement
- Batch scoring system implementation with three pipelines
- Monitoring dashboard creation

🔹 Potential applications:
- Route optimization for taxi services
- Urban demand analysis
- Mobility trend benchmarking for policy development

### 11. PDF Summarization with LLMs
**Summary:** Built a Python script to automate the summarization of multiple PDF files using large language models (LLMs) from OpenAI (GPT-3.5-turbo) or Anthropic (Claude-3.5-Haiku-latest), with customizable output formats (text or JSON).

🔹 Key features:
- Processes PDFs using `PyPDFLoader` and LangChain’s `map_reduce` summarization chain.
- Supports model selection, directory scanning, and flexible output formatting.
- Includes error handling and console feedback for user-friendly operation.

### 12. Employee Attrition Prediction
**Summary:** Developed an end-to-end machine learning pipeline to predict employee attrition for a fictional company, UseC, using the IBM HR Analytics dataset (1,470 employees). Following the CRISP-DM methodology, we built, evaluated, and deployed a LogisticRegression model (recall = 0.7102, precision = 0.27) on AWS ECS, enabling HR to reduce turnover by 10% within 12 months, potentially saving $825,000 annually. The project includes batch predictions for monthly workflows and a real-time FastAPI endpoint for on-demand predictions.

🔹 **Phase 1: Business Understanding (Week 1-2)**
- Defined SMART goal: Reduce attrition by 10% within 12 months, achieve ≥70% recall.
- Identified key drivers (e.g., low `JobSatisfaction`, high `OverTime`).
- Engaged HR stakeholders for alignment.

🔹 **Phase 2: Data Understanding (Week 3-4)**
- Explored dataset: 1,470 rows, 35 columns, no missing values.
- Flagged redundant columns (e.g., `Over18`, `EmployeeCount`) for removal.

🔹 **Phase 3: EDA & Insights (Week 5-6)**
- Confirmed key drivers: low `JobSatisfaction`, high `OverTime`, long `DistanceFromHome`.
- Visualized patterns (e.g., correlation heatmaps, attrition distributions).

🔹 **Phase 4: Data Preparation (Week 7-8)**
- Engineered features: `SatisfactionScore`, `TenureRatio`, `LongCommute`.
- Processed data: Scaled numerical features, one-hot encoded categorical features.

🔹 **Phase 5: Modeling & Experimentation (Week 9-10)**
- Tested LogisticRegression, RandomForest, XGBoost with MLflow tracking.
- Selected LogisticRegression: recall 0.7102 (threshold 0.45), precision 0.27.
- Used SHAP for interpretability (top features: `OverTime`, `SatisfactionScore`).

🔹 **Phase 6: Model Evaluation (Week 11)**
- Achieved recall goal (0.7102 ≥ 0.70); false positives (~383) mitigated via HR review.
- Estimated ROI: $825,000 savings, $1,915,000 false positive costs (at $5,000/intervention).

🔹 **Phase 7: Deployment & MLOps (Week 12)**
- Deployed on AWS ECS with Fargate: batch predictions (Docker) and real-time FastAPI API.
- Integrated CI/CD via GitHub Actions; planned Prometheus/Grafana monitoring.
- Handover to HR with training and documentation.

### 13. SQL Data Scientist Job Market Analysis
**Summary:** Analyzed a dataset of data science job postings using SQL to uncover trends in salaries, in-demand skills, and career opportunities. The project delivers actionable insights for job seekers and recruiters by identifying high-paying roles, optimal skills, and market demands.

🔹 Key components:
- **Database Design:** Created a PostgreSQL schema with relational tables for jobs, skills, and companies, visualized in an ER diagram.
- **SQL Queries:** Developed queries to analyze top-paying jobs, skill demand, salary trends, and skill co-occurrences (e.g., Python + SQL).
- **Insights:** Identified high-value skills (e.g., Snowflake, PyTorch) and trends like the dominance of remote roles and full-time positions.

## 📂 Project Architecture
Selected projects adopt a modular and reproducible structure, as follows:

```
├── LICENSE
├── README.md          <- Project overview and usage instructions
├── data
│   ├── external       <- Third-party sourced data
│   ├── interim        <- Intermediate data processing outputs
│   ├── processed      <- Final datasets ready for modeling
│   └── raw            <- Original unprocessed datasets
├── docs               <- Project documentation
├── models             <- Trained models and predictions
├── notebooks          <- Jupyter notebooks for analysis and experimentation
├── references         <- Research references and external documentation
├── reports            <- Generated analysis reports
│   └── figures        <- Graphs and visualizations
├── requirements.txt   <- Python dependencies file
├── environment.yml    <- Conda environment configuration file
├── src                <- Source code
│   ├── data           <- Scripts for data ingestion and processing
│   ├── features       <- Feature engineering scripts
│   ├── models         <- Model training and evaluation scripts
│   └── visualization  <- Data visualization scripts
```

## 🚀 How to Use This Repository
1. Clone the repository:
   ```bash
   git clone https://github.com/SebastianGarrido2790/Data-Science-Portfolio.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   or with Conda:
   ```bash
   conda env create -f environment.yml
   conda activate my_environment
   ```
3. Explore each project folder and follow the instructions provided in each `README.md` file.

---

📧 **Contact:**  
For inquiries or collaboration opportunities, please contact me at [sebastiangarrido2790@gmail.com] or connect with me on [LinkedIn](https://www.linkedin.com/in/sebastían-garrido-638959320).

Thank you for visiting my portfolio! 🎯
