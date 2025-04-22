# 📊 Data Science Portfolio – Sebastián Garrido

Welcome to my professional data science project portfolio. This repository presents a curated collection of projects that demonstrate expertise in data analysis, machine learning, predictive modeling, data visualization, and time series forecasting. Each project is accompanied by well-structured code, comprehensive documentation, and practical approaches to solving real-world problems.

## 🔍 Project Summary

| Project | Description |
|---------|-------------|
| [1. AI/ML Customer Churn Prediction](./1_AI-ML_Customer_Churn/) | Complete ML pipeline integrating embeddings, text summarization, and XGBoost for customer churn prediction. |
| [2. Healthcare Insurance Cost Prediction](./2_Predicción_de_Costos_Médicos/) | Application of linear and tree-based models to forecast individual medical expenses. |
| [3. KMeans Online Retail](./3_KMeans_Online_Retail/) | Customer segmentation through RFM analysis and clustering using KMeans. |
| [4. Sentiment Analysis – Amazon Alexa](./4_Análisis_de_Sentimiento_–_Amazon_Alexa/) | Review classification utilizing BERT, XGBoost, and a Flask API for deployment. |
| [5. Sentiment Analysis with Neural Networks](./5_Análisis_de_Sentimiento_con_Redes_Neuronales/) | Comparative study of FFNN, CNN, RNN, and LSTM architectures for sentiment analysis. |
| [6. Superstore Sales Analysis](./6_Análisis_de_Ventas_–_Superstore/) | Sales and profitability analysis across customer segments, product categories, and regions. |
| [7. U.S. Traffic Accidents](./7_Accidentes_de_Trafico_en_EEUU/) | Geospatial, temporal, and meteorological analysis of over 7 million traffic accidents. |
| [8. Time Series Projects](./8_Time_Series_Projects/) | Forecasting of energy consumption and sales using XGBoost, Prophet, and sktime. |
| [9. Churn Case Study](./9_Caso_Estudio_Churn/) | Comprehensive case study applying the full data science lifecycle to customer churn prediction. |
| [10. NYC Taxis Project](./10_NYC_Taxis_Project/) | Exploratory analysis of NYC taxi ride patterns from a business intelligence perspective. |

## 🛠 Technologies

- Python, Pandas, NumPy, Scikit-Learn, XGBoost, LightGBM
- TensorFlow, Keras, PyTorch
- Natural Language Processing (BERT, GloVe), Transformers
- LangChain
- Flask API development
- Prophet, sktime, Time Series Forecasting
- Matplotlib, Seaborn, Plotly
- Conda, Git, GitHub
- Visual Studio Code, Google Colab

## 📌 Project Details

### 1. AI/ML Customer Churn Prediction
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
**Summary:** Comprehensive case study demonstrating the application of the complete data science lifecycle to predict customer churn in streaming services, providing actionable insights for improving customer retention strategies.

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
   git clone https://github.com/SebastianGarrido2790/portafolio_ciencia_datos.git
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
