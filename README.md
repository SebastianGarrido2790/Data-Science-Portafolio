# 📊 Data Science Portafolio – Sebastián Garrido

Bienvenido a mi portafolio profesional de proyectos en ciencia de datos. Este repositorio contiene una colección de proyectos de ciencia de datos diseñados para demostrar habilidades en análisis de datos, aprendizaje automático, modelado predictivo, visualización y análisis de series temporales. Cada proyecto incluye código bien estructurado, documentación clara y enfoques prácticos para la resolución de problemas del mundo real.

## 🔍 Tabla de Resumen

| Proyecto | Descripción |
|----------|-------------|
| [1. AI/ML Customer Churn Prediction](./1_AI-ML_Customer_Churn/) | Pipeline de ML completo con embeddings, resumen de texto y XGBoost para predicción de fuga de clientes. |
| [2. Predicción de Costos Médicos](./2_Predicción_de_Costos_Médicos/) | Modelos lineales y de árbol para predecir gastos médicos individuales. |
| [3. KMeans Online Retail](./3_KMeans_Online_Retail/) | Segmentación de clientes basada en análisis RFM y clustering. |
| [4. Análisis de Sentimiento – Amazon Alexa](./4_Análisis_de_Sentimiento_–_Amazon_Alexa/) | Clasificación de reseñas con modelos BERT y XGBoost + API Flask. |
| [5. Análisis de Sentimiento con Redes Neuronales](./5_Análisis_de_Sentimiento_con_Redes_Neuronales/) | Comparativa de modelos NN: FFNN, CNN, RNN, LSTM. |
| [6. Análisis de Ventas – Superstore](./6_Análisis_de_Ventas_–_Superstore/) | Exploración de segmentos, productos y regiones para optimizar ventas. |
| [7. Accidentes de Tráfico en EE.UU](./7_Accidentes_de_Trafico_en_EEUU/) | Análisis geográfico, temporal y climático de más de 7M de accidentes. |
| [8. Series Temporales](./8_Time_Series_Projects/) | Predicción de energía y ventas con XGBoost, Prophet y sktime. |
| [9. Caso de estudio de Churn](./9_Caso_Estudio_Churn/) | Documento de Word donde se aplica metodologicamente el ciclo de vida de la ciencia de datos. |
| [10. NYC Taxis Project](./10_NYC_Taxis_Project/) | Análisis exploratorio de viajes en taxi en NYC: patrones geográficos, temporales y económicos. |

## 🛠 Tecnologías

- Python, Pandas, NumPy, Scikit-Learn, XGBoost, LightGBM
- TensorFlow, Keras, PyTorch
- NLP (BERT, GloVe), Flask API
- Transformers, OpenAI
- Prophet, sktime, Time Series Analysis
- Matplotlib, Seaborn, Plotly
- Conda, Git, GitHub
- VSCode, Colabs notebooks

## 📌 Proyectos

### 1. AI ML Customer Churn Prediction
**Descripción:** Implementación de un pipeline completo de aprendizaje automático para predecir la pérdida de clientes. Se emplean técnicas avanzadas de IA, como resumen de texto y embeddings, para extraer información relevante de notas de tickets de clientes. Estos datos, combinados con características numéricas, alimentan un clasificador XGBoost.

🔹Pasos del proyecto:
- **Text Summarization:** Uso de IA (Hugging Face u OpenAI) para condensar notas largas de tickets en resúmenes concisos.
- **Embeddings:** Transformación de resúmenes en representaciones numéricas mediante modelos de lenguaje preentrenados.
- **Modelado:** Integración de estas características enriquecidas en un clasificador XGBoost que se entrena, evalúa y guarda para su implementación.

### 2. Healthcare_Insurance_Costs.ipynb
**Descripción:** Predicción de costos de seguros médicos utilizando técnicas de regresión (Lineal, Ridge, Lasso, ElasticNet, Regresión Polinómica y Árboles de Decisión). El objetivo es ayudar a la planificación financiera y personalización de servicios.

🔹 Modelo principal: Regresión multivariable
🔹 Métricas evaluadas: MAE, RMSE, R²

### 3. KMeans Online Retail
**Descripción:** Segmentación de clientes de un comercio electrónico mediante análisis RFM y clustering con KMeans para estrategias de marketing personalizadas.

🔹 Pasos clave:
- Preprocesamiento y eliminación de valores atípicos
- Cálculo de métricas RFM
- Aplicación de KMeans y visualización de clusters

### 4. Sentiment Analysis of Amazon Alexa Reviews
**Descripción:** Análisis de sentimientos en reseñas de Amazon Alexa mediante un modelo DistilBERT ajustado, Random Forest y XGBoost. Se prioriza el recall para identificar comentarios negativos con alta precisión.

🔹 Implementación:
- Preprocesamiento de texto
- Entrenamiento de modelos
- API en Flask para predicciones

### 5. NN_Sentiment_Analysis.ipynb
**Descripción:** Comparación de múltiples redes neuronales para análisis de sentimientos utilizando embeddings, CNNs, RNNs y modelos bidireccionales LSTM sobre datos de IMDb, Amazon y Yelp.

🔹 Modelos probados:
- FFNN con embeddings preentrenados
- CNN + LSTM
- Redes recurrentes bidireccionales

### 6. Superstore_Sales_Analysis.ipynb
**Descripción:** Análisis de ventas de una cadena minorista para identificar tendencias, segmentar clientes y optimizar estrategias comerciales.

🔹 Incluye:
- Segmentación de clientes
- Análisis de métodos de envío
- Visualización geográfica de ventas

### 7. US_Traffic_Accidents.ipynb
**Descripción:** Exploración de datos de accidentes automovilísticos en EE.UU. para identificar patrones de severidad, impacto climático y distribución geográfica.

🔹 Preguntas clave abordadas:
- ¿Cuáles son los horarios pico de accidentes?
- ¿Qué ciudades tienen mayor frecuencia de accidentes?
- ¿Cómo afecta el clima a la tasa de accidentes?

### 8. Proyectos de Series Temporales (Carpeta `Time_Series_Projects`)
#### 8.1 Time_Series_XGBoost.ipynb
**Descripción:** Predicción del consumo energético horario utilizando XGBoost, con un enfoque en la mejora de precisión en transiciones nocturnas.

🔹 Estrategias aplicadas:
- Ingeniería de características temporales
- Modelos en ensamblado con LightGBM y LSTM

#### 8.2 Time Series Forecasting with Multiple Techniques
**Descripción:** Predicción de ventas diarias con Prophet, sktime y XGBoost. Se incluyen técnicas avanzadas de modelado y comparación de desempeño entre enfoques clásicos y modernos.

🔹 Técnicas utilizadas:
- Prophet para detección de estacionalidad y tendencias
- AutoARIMA y Prophet en sktime
- XGBoost con ingeniería de características

### 9. Análisis de Churn en Servicios de Streaming (Basado en Ciencia de Datos)

**Descripción:** En este documento se muestra mi entendimiento de cómo un enfoque sistemático de la ciencia de datos puede transformar la toma de decisiones basada principalmente en la intuición en una estrategia basada en datos. Al documentar de forma estructurada el desarrollo de un modelo predictivo de la pérdida de clientes (churn), el objetivo es proporcionar información práctica para la optimización de estrategias de retención y mejorar el rendimiento y la competitividad de la empresa.

### 10. NYC Taxis Project

**Descripción:** Este proyecto analiza datos de viajes en taxi en la ciudad de Nueva York a través de un enfoque exploratorio. El proyecto sigue la estructura clásica de canalización de ML: ingesta → ingeniería de características → modelado → puntuación de lotes → panel de control.

🔹 Problema empresarial 💼
Creemos un modelo predictivo para pronosticar el número de viajes en taxi en Manhattan (Nueva York).
- Paso 1. Obtener datos sin procesar.
- Paso 2. Transformar los datos sin procesar en (características, objetivos) y dividirlos en entrenamiento y prueba.
- Paso 3. Construir un modelo de línea base.
- Paso 4. Mejorar la línea base utilizando ML.
- Paso 5. Poner el modelo a trabajar con un arquitectura de tres pipelines, para construir un sistema de puntuación por lotes (batch-scoring system).
- Paso 6. Construya un tablero de monitoreo.

🔹 Aplicaciones potenciales:
- Optimización de rutas para taxistas.
- Análisis de demanda urbana.
- Benchmark para políticas de movilidad sostenible.

## 📂 Estructura de Proyectos Complejos
Algunos proyectos siguen la siguiente estructura para modularidad y reproducibilidad:

```
├── LICENSE
├── README.md          <- Descripción del proyecto y cómo usarlo
├── data
│   ├── external       <- Datos de terceros
│   ├── interim        <- Datos intermedios transformados
│   ├── processed      <- Datos finales listos para modelado
│   └── raw            <- Datos originales sin procesar
│
├── docs               <- Documentación del proyecto
│
├── models             <- Modelos entrenados y predicciones
│
├── notebooks          <- Análisis exploratorio en Jupyter
│
├── references         <- Documentación y referencias
│
├── reports            <- Reportes generados en HTML/PDF
│   └── figures        <- Visualizaciones y gráficos
│
├── requirements.txt   <- Dependencias del proyecto
├── environment.yml    <- Archivo para recrear el entorno Conda
│
├── src                <- Código fuente
│   ├── data           <- Scripts para obtener/procesar datos
│   ├── features       <- Ingeniería de características
│   ├── models         <- Entrenamiento y predicciones
│   └── visualization  <- Generación de gráficos y reportes
```

## 🚀 Cómo Usar este Repositorio
1. Clona este repositorio en tu máquina local:
   ```bash
   git clone https://github.com/SebastianGarrido2790/portafolio_ciencia_datos.git
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
   o si usas Conda:
   ```bash
   conda env create -f environment.yml
   conda activate mi_entorno
   ```
3. Explora cada proyecto dentro de su carpeta correspondiente y ejecuta los notebooks o scripts según las instrucciones en `README.md`.

---

📧 **Contacto:** Si tienes alguna pregunta o sugerencia, no dudes en contactarme en [sebastiangarrido2790@gmail.com] o vía LinkedIn [www.linkedin.com/in/sebastían-garrido-638959320].

¡Gracias por visitar mi portafolio! 🎯
