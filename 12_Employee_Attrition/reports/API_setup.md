## Step 7.2: Real-Time API Setup with FastAPI

**Objective**: Set up a real-time API using FastAPI for potential future HR needs, allowing on-demand predictions.

#### Design
- **Endpoint**: `/predict` accepts a JSON payload with employee data and returns predictions.
- **Scalability**: Use an ASGI server (e.g., Uvicorn) with Docker for deployment.
- **Security**: Add basic authentication (e.g., API key) to restrict access.

- **Script Explanation**:
  - Uses FastAPI to create a `/predict` endpoint accepting `EmployeeData` JSON.
  - Preprocesses single employee data, applies the optimized threshold (0.45), and returns probability and prediction.
  - Runs with Uvicorn for local testing.

2. **Dockerfile for API**:

```dockerfile
FROM python:3.9

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential
RUN pip install uv

COPY pyproject.toml uv.lock .
RUN uv sync

COPY app.py .
COPY index.html .
COPY models/scaler.pkl /app/models/scaler.pkl
COPY data/processed/X_train.csv /app/data/processed/X_train.csv

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

- **Requirements File** (update `requirements.txt`):

```plaintext
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
mlflow==2.12.2
xgboost==2.0.3
joblib==1.4.2
fastapi==0.110.0
uvicorn==0.29.0
```

3. **Run Locally**:
   - Build and run the Docker container:
     ```
     docker build -t employee-attrition-api .
     docker run -d -p 8000:8000 -v $(pwd)/models:/app/models employee-attrition-api
     ```
   - Test with curl:
     ```
     curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"Age": 35, "DailyRate": 1000, "DistanceFromHome": 5, "Education": 3, "EmployeeNumber": 1001, "EnvironmentSatisfaction": 3, "HourlyRate": 50, "JobInvolvement": 3, "JobLevel": 2, "JobSatisfaction": 4, "MonthlyIncome": 5000, "MonthlyRate": 12000, "NumCompaniesWorked": 2, "PercentSalaryHike": 10, "PerformanceRating": 3, "RelationshipSatisfaction": 3, "StockOptionLevel": 1, "TotalWorkingYears": 10, "TrainingTimesLastYear": 2, "WorkLifeBalance": 3, "YearsAtCompany": 5, "YearsInCurrentRole": 3, "YearsSinceLastPromotion": 2, "YearsWithCurrManager": 2, "BusinessTravel": "Travel_Rarely", "Department": "Research_&_Development", "EducationField": "Life_Sciences", "Gender": "Male", "JobRole": "Research_Scientist", "MaritalStatus": "Single", "OverTime": "No"}'
     ```
   - Expected response: `{"EmployeeNumber": 1001, "Attrition_Probability": 0.15, "Attrition_Prediction": 0}` (example values).

4. **Security**:
   - Add an API key using FastAPI middleware (e.g., `fastapi.security.HTTPBearer`).
   - Example: Require a header `X-API-Key: your-secret-key`.

5. **Deployment**:
   - Host on a cloud provider (e.g., AWS ECS, Google Cloud Run) with auto-scaling for real-time demands.
   - Use HTTPS with a reverse proxy (e.g., Nginx) for production.

#### Benefits
- Enables HR to query predictions on-demand, supporting dynamic retention planning.
- Scalable with cloud deployment, preparing for future growth.

---

### Next Steps
- Test the API with HR’s sample data and refine the input schema.
- Deploy to a cloud service (e.g., AWS ECS) with CI/CD integration, adding monitoring (e.g., Prometheus) for API uptime and latency.
