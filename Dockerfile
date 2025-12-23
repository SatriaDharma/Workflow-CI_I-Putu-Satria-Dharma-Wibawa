FROM python:3.12.7-slim
WORKDIR /app
RUN pip install mlflow pandas scikit-learn dagshub
COPY heart_disease_rf_model /app/model
EXPOSE 5001
CMD ["mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "5001", "--no-conda"]