# Kriteria 3: Workflow CI dengan MLflow Project

## 📋 Ringkasan
Repositori ini mengimplementasikan CI (Continuous Integration) untuk re-training model menggunakan MLflow Project dan Docker.

## 📁 Struktur Repositori
- `heart_disease_rf_model/`: Artifak model hasil pelatihan.
- `heart_preprocessing/`: Data yang sudah dibersihkan.
- `MLProject`: Definisi entry point dan environment.
- `conda.yaml`: Dependensi environment.
- `modelling_tuning.py`: Proses pelatihan model dengan hyperparameter tuning dan GridSearchCV.
- `Dockerfile`: Instruksi pembuatan image Docker.

## 🚀 Fitur Utama
- **MLflow Project:** Standarisasi eksekusi model.
- **Dockerized Model:** Model dikemas dalam kontainer untuk deployment yang konsisten.
- **GitHub Actions:** Otomatisasi build dan push image ke Docker Hub.
