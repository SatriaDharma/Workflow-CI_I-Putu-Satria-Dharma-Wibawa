# Kriteria 3: Workflow CI dengan MLflow Project

## 📋 Ringkasan
Repositori ini mengimplementasikan CI (Continuous Integration) untuk re-training model menggunakan MLflow Project dan Docker.

## 📁 Struktur Repositori
* `.github/workflows/ci.yml`: Konfigurasi GitHub Actions untuk otomatisasi *build* Docker Image dan *push* ke Docker Hub.
* `MLProject/`: Folder utama MLflow Project yang berisi:
    - `MLProject`: File konfigurasi (entry points) untuk MLflow.
    - `Dockerfile`: Instruksi pembuatan image Docker.
    - `conda.yaml`: Definisi *environment* untuk memastikan reproduktifitas model.
    - `modelling.py`: Script pelatihan model dasar (baseline).
    - `modelling_tuning.py`: Proses pelatihan model dengan hyperparameter tuning dan GridSearchCV.
    - `heart_disease_rf_model/`: Artefak model yang telah dilatih dan siap dibungkus ke Docker.
    - `heart_preprocessing/`: Dataset yang telah melalui tahap pembersihan.
* `Docker_Hub.txt`: Tautan menuju repositori Docker Hub publik.

## 🚀 Fitur Utama
- **MLflow Project:** Standarisasi eksekusi model.
- **Dockerized Model:** Model dikemas dalam kontainer untuk deployment yang konsisten.
- **GitHub Actions:** Otomatisasi build dan push image ke Docker Hub.
