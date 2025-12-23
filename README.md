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
- `Docker_Hub_Link.txt`: Tautan menuju Docker Hub.

## 🚀 Fitur Utama
- **MLflow Project:** Standarisasi eksekusi model.
- **Dockerized Model:** Model dikemas dalam kontainer untuk deployment yang konsisten.
- **GitHub Actions:** Otomatisasi build dan push image ke Docker Hub.

## 📄 Catatan
Saat ini Docker Hub sedang mengalami gangguan pada sistem UI (indeks repositori baru tidak muncul di web).
Namun, image telah berhasil di-push secara teknis ke satriadharma/heart-disease-model:v1. 
Anda dapat memverifikasi dengan menjalankan perintah docker pull satriadharma/heart-disease-model:v1 di terminal.
