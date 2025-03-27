# Immoscout Immobilien Preisprognose

End-to-End Lösung zur Vorhersage von Immobilienpreisen basierend auf gescrapten Daten von ImmoScout24.

## Spider

* Scrape von Immobiliendaten (Zimmeranzahl, Wohnfläche, Preis, PLZ & Ort) via Scrapy
* Speicherung der Daten direkt in MongoDB (MongoDB Atlas)
* Automatische Bereinigung der MongoDB Collection beim Crawl-Start
* Bot-Detection-Problematik (z.B. keine JavaScript-Elemente wie ÖV-Distanzen)
* Logging aller Crawl-Aktivitäten und Fehlerbehandlung integriert

## Model Training

* Preprocessing: Feature Engineering, One-Hot-Encoding
* Vergleich von Regressionsmodellen (Random Forest, XGBoost, Gradient Boosting, Lineare Regression)
* Metriken: MAE, MSE, RMSE, R²
* Bestes Modell: Random Forest mit MAE von 688.30 CHF
* Speichern des Modells als Pickle-Datei (`immoscout_model.pkl`)

## Azure Blob Storage

* Automatisierter Upload des Modells zu Azure Blob Storage
* Versionierung des Modells über Azure Container
* Zugriff über `AZURE_STORAGE_CONNECTION_STRING` als Secret / Umgebungsvariable für CI/CD & Docker

## GitHub Actions

* Automatisierte Pipeline:
    * Scrape von ImmoScout24
    * Load in MongoDB
    * Modelltraining und Evaluation
    * Upload in Azure Blob Storage
    * Docker-Build und Push in GitHub Container Registry (ghcr.io)
    * Deployment auf Azure App Service (Linux)
* Getrennte Workflows: `ModelOps (Update Model)` & `ModelOps (Build, Deploy)`

## App

* **Backend:** Python Flask (`backend/app.py`) mit REST API zur Preisprognose
* **Frontend:** HTML/CSS (Bootstrap) + Vanilla JavaScript
* Responsives UI für Desktop und Mobile
* Formular zur Eingabe von: Zimmeranzahl, Wohnfläche, PLZ + Ort
* Ausgabe der Preisprognose direkt im Frontend

## Deployment mit Docker

* Dockerfile mit Flask + Gunicorn Setup
* Nutzung von Azure Secrets und GitHub Actions für Deployment
* Deployment-Umgebung: Azure Web App Service (Linux)


## Weitere Hinweise

* Logging und sauberes Error-Handling wurden in Spider, Modelltraining und Flask-App umgesetzt.
* Clean Code Prinzipien mit sinnvoller Ordnerstruktur und gut dokumentierten Commits
* Secrets für Azure & MongoDB werden über GitHub Actions sicher eingebunden
* App wurde erfolgreich mobil getestet (responsive)
