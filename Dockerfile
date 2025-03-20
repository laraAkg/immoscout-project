# Optimiertes Dockerfile für Runtime-Container
FROM python:3.12-slim

WORKDIR /app

# Dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App und fertige Artefakte
COPY app ./app
COPY immo_spider/immoscout_listings.json ./immo_spider/immoscout_listings.json
COPY best_immoscout_model.joblib ./best_immoscout_model.joblib

# Azure-ready Startup mit Gunicorn
CMD ["gunicorn", "app:app", "--chdir", "./app", "--bind", "0.0.0.0:5000"]

EXPOSE 5000