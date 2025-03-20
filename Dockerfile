# Dockerfile für Scrapy + Modelltraining + Flask-App
FROM python:3.12-slim

# Verzeichnisse anlegen
WORKDIR /app

# Requirements kopieren und installieren
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Scrapy Spider und alle Skripte in das Image kopieren
COPY ./immo_spider ./immo_spider
COPY ./model.py ./model.py
COPY ./data ./data
COPY ./app ./app

# Scrapy Crawl ausführen + Modell trainieren (on build)
RUN scrapy crawl immoscout_spider -o immo_spider/immoscout_listings.json && python model.py

# Port für Azure App Service exposen
EXPOSE 5000

# Flask App starten mit gunicorn
WORKDIR /app/app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
