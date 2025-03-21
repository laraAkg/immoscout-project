# Basisimage mit Python 3.12
FROM python:3.12.7

# Arbeitsverzeichnis auf backend setzen
WORKDIR /usr/src/app/backend

# Code kopieren (komplettes Projekt, damit auch andere Ordner verfügbar sind)
COPY . .

# Abhängigkeiten installieren
RUN pip install --upgrade pip && pip install -r /usr/src/app/requirements.txt

# Port öffnen
EXPOSE 5000

# App starten (Production-ready mit gunicorn)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
