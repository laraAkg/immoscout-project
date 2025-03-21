# Basisimage mit Python 3.12
FROM python:3.12.7

# Arbeitsverzeichnis setzen
WORKDIR /usr/src/app

# Code kopieren
COPY . .

# Abhängigkeiten installieren
RUN pip install --upgrade pip && pip install -r requirements.txt

# Flask App-Start definieren
ENV FLASK_APP=/usr/src/app/app.py
EXPOSE 5000

# App starten (Dozenten-Style mit Flask)
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
