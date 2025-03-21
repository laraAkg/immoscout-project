from flask import Flask, render_template, request
import joblib
import numpy as np
import csv
import pandas as pd
import os
from azure.storage.blob import BlobServiceClient

app = Flask(__name__)

# Azure Blob Storage Setup
AZURE_BLOB_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER = "immoscout-models"

# Modell aus Azure Blob Storage laden
blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN_STR)
blob_client = blob_service_client.get_container_client(BLOB_CONTAINER)

# Automatisch das neueste Modell finden
blobs = list(blob_client.list_blobs())
filtered_blobs = [b for b in blobs if b.name.endswith(".pkl")]  # Nur .pkl-Dateien berücksichtigen

if not filtered_blobs:
    raise FileNotFoundError("❌ Keine .pkl-Dateien im Blob Storage gefunden!")

latest_blob = sorted(
    filtered_blobs,
    key=lambda b: int(b.name.split('-')[-1].replace('.pkl', ''))
)[-1]

with open("model/immoscout_model.pkl", "wb") as download_file:
    blob_stream = blob_client.download_blob(latest_blob.name)
    download_file.write(blob_stream.readall())

model = joblib.load("model/immoscout_model.pkl")

# PLZ-Daten laden
plz_ort = {}
with open("data/plz_ort.csv", mode="r") as infile:
    reader = csv.reader(infile)
    next(reader)
    for rows in reader:
        plz = int(rows[0])
        ort = rows[1]
        plz_ort[plz] = ort


@app.route("/")
def index():
    return render_template("index.html", plz_ort=plz_ort)


@app.route("/predict", methods=["POST"])
def predict():
    error_message = None
    prediction = None
    try:
        rooms = float(request.form["rooms"])
        size = float(request.form["size"])
        postal_code = request.form["postal_code"]

        if not postal_code.isdigit():
            raise ValueError("Die Postleitzahl muss eine Zahl sein.")

        postal_code = int(postal_code)

        if rooms <= 0 or size <= 0 or postal_code not in plz_ort:
            raise ValueError("Ungültige Eingabewerte.")

        input_df = pd.DataFrame(
            [[rooms, size, postal_code]],
            columns=["rooms", "size", "postal_code"]
        )

        input_df["postal_code"] = input_df["postal_code"].astype(str)
        input_df = pd.get_dummies(input_df, columns=["postal_code"], prefix="plz")

        for col in model.feature_names_in_:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model.feature_names_in_]

        predicted_price = model.predict(input_df)[0]
        prediction = f"Vorhergesagter Preis: CHF {predicted_price:.2f}"

    except Exception as e:
        error_message = f"Fehler: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        error_message=error_message,
        plz_ort=plz_ort,
    )

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)