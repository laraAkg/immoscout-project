"""
This script is a Flask-based web application for predicting real estate prices based on user input.
It integrates with Azure Blob Storage to fetch the latest machine learning model and uses it for predictions.

Modules:
- flask: For creating the web application and handling HTTP requests.
- joblib: For loading the machine learning model.
- numpy: For numerical operations.
- csv: For reading postal code and location data.
- pandas: For data manipulation and preparation.
- os: For accessing environment variables.
- azure.storage.blob: For interacting with Azure Blob Storage.

Environment Variables:
- AZURE_STORAGE_CONNECTION_STRING: Connection string for accessing Azure Blob Storage.

Routes:
- `/`: Renders the main page with a form for user input.
- `/predict`: Handles form submissions, validates input, and returns the predicted price or an error message.

Key Features:
- Downloads the latest `.pkl` model file from Azure Blob Storage.
- Loads postal code and location mappings from a CSV file.
- Validates user input for room count, size, and postal code.
- Prepares input data for the machine learning model and makes predictions.
- Displays the predicted price or an error message on the main page.

Error Handling:
- Raises a `FileNotFoundError` if no `.pkl` files are found in the Blob Storage.
- Handles invalid user input with appropriate error messages.

Usage:
Run the script to start the Flask application. Access the application in a web browser at `http://<host>:5000/`.

"""

from flask import Flask, render_template, request
import joblib
import numpy as np
import csv
import pandas as pd
import os
from azure.storage.blob import BlobServiceClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="../frontend/templates", static_url_path="/", static_folder="../frontend/static")

AZURE_BLOB_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER = "immoscout-models"

if not AZURE_BLOB_CONN_STR:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not set!")

blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN_STR)
blob_client = blob_service_client.get_container_client(BLOB_CONTAINER)

blobs = list(blob_client.list_blobs())
filtered_blobs = [b for b in blobs if b.name.endswith(".pkl")]

if not filtered_blobs:
    raise FileNotFoundError("No .pkl files found in the Blob Storage!")

latest_blob = sorted(
    filtered_blobs,
    key=lambda b: int(b.name.split('-')[-1].replace('.pkl', ''))
)[-1]

with open("model/immoscout_model.pkl", "wb") as download_file:
    blob_stream = blob_client.download_blob(latest_blob.name)
    download_file.write(blob_stream.readall())

model = joblib.load("model/immoscout_model.pkl")

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
        logger.info("📥 Received prediction request.")
        rooms = float(request.form["rooms"])
        size = float(request.form["size"])
        postal_code = request.form["postal_code"]

        if not postal_code.isdigit():
            raise ValueError("The postal code must be a number.")

        postal_code = int(postal_code)

        if rooms <= 0 or size <= 0 or postal_code not in plz_ort:
            raise ValueError("Invalid input values.")

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
        prediction = f"Predicted price: CHF {predicted_price:.2f}"
        logger.info(f"✅ Prediction successful: {prediction}")

    except Exception as e:
        error_message = f"Error: {str(e)}"
        logger.error(f"❌ Prediction failed: {error_message}")

    return render_template(
        "index.html",
        prediction=prediction,
        error_message=error_message,
        plz_ort=plz_ort,
    )

if __name__ == "__main__":
    logger.info("🚀 Starting Flask application...")
    app.run(debug=False, host="0.0.0.0", port=5000)