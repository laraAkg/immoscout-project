"""
Handles the prediction of real estate prices based on user input.

Route:
    /predict (POST)

Functionality:
    - Extracts input values (rooms, size, postal_code) from the form data.
    - Validates the input values for correctness:
        - Ensures `rooms` and `size` are positive numbers.
        - Ensures `postal_code` is a numeric value and exists in the `plz_ort` dictionary.
    - Prepares the input data for the prediction model.
    - Uses the `model` to predict the price of the property.
    - Returns the predicted price or an error message if validation or prediction fails.

Template:
    Renders the `index.html` template with the following context:
        - `prediction`: The predicted price (if successful).
        - `error_message`: An error message (if any validation or prediction error occurs).
        - `plz_ort`: A dictionary mapping postal codes to locations.

Exceptions:
    - Catches any exceptions during input validation or prediction and displays an appropriate error message.

Returns:
    Rendered HTML template with prediction or error message.
"""
from flask import Flask, render_template, request
import joblib
import numpy as np
import csv
import pandas as pd

app = Flask(__name__)

model = joblib.load("immo_spider/best_immoscout_model.joblib")

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

        # Erstelle ein DataFrame mit den Eingabedaten
        input_df = pd.DataFrame(
            [[rooms, size, postal_code]],
            columns=["rooms", "size", "postal_code"]
        )

        # Wende One-Hot-Encoding auf die Postleitzahl an
        input_df["postal_code"] = input_df["postal_code"].astype(str)
        input_df = pd.get_dummies(input_df, columns=["postal_code"], prefix="plz")

        # Stelle sicher, dass die Spaltenreihenfolge mit dem Modell übereinstimmt
        for col in model.feature_names_in_:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model.feature_names_in_]

        # Vorhersage durchführen
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
    app.run(debug=True)
