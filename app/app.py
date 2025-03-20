from flask import Flask, render_template, request
import joblib
import numpy as np
import csv

app = Flask(__name__)

model = joblib.load('best_immoscout_model.joblib')

# Lade die PLZ + Ort Daten aus der CSV-Datei
plz_ort = {}
with open('app/plz_ort.csv', mode='r') as infile:
    reader = csv.reader(infile)
    next(reader)  # Überspringe die Header-Zeile
    for rows in reader:
        plz = int(rows[0])
        ort = rows[1]
        plz_ort[plz] = ort

@app.route('/')
def index():
    return render_template('index.html', plz_ort=plz_ort)

@app.route('/predict', methods=['POST'])
def predict():
    error_message = None
    prediction = None
    try:
        rooms = int(request.form['rooms'])
        size = float(request.form['size'])
        postal_code = request.form['postal_code']

        # Validierung ob nur Zahlen eingegeben wurden
        if not postal_code.isdigit():
            raise ValueError("Die Postleitzahl muss eine Zahl sein.")

        postal_code = int(postal_code)

        if rooms <= 0 or size <= 0 or postal_code not in plz_ort:
            raise ValueError("Ungültige Eingabewerte.")

        input_data = np.array([rooms, size, postal_code]).reshape(1, -1)
        predicted_price = model.predict(input_data)[0]
        prediction = f"Vorhergesagter Preis: CHF {predicted_price:.2f}"

    except Exception as e:
        error_message = f"Fehler: {str(e)}"

    return render_template('index.html', prediction=prediction, error_message=error_message, plz_ort=plz_ort)

if __name__ == '__main__':
    app.run(debug=True)