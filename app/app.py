from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Modell laden
model = joblib.load('./best_model.joblib')

# Wenn Linear Regression: kommt Tuple (model, scaler)
if isinstance(model, tuple):
    scaler = model[1]
    model = model[0]
else:
    scaler = None

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        rooms = float(request.form['rooms'])
        size_m2 = float(request.form['size_m2'])
        plz_code = float(request.form['plz_code'])

        X = np.array([[rooms, size_m2, plz_code]])

        if scaler:
            X = scaler.transform(X)

        price = model.predict(X)[0]
        prediction = round(price, 2)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
