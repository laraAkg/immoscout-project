import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------
# 1. Daten laden & vorbereiten
# ---------------------------------
df = pd.read_csv('immoscout_clean.csv')

# PLZ extrahieren
df['plz_code'] = df['plz'].str.extract(r'(\d{4})').astype(float)

# NaNs entfernen
df = df.dropna(subset=['rooms', 'size_m2', 'plz_code', 'price_chf'])

# ---------------------------------
# 2. Features & Target
# ---------------------------------
X = df[['rooms', 'size_m2', 'plz_code']]
y = df['price_chf']

# ---------------------------------
# 3. Split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------
# 4. Modelle definieren
# ---------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_score = -float('inf')
results = []

# ---------------------------------
# 5. Modelle trainieren & vergleichen
# ---------------------------------
for name, model in models.items():
    # nur Linear Regression braucht Scaling
    if name == "LinearRegression":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Metriken
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    results.append({
        "model": name,
        "R2": round(r2, 3),
        "MAE": round(mae, 2),
        "MSE": round(mse, 2)
    })

    print(f"\n{name}: R2={r2:.3f}, MAE={mae:.2f}, MSE={mse:.2f}")

    # bestes Modell speichern
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_model_name = name
        if name == "LinearRegression":
            joblib.dump((model, scaler), "best_model.joblib")
        else:
            joblib.dump(model, "best_model.joblib")

# ---------------------------------
# 6. Ergebnisse loggen
# ---------------------------------
print("\nVergleich:")
print(pd.DataFrame(results))
print(f"\n➡️ Bestes Modell: {best_model_name} mit R² = {best_score:.3f}")
print("Modell wurde als best_model.joblib gespeichert.")
