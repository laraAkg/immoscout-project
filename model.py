import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# ---------------------------------
# 1. Daten laden & vorbereiten
# ---------------------------------
# Lade die JSON-Daten
with open('immo_spider/immoscout_listings.json', 'r') as f:
    data = json.load(f)

# Wandeln Sie die JSON-Daten in einen DataFrame um
df = pd.DataFrame(data)

# ---------------------------------
# 2. Datenvorbereitung
# ---------------------------------
# Stellen sicher, dass alle numerischen Werte als Zahlen interpretiert werden
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['rooms'] = pd.to_numeric(df['rooms'], errors='coerce')
df['size'] = pd.to_numeric(df['size'], errors='coerce')
df['postal_code'] = pd.to_numeric(df['postal_code'], errors='coerce')

# Entfernen von Zeilen mit fehlenden Werten
df = df.dropna(subset=['price', 'rooms', 'size'])

# Features und Zielvariable definieren
X = df[['rooms', 'size', 'postal_code']]  # Wir nehmen "rooms", "size", "postal_code" als Features
y = df['price']  # Wir nehmen den "price" als Zielvariable

# ---------------------------------
# 3. Train-Test-Split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------
# 4. Modelle trainieren
# ---------------------------------
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression(),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
}

# ---------------------------------
# 5. Modellbewertung
# ---------------------------------
best_model = None
best_mae = float('inf')
best_model_name = ""

# Trainiere jedes Modell und bewerten Sie die Leistung
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Berechnung der Fehlerkennzahlen
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name} - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")
    
    # Wählen Sie das Modell mit dem niedrigsten MAE aus
    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_model_name = model_name

print(f"\nDas beste Modell ist {best_model_name} mit einem MAE von {best_mae}")

# Speichern des besten Modells
joblib.dump(best_model, 'best_immoscout_model.joblib')
