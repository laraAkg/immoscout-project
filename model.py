import os
import pandas as pd
import joblib
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# ENV-Variablen
MONGO_URI = os.getenv("MONGO_URI") or os.getenv("MONGO_URI")  # Flexibler

if not MONGO_URI:
    raise ValueError("❌ MONGO_URI ist nicht gesetzt! Bitte ENV-Variable oder Secret einrichten.")
client = MongoClient(MONGO_URI)

db = client["immoscout_db"]
collection = db["listings"]

data = list(collection.find())
df = pd.DataFrame(data)

# _id & unnötige Spalten droppen
df.drop(columns=["_id"], errors="ignore", inplace=True)
df.drop(columns=["location", "canton", "page"], errors="ignore", inplace=True)

# Datentypen sauber konvertieren
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")
df["size"] = pd.to_numeric(df["size"], errors="coerce")
df["postal_code"] = df["postal_code"].astype(str)

# NaNs entfernen
df = df.dropna(subset=["price", "rooms", "size", "postal_code"])

# One-Hot-Encoding für PLZ
df = pd.get_dummies(df, columns=["postal_code"], prefix="plz")

# Features und Target
X = df.drop(columns=["price"])
y = df["price"]

# Datenprüfung
if X.shape[0] == 0:
    print("❌ Keine Datensätze vorhanden. Training wird abgebrochen.")
    client.close()
    exit(1)

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelle definieren
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression(),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42),
}

best_model = None
best_mae = float("inf")
best_model_name = ""

# Training & Evaluation
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"{model_name} - MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")

    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_model_name = model_name

print(f"\n✅ Das beste Modell ist {best_model_name} mit einem MAE von {best_mae:.2f}")

# Speichern im Root-Verzeichnis für save.py
joblib.dump(best_model, "../best_immoscout_model.joblib")
print("💾 Modell wurde als '../best_immoscout_model.joblib' gespeichert")

client.close()
