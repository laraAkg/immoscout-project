import os
import pandas as pd
import pickle  # Verwende pickle anstelle von joblib
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb



MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["immoscout_db"]
collection = db["listings"]

data = list(collection.find())
df = pd.DataFrame(data)

df.drop(columns=["_id"], errors="ignore", inplace=True)
df.drop(columns=["location", "canton", "page"], errors="ignore", inplace=True)

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")
df["size"] = pd.to_numeric(df["size"], errors="coerce")
df["postal_code"] = df["postal_code"].astype(str)

df = df.dropna(subset=["price", "rooms", "size", "postal_code"])

df = pd.get_dummies(df, columns=["postal_code"], prefix="plz")

X = df.drop(columns=["price"])
y = df["price"]

if X.shape[0] == 0:
    print("❌ Keine Datensätze vorhanden. Training wird abgebrochen.")
    client.close()
    exit(1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression(),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42),
}

best_model = None
best_mae = float("inf")
best_model_name = ""

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

# Speichern als .pkl-Datei
os.makedirs("model", exist_ok=True)  # Verzeichnis erstellen, falls es nicht existiert
with open("model/immoscout_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("💾 Modell wurde als 'model/immoscout_model.pkl' gespeichert")

client.close()