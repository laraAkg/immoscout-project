import pandas as pd
"""
This script trains and evaluates multiple regression models to predict real estate prices 
based on features such as the number of rooms, size, and postal code. The best-performing 
model is saved for future use.

Workflow:
    1. Load real estate data from a JSON file.
    2. Preprocess the data by converting columns to numeric and handling missing values.
    3. Split the data into training and testing sets.
    4. Train and evaluate multiple regression models (Random Forest, Gradient Boosting, 
       Linear Regression, and XGBoost).
    5. Select the best model based on the lowest Mean Absolute Error (MAE).
    6. Save the best model to a file using joblib.

Outputs:
    - Prints the performance metrics (MAE, MSE, RMSE, R2) for each model.
    - Prints the name and MAE of the best-performing model.
    - Saves the best model to a file named "best_immoscout_model.joblib".
"""
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

with open("immo_spider/immoscout_listings.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")
df["size"] = pd.to_numeric(df["size"], errors="coerce")
df["postal_code"] = pd.to_numeric(df["postal_code"], errors="coerce")

df = df.dropna(subset=["price", "rooms", "size"])

X = df[["rooms", "size", "postal_code"]]
y = df["price"]

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
    rmse = mse**0.5
    r2 = r2_score(y_test, y_pred)

    print(f"{model_name} - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")

    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_model_name = model_name

print(f"\nDas beste Modell ist {best_model_name} mit einem MAE von {best_mae}")

joblib.dump(best_model, "best_immoscout_model.joblib")
