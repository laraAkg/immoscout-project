"""
This script connects to a MongoDB database, retrieves real estate listings data, 
cleans and preprocesses the data, trains multiple regression models to predict 
property prices, and saves the best-performing model to a file.

Modules:
    - os: Provides functions to interact with the operating system.
    - pandas: Used for data manipulation and analysis.
    - pickle: Used for serializing and saving the trained model.
    - logging: Used for logging messages.
    - pymongo: Used for connecting to and interacting with MongoDB.
    - sklearn: Provides tools for model training, evaluation, and splitting data.
    - xgboost: Provides the XGBoost regression model.

Constants:
    - MONGO_URI: The MongoDB connection URI, retrieved from environment variables.

Functions:
    - None

Workflow:
    1. Connects to a MongoDB database and retrieves data from the "listings" collection.
    2. Cleans and preprocesses the data, including handling missing values and encoding categorical variables.
    3. Splits the data into training and testing sets.
    4. Trains multiple regression models (Random Forest, Gradient Boosting, Linear Regression, XGBoost).
    5. Evaluates the models using metrics such as MAE, MSE, RMSE, and R2 score.
    6. Selects the best-performing model based on the lowest MAE.
    7. Saves the best model to a file named "immoscout_model.pkl".
    8. Closes the MongoDB connection.

Logging:
    - Logs the number of records retrieved from MongoDB.
    - Logs the number of records remaining after data cleaning.
    - Logs the performance metrics for each model.
    - Logs the best-performing model and its MAE.

Error Handling:
    - Raises a ValueError if the MONGO_URI environment variable is not set.
    - Exits the script if no data is available for training after preprocessing.

"""


import os
import pandas as pd
import pickle
import logging
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI is not set!")

client = MongoClient(MONGO_URI)
db = client["immoscout_db"]
collection = db["listings"]

data = list(collection.find())
df = pd.DataFrame(data)

logger.info(f"✅ Retrieved {len(df)} records from MongoDB.")
df.drop(columns=["_id"], errors="ignore", inplace=True)
df.drop(columns=["location", "canton", "page"], errors="ignore", inplace=True)

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")
df["size"] = pd.to_numeric(df["size"], errors="coerce")
df["postal_code"] = df["postal_code"].astype(str)

df = df.dropna(subset=["price", "rooms", "size", "postal_code"])
logger.info(f"✅ Data cleaned. Remaining records: {len(df)}.")

df = pd.get_dummies(df, columns=["postal_code"], prefix="plz")

X = df.drop(columns=["price"])
y = df["price"]

if X.shape[0] == 0:
    logger.error("❌ No data available for training. Exiting...")
    client.close()
    exit(1)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Model training
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression(),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42),
}

best_model = None
best_mae = float("inf")
best_model_name = ""

logger.info("🚀 Starting model training...")
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_test, y_pred)

    logger.info(
        f"{model_name} - MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}"
    )

    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_model_name = model_name

logger.info(f"✅ Best model: {best_model_name} with MAE: {best_mae:.2f}")

os.makedirs("model", exist_ok=True)
with open("model/immoscout_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

client.close()
logger.info("🔒 MongoDB connection closed.")