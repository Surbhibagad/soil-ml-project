import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# -----------------------------
# Load data
# -----------------------------
data = pd.read_excel("data/Soil Moisture.xlsx")

print("Columns:", data.columns.tolist())


# -----------------------------
# Feature Engineering
# -----------------------------

# Lag feature
data["moisture_lag_1"] = data["Soil Moisture"].shift(1)

# Change
data["moisture_delta"] = data["Soil Moisture"].diff()

# Evaporation index
data["evapo_index"] = (
    data["Temperature"] * (100 - data["Air Humidity"]) / 100
)

# Rolling mean
data["moisture_mean_3"] = data["Soil Moisture"].rolling(3).mean()

# Remove NaN rows
data = data.dropna()


# -----------------------------
# Features / Target
# -----------------------------

FEATURES = [
    "Temperature",
    "Air Humidity",
    "Pump Data",
    "moisture_lag_1",
    "moisture_delta",
    "evapo_index",
    "moisture_mean_3"
]

TARGET = "Soil Moisture"


X = data[FEATURES]
y = data[TARGET]


# -----------------------------
# Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Train
# -----------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# -----------------------------
# Evaluate
# -----------------------------

pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)
rmse = mse ** 0.5

print("RMSE:", rmse)


# -----------------------------
# Save
# -----------------------------

os.makedirs("model", exist_ok=True)
os.makedirs("data/split", exist_ok=True)

joblib.dump(model, "model/model.pkl")

X_test.to_csv("data/split/X_test.csv", index=False)
y_test.to_csv("data/split/y_test.csv", index=False)

print("Training completed")
