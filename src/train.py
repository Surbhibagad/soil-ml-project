import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
data = pd.read_excel("data/Soil Moisture.xlsx")

print("Columns:", data.columns.tolist())


# Feature Engineering
data["Heat_Dryness"] = data["Temperature"] * (100 - data["Air Humidity"])


# Features and Target
FEATURES = [
    "Temperature",
    "Air Humidity",
    "Pump Data",
    "Heat_Dryness"
]

TARGET = "Soil Moisture"


X = data[FEATURES]
y = data[TARGET]


# Train-test split (IMPORTANT: before scaling)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# Predictions
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


# Create folders
os.makedirs("model", exist_ok=True)
os.makedirs("data/split", exist_ok=True)


# Save model and scaler
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")


# Save RAW test data (NOT scaled)
X_test.to_csv("data/split/X_test.csv", index=False)
y_test.to_csv("data/split/y_test.csv", index=False)


# Print results
print("\nTraining Completed")
print("------------------")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")
print("Model saved: model/model.pkl")
print("Scaler saved: model/scaler.pkl")
