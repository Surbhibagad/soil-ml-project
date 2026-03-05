import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
data = pd.read_excel("data/Soil Moisture.xlsx")

print("Columns:", data.columns.tolist())


# =========================
# Feature Engineering
# =========================

data["Heat_Dryness"] = data["Temperature"] * (100 - data["Air Humidity"])

data["Temp_Humidity"] = data["Temperature"] * data["Air Humidity"]

data["Humidity_Squared"] = data["Air Humidity"] ** 2

data["Temp_Squared"] = data["Temperature"] ** 2


# Features and Target
FEATURES = [
    "Temperature",
    "Air Humidity",
    "Pump Data",
    "Heat_Dryness",
    "Temp_Humidity",
    "Humidity_Squared",
    "Temp_Squared"
]

TARGET = "Soil Moisture"


X = data[FEATURES]
y = data[TARGET]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# Model + Hyperparameter tuning
# =========================

model = GradientBoostingRegressor()

param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5]
}


grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_


# Predictions
y_pred = best_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


# Create folders
os.makedirs("model", exist_ok=True)
os.makedirs("data/split", exist_ok=True)


# Save model
joblib.dump(best_model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")


# Save test data
X_test.to_csv("data/split/X_test.csv", index=False)
y_test.to_csv("data/split/y_test.csv", index=False)


# Output
print("\nTraining Completed")
print("------------------")
print("Best Parameters:", grid.best_params_)
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")
print("Model saved: model/model.pkl")
