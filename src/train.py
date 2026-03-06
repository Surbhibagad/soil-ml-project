import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# Load dataset
# =========================
data = pd.read_excel("data/Soil Moisture.xlsx")

print("Columns:", data.columns.tolist())
print("Shape:", data.shape)

# =========================
# Features and target
# =========================
FEATURES = ["Temperature", "Air Humidity", "Pump Data"]
TARGET = "Soil Moisture"

X = data[FEATURES]
y = data[TARGET]

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# =========================
# Build pipeline
# =========================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor(random_state=42))
])

# =========================
# Hyperparameter tuning
# =========================
param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 4]
}

print("\nRunning GridSearchCV...")

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

best_pipeline = grid.best_estimator_

# =========================
# Evaluation
# =========================
y_pred = best_pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# =========================
# Save model
# =========================
os.makedirs("model", exist_ok=True)
os.makedirs("data/split", exist_ok=True)

joblib.dump(best_pipeline, "model/full_pipeline.pkl")

X_test.to_csv("data/split/X_test.csv", index=False)
y_test.to_csv("data/split/y_test.csv", index=False)

print("\n========== Training Completed ==========")
print("Best Parameters:", grid.best_params_)
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")
print("Pipeline saved: model/full_pipeline.pkl")
