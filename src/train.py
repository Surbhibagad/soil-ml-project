import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# Load dataset
# =========================
data = pd.read_excel("data/Soil Moisture.xlsx")

print("Columns:", data.columns.tolist())


# =========================
# Features and Target
# =========================
FEATURES = ["Temperature", "Air Humidity", "Pump Data"]
TARGET = "Soil Moisture"

X = data[FEATURES]
y = data[TARGET]


# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# =========================
# Build ML Pipeline
# =========================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ))
])


# =========================
# Train
# =========================
print("🚀 Training model...")
pipeline.fit(X_train, y_train)


# =========================
# Create folders
# =========================
os.makedirs("model", exist_ok=True)
os.makedirs("data/split", exist_ok=True)


# =========================
# Save model
# =========================
joblib.dump(pipeline, "model/model.pkl")


# =========================
# Save test data
# =========================
X_test.to_csv("data/split/X_test.csv", index=False)
y_test.to_csv("data/split/y_test.csv", index=False)


# =========================
# Done
# =========================
print("✅ Training completed")
print("✅ Pipeline model saved: model/model.pkl")
print("✅ Test data saved: data/split/")
