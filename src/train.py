import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load dataset
data = pd.read_excel("data/Soil Moisture.xlsx")

# Define features and target (MUST match everywhere)
FEATURES = ["Temperature", "Air Humidity", "Pressure"]
TARGET = "Soil Moisture"

X = data[FEATURES]
y = data[TARGET]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(model, "model/model.pkl")

print("✅ Training completed. Model saved to model/model.pkl")