import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_excel("data/Soil Moisture.xlsx")

print("Columns:", data.columns.tolist())

# Features and Target
FEATURES = ["Temperature", "Air Humidity", "Pump Data"]
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

# Create folders
os.makedirs("model", exist_ok=True)
os.makedirs("data/split", exist_ok=True)

# Save trained model
joblib.dump(model, "model/model.pkl")

# Save test data
X_test.to_csv("data/split/X_test.csv", index=False)
y_test.to_csv("data/split/y_test.csv", index=False)

print("✅ Training completed")
print("✅ Model saved: model/model.pkl")
print("✅ Test data saved: data/split/")
  