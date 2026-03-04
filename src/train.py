import pandas as pd
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
# Heat dryness = drying power of air
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


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)


# Train model (Better than Linear Regression)
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Evaluation
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)


# Create folders
os.makedirs("model", exist_ok=True)
os.makedirs("data/split", exist_ok=True)


# Save model and scaler
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")


# Save test data
pd.DataFrame(X_test, columns=FEATURES).to_csv(
    "data/split/X_test.csv", index=False
)

y_test.to_csv("data/split/y_test.csv", index=False)


# Print results
print("\nTraining Completed")
print("------------------")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")
print("Model saved: model/model.pkl")
print("Scaler saved: model/scaler.pkl")