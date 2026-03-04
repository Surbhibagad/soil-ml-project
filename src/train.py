import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
data = pd.read_excel("data/Soil Moisture.xlsx")

print("Columns:", data.columns.tolist())


# Features & Target
FEATURES = ["Temperature", "Air Humidity", "Pump Data"]
TARGET = "Soil Moisture"

X = data[FEATURES]
y = data[TARGET]


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train Model
model = LinearRegression()
model.fit(X_train, y_train)


# Predictions
pred = model.predict(X_test)


# Metrics
mse = mean_squared_error(y_test, pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, pred)

print("RMSE:", rmse)
print("R2 Score:", r2)


# Save folders
os.makedirs("model", exist_ok=True)
os.makedirs("data/split", exist_ok=True)


# Save model
joblib.dump(model, "model/model.pkl")


# Save test data
X_test.to_csv("data/split/X_test.csv", index=False)
y_test.to_csv("data/split/y_test.csv", index=False)


# Save metrics
with open("model/metrics.txt", "w") as f:
    f.write(f"RMSE: {rmse}\n")
    f.write(f"R2: {r2}\n")


print("Training completed")
print("Model saved")
