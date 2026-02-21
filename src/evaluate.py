import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_excel("data/Soil Moisture.xlsx")

# Define features and target
FEATURES = ["Temperature", "Air Humidity", "Pressure"]
TARGET = "Soil Moisture"

X = data[FEATURES]
y = data[TARGET]

# Load trained model
model = joblib.load("model/model.pkl")

# Predict
y_pred = model.predict(X)

# Evaluate
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("📊 Model Evaluation Results")
print("MSE:", mse)
print("R2 Score:", r2)