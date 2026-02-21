import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_excel("data/Soil Moisture.xlsx")

# Features and Target
FEATURES = ["Temperature", "Humidity", "pump data"]
TARGET = "Soil Moisture"

X = data[FEATURES]
y = data[TARGET]

# Load model
model = joblib.load("model/model.pkl")

# Predict
y_pred = model.predict(X)

# Evaluate
print("📊 Model Evaluation")
print("MSE:", mean_squared_error(y, y_pred))
print("R2 Score:", r2_score(y, y_pred))