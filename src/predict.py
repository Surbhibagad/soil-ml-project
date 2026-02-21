import joblib
import pandas as pd
import sys
import os

# Load trained model
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Example:
# docker run soil-ml 25 60 1013

if len(sys.argv) != 4:
    print("Usage: python src/predict.py <temp> <humidity> <pressure>")
    sys.exit(1)

temp = float(sys.argv[1])
humidity = float(sys.argv[2])
pressure = float(sys.argv[3])

# Create DataFrame
input_data = pd.DataFrame(
    [[temp, humidity, pressure]],
    columns=["temperature", "humidity", "pressure"]
)

# Predict
prediction = model.predict(input_data)

print("Predicted Soil Moisture:", float(prediction[0]))