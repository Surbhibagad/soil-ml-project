import joblib
import pandas as pd
import sys


# Load model
try:
    model = joblib.load("model/model.pkl")
except Exception as e:
    print("❌ Cannot load model:", e)
    sys.exit(1)


# Validate input
if len(sys.argv) != 4:
    print("Usage: python predict.py <Temperature> <Humidity> <PumpData>")
    sys.exit(1)


try:
    temp = float(sys.argv[1])
    humidity = float(sys.argv[2])
    pump = float(sys.argv[3])
except ValueError:
    print("❌ Inputs must be numeric")
    sys.exit(1)


# Range validation (optional)
if not (-20 <= temp <= 60):
    print("❌ Temperature out of range")
    sys.exit(1)

if not (0 <= humidity <= 100):
    print("❌ Humidity must be 0-100")
    sys.exit(1)

if pump < 0:
    print("❌ Pump must be positive")
    sys.exit(1)


# Prepare input
input_data = pd.DataFrame(
    [[temp, humidity, pump]],
    columns=["Temperature", "Air Humidity", "Pump Data"]
)


# Predict
prediction = model.predict(input_data)

print(f"🌱 Predicted Soil Moisture: {prediction[0]:.2f}")