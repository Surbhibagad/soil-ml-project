import joblib
import pandas as pd
import sys

# Load model
model = joblib.load("model/model.pkl")

# Check input
# Example:
# docker run image 30 60 1
# (Temperature Humidity pump_data)

if len(sys.argv) != 4:
    print("Usage: python predict.py <Temperature> <Air Humidity> <Pump_Data>")
    sys.exit(1)

temp = float(sys.argv[1])
humidity = float(sys.argv[2])
pump = float(sys.argv[3])

# Create input dataframe
input_data = pd.DataFrame(
    [[temp, humidity, pump]],
    columns=["Temperature", "Air Humidity", "Pump Data"]
)

# Predict
prediction = model.predict(input_data)

print("🌱 Predicted Soil Moisture:", prediction[0])