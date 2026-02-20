import joblib
import pandas as pd
import sys

# Load trained model
model = joblib.load("model.pkl")

# Example: take input from command line
# python src/predict.py 25 60 1013
# (temperature, humidity, pressure)

if len(sys.argv) != 4:
    print("Usage: python predict.py <temp> <humidity> <pressure>")
    sys.exit(1)

temp = float(sys.argv[1])
humidity = float(sys.argv[2])
pressure = float(sys.argv[3])

# Create DataFrame (same format as training data)
input_data = pd.DataFrame([[temp, humidity, pressure]],
                          columns=["temperature", "humidity", "pressure"])

# Predict
prediction = model.predict(input_data)

print("Predicted Soil Moisture:", prediction[0])