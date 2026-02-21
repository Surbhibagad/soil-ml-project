import joblib
import pandas as pd
import sys

# Load trained model
model = joblib.load("model/model.pkl")

# Check input arguments
# Usage: docker run image temp humidity pressure
if len(sys.argv) != 4:
    print("Usage: python predict.py <Temperature> <AirHumidity> <Pressure>")
    sys.exit(1)

# Read inputs
temperature = float(sys.argv[1])
humidity = float(sys.argv[2])
pressure = float(sys.argv[3])

# Create input DataFrame (same format as training)
input_data = pd.DataFrame(
    [[temperature, humidity, pressure]],
    columns=["Temperature", "Air Humidity", "Pressure"]
)

# Predict
prediction = model.predict(input_data)

print("🌱 Predicted Soil Moisture:", prediction[0])