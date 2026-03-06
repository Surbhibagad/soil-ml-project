import pandas as pd
import joblib
import sys
import os

# =========================
# Load pipeline
# =========================
MODEL_PATH = "model/full_pipeline.pkl"

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model not found. Run train.py first.")
    sys.exit(1)

pipeline = joblib.load(MODEL_PATH)

# =========================
# Input arguments
# =========================
if len(sys.argv) != 4:
    print("Usage: python predict.py <Temperature> <Humidity> <Pump>")
    sys.exit(1)

try:
    temp = float(sys.argv[1])
    humidity = float(sys.argv[2])
    pump = int(sys.argv[3])
except ValueError:
    print("ERROR: Inputs must be numeric")
    sys.exit(1)

# =========================
# Create dataframe
# =========================
input_df = pd.DataFrame(
    [[temp, humidity, pump]],
    columns=["Temperature", "Air Humidity", "Pump Data"]
)

# =========================
# Prediction
# =========================
prediction = pipeline.predict(input_df)[0]

# =========================
# Status logic
# =========================
if prediction < 300:
    status = "Dry"
    irrigation = "Immediately"
    water = "18 Liters"
    crop = "Millet, Barley"

elif prediction < 550:
    status = "Moderate"
    irrigation = "After 6 Hours"
    water = "10 Liters"
    crop = "Wheat, Maize"

else:
    status = "Wet"
    irrigation = "After 24 Hours"
    water = "5 Liters"
    crop = "Rice, Sugarcane"

# =========================
# Output
# =========================
print("\nSoil Moisture Prediction")
print("-------------------------")
print(f"Temperature : {temp}°C")
print(f"Humidity    : {humidity}%")
print(f"Pump        : {'ON' if pump else 'OFF'}")

print(f"\nPredicted Soil Moisture : {prediction:.2f}")
print(f"Status                  : {status}")
print(f"Irrigation Time         : {irrigation}")
print(f"Water Needed            : {water}")
print(f"Suitable Crops          : {crop}")
