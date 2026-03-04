import joblib
import pandas as pd
import sys


# Load model
try:
    model = joblib.load("model/model.pkl")
    scaler = joblib.load("model/scaler.pkl")

except FileNotFoundError:
    print("Model or scaler not found. Train first.")
    sys.exit(1)

except Exception as e:
    print("Loading error:", e)
    sys.exit(1)


# Check arguments
if len(sys.argv) != 4:
    print("Usage: python predict.py <Temp> <Humidity> <Pump>")
    sys.exit(1)


# Convert inputs
try:
    temp = float(sys.argv[1])
    humidity = float(sys.argv[2])
    pump = float(sys.argv[3])

except ValueError:
    print("Inputs must be numbers")
    sys.exit(1)


# Validate ranges
if not (-20 <= temp <= 60):
    print("Temperature out of range")
    sys.exit(1)

if not (0 <= humidity <= 100):
    print("Humidity out of range")
    sys.exit(1)

if pump not in [0, 1]:
    print("Pump must be 0 or 1")
    sys.exit(1)


# Feature engineering
heat_dryness = temp * (100 - humidity)


# Prepare input
input_df = pd.DataFrame(
    [[temp, humidity, pump, heat_dryness]],
    columns=[
        "Temperature",
        "Air Humidity",
        "Pump Data",
        "Heat_Dryness"
    ]
)


# Scale input
input_scaled = scaler.transform(input_df)


# Predict
prediction = model.predict(input_scaled)[0]


# Interpretation
if prediction < 300:
    status = "Very Dry"
    advice = "Turn Pump ON Immediately"
    irrigation = "Now"
    water = "20 Liters"
    crop = "Millet, Barley"
    alert = "Drought Risk"

elif prediction < 450:
    status = "Moderate"
    advice = "Irrigation Recommended"
    irrigation = "After 6 Hours"
    water = "12 Liters"
    crop = "Wheat, Maize"
    alert = "Normal"

else:
    status = "Wet"
    advice = "No Irrigation Needed"
    irrigation = "After 24 Hours"
    water = "5 Liters"
    crop = "Rice, Sugarcane"
    alert = "Flood Risk"


# Output
print("\nPrediction Result")
print("-------------------")
print(f"Soil Moisture: {prediction:.2f}")
print(f"Status: {status}")
print(f"Irrigation Time: {irrigation}")
print(f"Water Needed: {water}")
print(f"Suitable Crops: {crop}")
print(f"Recommendation: {advice}")
print(f"Warning: {alert}")
