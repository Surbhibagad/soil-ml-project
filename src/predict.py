import joblib
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


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
if not (18 <= temp <= 39):
    print("Temperature outside training range (18–39)")
    sys.exit(1)

if not (38 <= humidity <= 81):
    print("Humidity outside training range (38–81)")
    sys.exit(1)

if pump not in [0, 1]:
    print("Pump must be 0 or 1")
    sys.exit(1)


# Feature engineering
heat_dryness = (temp/40) * (1 - humidity/100)


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
# Interpretation
if prediction < 500:
    status = "Dry"
    advice = "Turn Pump ON"
    irrigation = "Immediately"
    water = "18 Liters"
    crop = "Millet, Barley"
    alert = "Low Moisture"

elif prediction < 750:
    status = "Moderate"
    advice = "Irrigation Recommended"
    irrigation = "After 6 Hours"
    water = "10 Liters"
    crop = "Wheat, Maize"
    alert = "Normal"

else:
    status = "Wet"
    advice = "No Irrigation Needed"
    irrigation = "After 24 Hours"
    water = "5 Liters"
    crop = "Rice, Sugarcane"
    alert = "High Moisture"


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
