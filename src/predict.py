import joblib
import pandas as pd
import sys


# Load model
try:
    model = joblib.load("model/model.pkl")
except FileNotFoundError:
    print("Model file not found. Train the model first.")
    sys.exit(1)
except Exception as e:
    print("Error loading model:", e)
    sys.exit(1)


# Validate input
if len(sys.argv) != 4:
    print("Usage: python predict.py <Temp> <Humidity> <Pump>")
    sys.exit(1)


# Convert inputs
try:
    temp = float(sys.argv[1])
    humidity = float(sys.argv[2])
    pump = float(sys.argv[3])
except ValueError:
    print("Inputs must be valid numbers")
    sys.exit(1)


# Input range check
if not (-20 <= temp <= 60):
    print("Temperature out of range")
    sys.exit(1)

if not (0 <= humidity <= 100):
    print("Humidity out of range")
    sys.exit(1)

if pump < 0:
    print("Pump must be positive")
    sys.exit(1)


# Prepare input
input_data = pd.DataFrame(
    [[temp, humidity, pump]],
    columns=["Temperature", "Air Humidity", "Pump Data"]
)


# Predict
prediction = model.predict(input_data)[0]


# Soil status
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
