import joblib
import pandas as pd
import sys


# -----------------------------
# Load model
# -----------------------------

model = joblib.load("model/model.pkl")


# -----------------------------
# Load history
# -----------------------------

history = pd.read_excel("data/Soil Moisture.xlsx")


# -----------------------------
# Input
# -----------------------------

if len(sys.argv) != 4:
    print("Usage: python predict.py <Temp> <Humidity> <Pump>")
    sys.exit(1)


temp = float(sys.argv[1])
hum = float(sys.argv[2])
pump = float(sys.argv[3])


# -----------------------------
# Feature building
# -----------------------------

last = history.iloc[-1]
prev = history.iloc[-2]


moisture_lag = last["Soil Moisture"]
moisture_delta = last["Soil Moisture"] - prev["Soil Moisture"]
moisture_mean = history["Soil Moisture"].tail(3).mean()

evapo = temp * (100 - hum) / 100


# -----------------------------
# Input frame
# -----------------------------

X = pd.DataFrame([[
    temp,
    hum,
    pump,
    moisture_lag,
    moisture_delta,
    evapo,
    moisture_mean
]], columns=[
    "Temperature",
    "Air Humidity",
    "Pump Data",
    "moisture_lag_1",
    "moisture_delta",
    "evapo_index",
    "moisture_mean_3"
])


# -----------------------------
# Predict
# -----------------------------

pred = model.predict(X)[0]


# -----------------------------
# Advice
# -----------------------------

if pred < 500:
    advice = "Turn Pump ON"
else:
    advice = "No Irrigation Needed"


# -----------------------------
# Output
# -----------------------------

print("\nPrediction Result")
print("-----------------")
print("Predicted Moisture:", round(pred, 2))
print("Recommendation:", advice)