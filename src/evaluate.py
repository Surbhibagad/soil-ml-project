import joblib
import pandas as pd
import sys

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# Load model and scaler
try:
    model = joblib.load("model/model.pkl")
    scaler = joblib.load("model/scaler.pkl")

except FileNotFoundError:
    print("Model or scaler not found. Train first.")
    sys.exit(1)

except Exception as e:
    print("Loading error:", e)
    sys.exit(1)


# Load test data
try:
    X_test = pd.read_csv("data/split/X_test.csv")
    y_test = pd.read_csv("data/split/y_test.csv")

except FileNotFoundError:
    print("Test data not found. Run train.py first.")
    sys.exit(1)


# Scale features
X_test_scaled = scaler.transform(X_test)


# Predict
y_pred = model.predict(X_test_scaled)


# Evaluation metrics
rmse = mean_squared_error(
    y_test,
    y_pred,
    squared=False
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


# Output
print("\nModel Evaluation Report")
print("------------------------")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"R2   : {r2:.2f}")


# Quality message
if r2 > 0.8:
    print("Model Quality: Excellent")

elif r2 > 0.6:
    print("Model Quality: Good")

elif r2 > 0.4:
    print("Model Quality: Average")

else:
    print("Model Quality: Poor (Needs Improvement)")
    