import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =========================
# Load model
# =========================
pipeline = joblib.load("model/full_pipeline.pkl")

# =========================
# Load test data
# =========================
X_test = pd.read_csv("data/split/X_test.csv")
y_test = pd.read_csv("data/split/y_test.csv").values.ravel()

# =========================
# Prediction
# =========================
y_pred = pipeline.predict(X_test)

# =========================
# Metrics
# =========================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# =========================
# Output
# =========================
print("\nModel Evaluation Report")
print("-----------------------")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"R2   : {r2:.2f}")

if r2 > 0.8:
    print("Model Quality: Excellent")
elif r2 > 0.6:
    print("Model Quality: Good")
elif r2 > 0.4:
    print("Model Quality: Average")
else:
    print("Model Quality: Poor")
