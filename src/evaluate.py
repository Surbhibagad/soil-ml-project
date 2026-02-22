import sys
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Quality thresholds (adjust after testing)
MAX_MSE = 50.0
MIN_R2 = 0.75

# Load test data
X_test = pd.read_csv("data/split/X_test.csv")
y_test = pd.read_csv("data/split/y_test.csv").squeeze()

# Load model
model = joblib.load("model/model.pkl")

# Predict
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 Model Evaluation")
print(f"MSE: {mse:.4f}")
print(f"R2: {r2:.4f}")

# Quality gate
failed = False

if mse > MAX_MSE:
    print(f"❌ MSE too high: {mse:.2f} > {MAX_MSE}")
    failed = True

if r2 < MIN_R2:
    print(f"❌ R2 too low: {r2:.2f} < {MIN_R2}")
    failed = True

if failed:
    print("🚫 Model failed quality checks")
    sys.exit(1)

print("✅ Model passed quality checks")