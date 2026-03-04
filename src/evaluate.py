import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error


# -----------------------------
# Load
# -----------------------------

model = joblib.load("model/model.pkl")

X_test = pd.read_csv("data/split/X_test.csv")
y_test = pd.read_csv("data/split/y_test.csv")


# -----------------------------
# Predict
# -----------------------------

pred = model.predict(X_test)


# -----------------------------
# Metrics
# -----------------------------

mse = mean_squared_error(y_test, pred)
rmse = mse ** 0.5

print("Evaluation RMSE:", rmse)


# Fail pipeline if bad
if rmse > 60:
    raise Exception("Model performance too low")
