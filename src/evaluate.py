import joblib
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score


# Load model
model = joblib.load("model/model.pkl")


# Load test data
X_test = pd.read_csv("data/split/X_test.csv")
y_test = pd.read_csv("data/split/y_test.csv").squeeze()


# Predict
pred = model.predict(X_test)


# Metrics
mse = mean_squared_error(y_test, pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, pred)


print("Evaluation Results")
print("------------------")
print("RMSE:", rmse)
print("R2 Score:", r2)
