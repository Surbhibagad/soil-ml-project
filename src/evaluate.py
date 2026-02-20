import joblib
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

data = pd.read_excel("data/Soil Moisture.xlsx")

X = data.drop("moisture", axis=1)
y = data["moisture"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = joblib.load("model.pkl")

preds = model.predict(X_test)

score = r2_score(y_test, preds)

print("R2 Score:", score)

# Fail pipeline if performance is bad
if score < 0.7:
    raise Exception("Model performance too low!")