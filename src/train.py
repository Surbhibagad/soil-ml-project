import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
data = pd.read_excel("data/Soil Moisture.xlsx")

X = data.drop("moisture", axis=1)
y = data["moisture"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Train
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")